import os
import time
import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from rich import print

from simulator import Simulator
from q_network import DeepSetQ, QMixer, VDNMixer

# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
L = 5000                  # Number of episodes to train for
K = 64                    # Minibatch size
M = 100                   # Number of steps per episode
E = 0.90                  # Initial epsilon
S = 30000                 # Experience replay buffer size
EPSILON_DECAY_FACTOR = 0.999
MIN_EPSILON = 0.05
GAMMA = 0.99
TAU = 0.005               # Polyak averaging factor
LR = 1e-4

# Environment / observation constants
SECTORS_PER_BS = 3
NUM_SLEEP_MODES = 4
NUM_ACTIONS = SECTORS_PER_BS * NUM_SLEEP_MODES   # (which own sector) x (which sleep mode)
NUM_NEIGHBOR_BS = 6        # own BS + 5 nearest -> 18 sectors per neighbourhood
MAX_SET_SIZE = 96         # Max UEs per agent set (padded / truncated)
SLA_MBPS = 8.0
MAX_COUNTDOWN = 15.0      # longest sleep-mode lock in SleepModeManager (5 + 10 steps)
ALPHA, BETA = 1.0, 1.7    # Reward weights (energy, SLA)

assert K >= 2, "Minibatch size must be greater than 1"

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_EYE_SLEEP = np.eye(NUM_SLEEP_MODES, dtype=np.float32)


# ----------------------------------------------------------------------------
# Topology: per-agent neighbourhoods
# ----------------------------------------------------------------------------
def build_neighborhoods(simulator: Simulator) -> np.ndarray:
    """
    Neighbourhood of agent a (base station a) = its own BS plus the NUM_NEIGHBOR_BS-1 nearest BSs
    (by horizontal distance), i.e. 6 BSs x 3 sectors = 18 sectors. The agent's own sectors come first,
    so local index 0..2 always means "own sector".

    This is computed from base-station positions rather than read from
    sector_manager.neighboring_sectors_indices_matrix: VisibilityBasedNetworkTopologyHelper (what the
    Simulator uses) never populates that matrix, so it stays all zeros.
    Returns (num_agents, N) int64. Computed once; BS positions are fixed for the run.
    """
    A = simulator.num_base_stations
    assert A >= NUM_NEIGHBOR_BS, f"Need at least {NUM_NEIGHBOR_BS} base stations, got {A}"
    pos = np.asarray(simulator.base_station_manager.get_all_positions(), dtype=np.float64)[:, :2]
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    nbrs = np.zeros((A, NUM_NEIGHBOR_BS * SECTORS_PER_BS), dtype=np.int64)
    for a in range(A):
        order = np.argsort(dist[a], kind="stable")
        order = np.concatenate([[a], order[order != a]])[:NUM_NEIGHBOR_BS]      # own BS first
        nbrs[a] = (order[:, None] * SECTORS_PER_BS + np.arange(SECTORS_PER_BS)).ravel()
    return nbrs


# ----------------------------------------------------------------------------
# Observation / state / reward construction
# ----------------------------------------------------------------------------
def build_observations(simulator: Simulator, nbrs: np.ndarray, elapsed_ms: int, t_idx: int) -> dict:
    """
    For ALL agents at once:
      feats  (A, MAX_SET_SIZE, F)  per-UE KPI vectors
      counts (A,)                  number of valid UEs (rest is padding)
      ctx    (A, 7N+1)             per neighbour sector: sleep one-hot(4), countdown, PRB util, UE count; + time
      state  (S*10+1,)             global per-sector summary for the mixer

    A UE is in agent a's set if its SERVING sector or its STRONGEST sector (ignoring sleep) lies in
    a's neighbourhood. Handover skips sleeping sectors, so the second condition is what keeps UEs
    displaced by a sleeping sector visible to the agent that put it to sleep.
    """
    rcm = simulator.radio_channel_model
    rx = rcm.received_power_dbm_matrix_per_resource_element
    snr = rcm.snr_dbm_matrix_per_slot
    se = rcm.spectral_efficiency_matrix
    num_sectors, num_devices = rx.shape
    A, N = nbrs.shape
    dev = np.arange(num_devices)

    sleep = np.asarray(simulator.sleep_mode_manager.sector_sleep_mode_matrix).astype(np.int64)
    countdown = np.asarray(simulator.sleep_mode_manager.sector_sleep_mode_countdown, dtype=np.float32) / MAX_COUNTDOWN
    util = np.asarray(simulator.sector_manager.sector_physical_resource_block_utilization, dtype=np.float32)

    serving = simulator.handover_manager.get_serving_sector_indices()   # actual (awake) serving sector
    best = np.argmax(rx, axis=0)                                        # strongest sector, sleeping or not

    ue_rx, ue_snr, ue_se = rx[serving, dev], snr[serving, dev], se[serving, dev]
    gap = np.clip((rx[best, dev] - ue_rx) / 30.0, 0.0, 3.0)             # RSRP lost by not using the best sector

    tput = np.asarray(simulator.kpi_handler.calculate_throughput_mbps(elapsed_ms))
    tput_norm = np.minimum(tput / SLA_MBPS, 2.0)
    sla_ok = (tput >= SLA_MBPS).astype(np.float32)

    tg = simulator.traffic_generator
    demand = np.clip(tg.device_downlink_bits_matrix[:, t_idx] / max(getattr(tg, "high_rate", 1.0), 1.0), 0.0, 1.0)

    base = np.column_stack([
        (ue_rx + 120.0) / 30.0, ue_snr / 35.0, ue_se / 7.4,
        tput_norm, sla_ok, demand, gap,
        _EYE_SLEEP[sleep[best]],                                        # sleep mode of the strongest sector (4)
    ]).astype(np.float32)                                               # F0 = 11

    F = base.shape[1] + 2 + 2 * N
    feats = np.zeros((A, MAX_SET_SIZE, F), dtype=np.float32)
    counts = np.zeros(A, dtype=np.int16)
    ctx = np.zeros((A, 7 * N + 1), dtype=np.float32)
    eye_ext = np.vstack([np.eye(N, dtype=np.float32), np.zeros((1, N), dtype=np.float32)])  # row N = "not in neighbourhood"
    local = np.full(num_sectors, -1, dtype=np.int64)
    time_feat = t_idx / float(simulator.simulation_length_ms)

    for a in range(A):
        local[:] = -1
        local[nbrs[a]] = np.arange(N)
        sl, bl = local[serving], local[best]
        members = np.flatnonzero((sl >= 0) | (bl >= 0))
        per_sector_counts = np.bincount(sl[sl >= 0], minlength=N)

        if len(members) > MAX_SET_SIZE:                                 # keep the UEs in the worst shape
            members = members[np.argsort(tput[members])[:MAX_SET_SIZE]]
        n = len(members)
        s_m, b_m = sl[members], bl[members]

        serving_is_own = ((s_m >= 0) & (s_m < SECTORS_PER_BS)).astype(np.float32)[:, None]
        best_is_own = ((b_m >= 0) & (b_m < SECTORS_PER_BS)).astype(np.float32)[:, None]
        feats[a, :n] = np.concatenate([
            base[members], serving_is_own, best_is_own,
            eye_ext[np.where(s_m < 0, N, s_m)], eye_ext[np.where(b_m < 0, N, b_m)],
        ], axis=1)
        counts[a] = n

        idx = nbrs[a]
        ctx[a] = np.concatenate([
            _EYE_SLEEP[sleep[idx]].ravel(), countdown[idx], util[idx],
            per_sector_counts / 10.0, [time_feat],
        ])

    # Global state: 10 values per sector (+ time)
    n_per = np.bincount(serving, minlength=num_sectors).astype(np.float32)
    denom = np.maximum(n_per, 1.0)

    def sector_mean(v):
        return np.bincount(serving, weights=v, minlength=num_sectors) / denom

    state = np.concatenate([
        np.column_stack([
            _EYE_SLEEP[sleep], countdown, util, n_per / 10.0,
            sector_mean(tput_norm), sector_mean(sla_ok), sector_mean(ue_snr / 35.0),
        ]).astype(np.float32).ravel(),
        [time_feat],
    ]).astype(np.float32)

    return {"feats": feats, "counts": counts, "ctx": ctx, "state": state}


def team_reward(simulator: Simulator, elapsed_ms: int) -> float:
    """Single shared team reward (QMIX/VDN learn from one team reward, not one per agent)."""
    sleep = np.asarray(simulator.sleep_mode_manager.sector_sleep_mode_matrix)
    tput = np.asarray(simulator.kpi_handler.calculate_throughput_mbps(elapsed_ms))
    r_sleep = ALPHA * (
        -3 * np.mean(sleep == 0) + 1 * np.mean(sleep == 1)
        + 3 * np.mean(sleep == 2) + 6 * np.mean(sleep == 3)
    )
    r_sla = BETA * 6 * np.mean(tput >= SLA_MBPS)
    return float(r_sleep + r_sla)


def warm_start(simulator: Simulator):
    """
    reset() zeroes the radio/geometry/association matrices, so before the first step every UE would look
    like it is served by sector 0 with zero power. Run one step to populate them, then discard the KPIs
    that step accumulated so the episode's throughput starts from zero.
    """
    simulator.step(0)
    simulator.kpi_handler.total_transmitted_bits_per_device[:] = 0.0


# ----------------------------------------------------------------------------
# Prioritised replay (SumTree backed by memmaps)
# ----------------------------------------------------------------------------
class SumTree:
    def __init__(self, capacity, specs, directory, rng=None):
        os.makedirs(directory, exist_ok=True)
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = {
            name: np.memmap(f"{directory}/{name}.dat", dtype=dtype, mode="w+", shape=(capacity,) + shape)
            for name, (shape, dtype) in specs.items()
        }
        self.write = 0
        self.n_entries = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    def _propagate(self, idx, change):
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1

    def total(self):
        return self.tree[0]

    def add(self, p, sample: dict):
        for name, arr in self.data.items():
            arr[self.write] = sample[name]
        self.update(self.write + self.capacity - 1, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(s)
        data_idx = min(idx - self.capacity + 1, self.n_entries - 1)
        tree_idx = data_idx + self.capacity - 1
        return tree_idx, self.tree[tree_idx], data_idx

    def flush(self):
        for arr in self.data.values():
            arr.flush()


class Memory:
    e = 0.01
    a = 0.9
    beta = 1.0   # starts at 1 as in the original; lower (e.g. 0.4) to anneal upward
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, specs, directory, rng=None):
        self.tree = SumTree(capacity, specs, directory, rng)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        assert not np.isnan(error), "NaN error detected."
        self.tree.add(self._get_priority(error), sample)

    def sample(self, n):
        segment = self.tree.total() / n
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        idxs, data_idxs, priorities = [], [], []
        for i in range(n):
            s = self.tree.rng.uniform(segment * i, segment * (i + 1))
            idx, p, data_idx = self.tree.get(s)
            idxs.append(idx); data_idxs.append(data_idx); priorities.append(p)

        probs = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * probs, -self.beta)
        is_weight /= is_weight.max()

        batch = {name: np.asarray(arr[data_idxs]) for name, arr in self.tree.data.items()}
        return batch, idxs, is_weight

    def update(self, idx, error):
        self.tree.update(idx, self._get_priority(error))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_agent_tensors(feats, counts, ctx):
    """numpy (B, A, M, F), (B, A), (B, A, C) -> flat per-agent tensors (B*A, ...)."""
    B, A, Mx, F = feats.shape
    f = torch.as_tensor(feats, dtype=torch.float32, device=gpu).reshape(B * A, Mx, F)
    c = torch.as_tensor(counts.astype(np.int64), device=gpu).reshape(B * A)
    mask = torch.arange(Mx, device=gpu)[None, :] < c[:, None]
    x = torch.as_tensor(ctx, dtype=torch.float32, device=gpu).reshape(B * A, -1)
    return f, mask, x


def select_actions(q_net, obs, epsilon, rng):
    A = obs["feats"].shape[0]
    with torch.no_grad():
        f, mask, x = to_agent_tensors(obs["feats"][None], obs["counts"][None], obs["ctx"][None])
        greedy = q_net(f, mask, x).argmax(dim=1).cpu().numpy()
    explore = rng.uniform(0, 1, size=A) <= epsilon
    return np.where(explore, rng.integers(0, NUM_ACTIONS, size=A), greedy).astype(np.int32)


def polyak_update(target, online):
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - TAU).add_(TAU * op.data)


# ----------------------------------------------------------------------------
# Training (shared by QMIX and VDN)
# ----------------------------------------------------------------------------
def train(mixer_type: str = "qmix", dueling: bool = True):
    assert mixer_type in ("qmix", "vdn", "iql"), "mixer_type must be 'qmix', 'vdn' or 'iql'"
    run_name = mixer_type + ("" if dueling else "_nodueling")   # keeps ablation outputs separate
    initial_seed = int(time.time())
    simulator = Simulator(19, 200, M, seed=initial_seed)
    nbrs = build_neighborhoods(simulator)
    warm_start(simulator)

    obs = build_observations(simulator, nbrs, 0, 0)
    A, _, F = obs["feats"].shape
    C = obs["ctx"].shape[1]
    SD = obs["state"].shape[0]
    print(f"[{run_name}] dueling={dueling} agents={A} ue_feat_dim={F} context_dim={C} state_dim={SD}")

    q_net = DeepSetQ(F, C, NUM_ACTIONS, dueling=dueling)
    target_q_net = deepcopy(q_net)
    mixer = QMixer(A, SD) if mixer_type == "qmix" else VDNMixer()
    target_mixer = deepcopy(mixer)

    params = list(q_net.parameters()) + list(mixer.parameters())
    optimizer = torch.optim.AdamW(params=params, lr=LR)
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    f16, f32, i32, i16 = np.float16, np.float32, np.int32, np.int16
    specs = {
        "feats": ((A, MAX_SET_SIZE, F), f16), "counts": ((A,), i16), "ctx": ((A, C), f16), "state": ((SD,), f16),
        "next_feats": ((A, MAX_SET_SIZE, F), f16), "next_counts": ((A,), i16),
        "next_ctx": ((A, C), f16), "next_state": ((SD,), f16),
        "actions": ((A,), i32), "reward": ((), f32), "done": ((), f32),
    }
    memory = Memory(S, specs, f"../sumtree_{run_name}", np.random.default_rng(initial_seed))

    kpi_list, columns = [], ['episode', 'loss', 'reward', 'epsilon']
    epsilon = E

    for episode in range(1, L + 1):
        print(f"Starting episode: {episode}!")
        for step in range(M):
            # ---- act ----
            actions = select_actions(q_net, obs, epsilon, simulator.rng)
            for agent in range(A):
                sector_id = int(agent * SECTORS_PER_BS + actions[agent] // NUM_SLEEP_MODES)
                sleep_mode_id = int(actions[agent] % NUM_SLEEP_MODES)
                simulator.sleep_mode_manager.set_sleep_mode(sector_id, sleep_mode_id, simulator.sector_manager)

            simulator.step(step)

            # ---- observe ----
            # Observation for step s+1 == next observation of step s (reused below).
            next_obs = build_observations(simulator, nbrs, step + 1, min(step + 1, M - 1))
            reward = team_reward(simulator, step + 1)
            done = float(step == M - 1)

            memory.add(error=1e2, sample={
                "feats": obs["feats"], "counts": obs["counts"], "ctx": obs["ctx"], "state": obs["state"],
                "next_feats": next_obs["feats"], "next_counts": next_obs["counts"],
                "next_ctx": next_obs["ctx"], "next_state": next_obs["state"],
                "actions": actions, "reward": reward, "done": done,
            })
            obs = next_obs

            # ---- learn ----
            if memory.tree.n_entries >= K:
                b, idxs, is_weight = memory.sample(K)
                is_w = torch.as_tensor(is_weight, dtype=torch.float32, device=gpu).view(K, 1)

                f, mask, x = to_agent_tensors(b["feats"], b["counts"], b["ctx"])
                nf, nmask, nx = to_agent_tensors(b["next_feats"], b["next_counts"], b["next_ctx"])
                states = torch.as_tensor(b["state"], dtype=torch.float32, device=gpu)
                next_states = torch.as_tensor(b["next_state"], dtype=torch.float32, device=gpu)
                acts = torch.as_tensor(b["actions"].astype(np.int64), device=gpu).view(K * A, 1)
                rewards = torch.as_tensor(b["reward"], dtype=torch.float32, device=gpu).view(K, 1)
                dones = torch.as_tensor(b["done"], dtype=torch.float32, device=gpu).view(K, 1)

                chosen_q = q_net(f, mask, x).gather(1, acts).view(K, A)

                with torch.no_grad():
                    # Double-DQN: online net picks, target net evaluates
                    best = q_net(nf, nmask, nx).argmax(dim=1, keepdim=True)
                    next_q = target_q_net(nf, nmask, nx).gather(1, best).view(K, A)
                    if mixer_type == "iql":
                        # Each agent bootstraps from its own next Q-value; the shared team reward
                        # broadcasts over agents: (K,1) + (K,A) -> (K,A)
                        target = rewards + GAMMA * (1.0 - dones) * next_q
                    else:
                        total_next_q = target_mixer(next_q, next_states)
                        target = rewards + GAMMA * (1.0 - dones) * total_next_q

                # IQL: per-agent Q vs per-agent target. VDN/QMIX: mixed Q_tot vs team target.
                pred = chosen_q if mixer_type == "iql" else mixer(chosen_q, states)

                # Reduce to one value per replay sample (mean over agents for IQL; identity otherwise)
                td = (pred - target).abs().reshape(K, -1).mean(dim=1)
                per_sample_loss = loss_fn(pred, target).reshape(K, -1).mean(dim=1, keepdim=True)
                loss = (is_w * per_sample_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                for i, p in zip(idxs, td.detach().cpu().numpy().ravel()):
                    memory.update(i, p)

                polyak_update(target_q_net, q_net)
                polyak_update(target_mixer, mixer)

                kpi_list.append((episode, loss.item(), reward, epsilon))

        print("Ended episode!")
        epsilon = max(epsilon * EPSILON_DECAY_FACTOR, MIN_EPSILON)
        print(f"Restarting simulator with seed {initial_seed + episode}")
        simulator.reset(initial_seed + episode)
        warm_start(simulator)
        obs = build_observations(simulator, nbrs, 0, 0)

        if episode % 20 == 0:
            try:
                torch.save({
                    "q_net": q_net.state_dict(), "mixer": mixer.state_dict(),
                    "dims": {"ue_feat_dim": F, "context_dim": C, "state_dim": SD,
                             "num_agents": A, "num_actions": NUM_ACTIONS, "dueling": dueling},
                }, f"{run_name}_checkpoint.pth")
                print("Networks saved!")
            except Exception:
                print("Error saving networks:")
                traceback.print_exc()

        if episode % 10 == 0:
            try:
                memory.tree.flush()
                print("SumTree flushed!")
            except Exception:
                print("Error flushing SumTree:")
                traceback.print_exc()

        if episode % 5 == 0:
            print("Saving KPIs!")
            pd.DataFrame(kpi_list, columns=columns).to_csv(f'training_log_{run_name}.csv', index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", nargs="?", default="qmix", choices=["qmix", "vdn", "iql"])
    parser.add_argument("--no-dueling", action="store_true", help="ablation: plain Q-head instead of dueling")
    args = parser.parse_args()
    train(args.algo, dueling=not args.no_dueling)