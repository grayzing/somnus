import numpy as np
import pandas as pd
from simulator import Simulator
from q_network import Q, MixingNetwork
from collections import deque
from copy import deepcopy
from rich import print
from numba import njit, prange
import random
import torch
import time
import traceback

# Hyperparameters
L = 5000 # Number of episodes to train for
K = 64 # Minibatch size
M = 100 # Number of steps per episode
E = 0.90 # Initial epsilon
S = 30000 # Experience replay buffer size
EPSILON_DECAY_FACTOR = 0.999 # Epsilon decay factor
MIN_EPSILON = 0.05 # Minimum epsilon
GAMMA = 0.99 # Discount factor
TAU = 0.005 # Polyak averaging factor

assert K >= 2, "Minibatch size must be greater than 1"

# Filesystem stuff
base_dir = "../sumtree"

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

class SumTree:
    write = 0

    def __init__(self, capacity, rng=np.random.default_rng()):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = {
            "observations": np.memmap(f'{base_dir}/observations.dat', dtype=np.float16, mode='w+', shape=(capacity,19,7218)),
            "actions": np.memmap(f'{base_dir}/actions.dat', dtype=np.int32, mode='w+', shape=(capacity,19,1)),
            "rewards": np.memmap(f'{base_dir}/rewards.dat', dtype=np.float16, mode='w+', shape=(capacity,19,1)),
            "next_observations": np.memmap(f'{base_dir}/next_observations.dat', dtype=np.float16, mode='w+', shape=(capacity,19,7218))
        }
        self.n_entries = 0
        self.rng = rng

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data["observations"][self.write] = data["observations"]
        self.data["actions"][self.write] = data["actions"]
        self.data["rewards"][self.write] = data["rewards"]
        self.data["next_observations"][self.write] = data["next_observations"]
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        # Modify with the O, A, R, O'
        return (idx, self.tree[idx], self.data["observations"][dataIdx].copy(), self.data["actions"][dataIdx].copy(), self.data["rewards"][dataIdx].copy(), self.data["next_observations"][dataIdx].copy())

    # write data
    def flush(self):
        self.data["observations"].flush()
        self.data["actions"].flush()
        self.data["rewards"].flush()
        self.data["next_observations"].flush()

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.9
    beta = 1
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        assert not np.isnan(error), "NaN error detected."
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        samples = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = self.tree.rng.uniform(a, b)
            (idx, p, data_observations, data_actions, data_rewards, data_next_observations) = self.tree.get(s)
            priorities.append(p)
            samples.append(
                (
                    np.array(data_observations),
                    np.array(data_actions),
                    np.array(data_rewards),
                    np.array(data_next_observations)
                )
            )

            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        observations, actions, rewards, next_observations = zip(*samples)
        #print(observations)

        return (
            observations,
            actions,
            rewards,
            next_observations,
            idxs,
            is_weight
        )

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

def take_observation(agent_id: int, simulator: Simulator) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(
        [np.stack(
            [
                simulator.radio_channel_model.received_power_dbm_matrix_per_resource_element[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]],
                simulator.radio_channel_model.snr_dbm_matrix_per_slot[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]]
            ],
            axis=0
            ).flatten(),
            simulator.sleep_mode_manager.sector_sleep_mode_matrix[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]]
        ]
    )).to(torch.float32).to(gpu)

def epsilon_greedy(q_net: Q, observation: torch.Tensor, epsilon: float, rng=np.random.default_rng()) -> torch.Tensor:
    epsilon_greedy_random_variable = rng.uniform(0,1)
    if epsilon_greedy_random_variable <= epsilon:
        # Take a random action
        return torch.tensor(rng.integers(low=0,high=12))
    # Otherwise, take the Q estimate
    argmax_q = torch.argmax(q_net(observation))
    return argmax_q

if __name__ == "__main__":
    kpi_list = []
    columns = ['episode', 'loss', 'reward', 'epsilon']
    memory: Memory = Memory(S)
    epsilon = E
    initial_seed = int(time.time())
    q_net: Q = Q()
    target_q_net: Q = deepcopy(q_net)
    mixing_net: MixingNetwork = MixingNetwork()
    simulator: Simulator = Simulator(19, 200, M, seed=initial_seed)
    simulator.step(0)
    optimizer = torch.optim.AdamW(params=q_net.parameters(), lr=1e-4)
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    for episode in range(1, L):
        print(f"Starting episode: {episode}!")
        for step in range(M):
            # Get Q for each agent
            total_observations = []
            total_actions = []
            total_rewards = []
            total_next_observations = []
            for agent in range(0,simulator.num_base_stations):
                observation_it = take_observation(agent, simulator)
                action_it = epsilon_greedy(q_net, observation_it, epsilon, simulator.rng)
                total_observations.append(observation_it)
                total_actions.append(action_it)

                # Now take the action specified
                sector_id = int(agent * 3 + (action_it // 4))
                sleep_mode_id = int(action_it % 4)
                simulator.sleep_mode_manager.set_sleep_mode(sector_id, sleep_mode_id, simulator.sector_manager)
        
            simulator.step(step)
            for agent in range(0, simulator.num_base_stations):
                next_observation_it = take_observation(agent, simulator)
                # Reward calculations
                alpha = 1
                beta = 1.7

                ratio_of_active_sectors = np.count_nonzero(simulator.sleep_mode_manager.sector_sleep_mode_matrix == 0) / simulator.num_base_stations
                ratio_of_sm1_sectors = np.count_nonzero(simulator.sleep_mode_manager.sector_sleep_mode_matrix == 1) / simulator.num_base_stations
                ratio_of_sm2_sectors = np.count_nonzero(simulator.sleep_mode_manager.sector_sleep_mode_matrix == 2) / simulator.num_base_stations
                ratio_of_sm3_sectors = np.count_nonzero(simulator.sleep_mode_manager.sector_sleep_mode_matrix == 3) / simulator.num_base_stations

                ratio_of_sla_acceptable_devices = np.count_nonzero(simulator.kpi_handler.calculate_throughput_mbps(step) >= 8.0) / simulator.num_base_stations

                reward_sleep = alpha * -3 * ratio_of_active_sectors + ratio_of_sm1_sectors + 3 * ratio_of_sm2_sectors + 6 * ratio_of_sm3_sectors
                reward_sla = beta * 6 * ratio_of_sla_acceptable_devices
                reward_it = reward_sleep + reward_sla

                total_next_observations.append(next_observation_it)
                total_rewards.append(torch.tensor(reward_it))

            total_observations = torch.vstack(total_observations).to(cpu)
            total_actions = torch.vstack(total_actions).to(cpu)
            total_rewards = torch.vstack(total_rewards).to(cpu)
            total_next_observations = torch.vstack(total_next_observations).to(cpu)
            # Placeholder error, change this!!
            memory.add(
                sample={
                    "observations": total_observations.detach().cpu().numpy(),
                    "rewards": total_rewards.detach().cpu().numpy(),
                    "actions": total_actions.detach().cpu().numpy(),
                    "next_observations": total_next_observations.detach().cpu().numpy()
                },
                error=1e2
            )

            if memory.tree.n_entries >= K:
                observations, actions, rewards, next_observations, idxs, is_weight = memory.sample(K)
                #print(idxs)
                #print(observations)

                # Convert the data

                is_weight_tensor = torch.FloatTensor(is_weight).to(gpu)
                batched_observations = torch.as_tensor(np.stack(observations).astype(np.float16)).to(gpu)
                batched_actions = torch.as_tensor(np.stack(actions).astype(np.int32)).to(gpu)
                batched_rewards = torch.as_tensor(np.stack(rewards).astype(np.float16)).to(gpu)
                batched_next_observations = torch.as_tensor(np.stack(next_observations).astype(np.float16)).to(gpu)

                #print(batched_observations[0, 0, :10])
                #print(batched_observations[1, 0, :10])

                #print(batched_observations.shape)
                #print(batched_actions.shape)
                #print(batched_rewards.shape)
                #print(batched_next_observations.shape)

                batched_observations = batched_observations.float()
                batched_next_observations = batched_next_observations.float()
                batched_actions = batched_actions
                batched_rewards = batched_rewards

                K, num_agents, obs_dim = batched_observations.shape
                q_values = q_net(batched_observations.view(K * num_agents, -1))# Shape: (K * num_agents, 12)

                gathered_q = torch.gather(q_values, dim=1, index=batched_actions.view(K * num_agents, -1))
                gathered_q = gathered_q.view(K, num_agents)
                total_q = mixing_net(gathered_q)

                with torch.no_grad():
                    # Double deep Q target
                    argmax_next_q_values = torch.argmax(q_net(batched_next_observations.view(K * num_agents, -1)),dim=1)
                    # print(argmax_next_q_values.shape)
                    next_q_values = target_q_net(batched_next_observations.view(K * num_agents, -1))
                    # print(next_q_values.shape)
                    action_next_q = torch.gather(next_q_values, dim=1, index=argmax_next_q_values.view(K * num_agents, -1))
                    # print(action_next_q.shape)
                    action_next_q = action_next_q.view(K, num_agents)
                    total_next_q = mixing_net(action_next_q) # Shape: (K, 1)
                    #print(f"Total next q: {total_next_q}")

                batch_reward_sum = batched_rewards.sum(dim=1)
                target = batch_reward_sum + GAMMA * total_next_q
                #print(target.shape)
                #print(f"Total q shape: {total_q.shape}")
                
                loss = loss_fn(total_q, target)
                #print(loss.shape)
                batch_priorities = torch.abs(total_q - target).squeeze().detach().cpu().numpy()
                #print(batch_priorities)
                # Update the priority
                for i in prange(K):
                    idx = idxs[i]
                    memory.update(idx, batch_priorities[i])
                    #print(batch_priorities[i])

                loss_for_backprop = (is_weight_tensor*loss).mean()
                optimizer.zero_grad()
                loss_for_backprop.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0) # for staiblity
                optimizer.step()

                kpi_list.append((episode, loss_for_backprop.item(), batch_reward_sum.squeeze().mean().item(), epsilon))

            # Polyak averaging
            with torch.no_grad():
                for target_param, online_param in zip(target_q_net.parameters(), q_net.parameters()):
                    target_param.data.mul_(1.0 - TAU)
                    target_param.data.add_(TAU * online_param.data)
        print(f"Ended episode!")
        epsilon = max(epsilon * EPSILON_DECAY_FACTOR, MIN_EPSILON)
        print(f"Restarting simulator with seed {initial_seed + episode}")
        simulator.reset(initial_seed + episode)

        # Saving some stuff
        if episode % 20 == 0:
            print("Attemping to save Q network...")
            try:
                torch.save(q_net, "q_net.pth")
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
            finally:
                print("Q network successfully saved!")

        if episode % 10 == 0:
            print("Attemping to flush the SumTree memmap...")
            try:
                memory.tree.flush()
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
            finally:
                print("SumTree successfully flushed!")

        if episode % 5 == 0:
            print("Saving KPIs!")
            df = pd.DataFrame(kpi_list, columns=columns)
            df.to_csv('training_log.csv', index=False)