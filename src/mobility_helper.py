import numpy as np
from abc import ABC, abstractmethod
from device_manager import DeviceManager


class MobilityHelper(ABC):
    @abstractmethod
    def step_devices(self, device_manager: DeviceManager) -> None:
        pass


class CommunityBasedMobilityHelper(MobilityHelper):
    """
    Time-varying community-based UE mobility model from Masrur et al.,
    "Energy-Efficient Sleep Mode Optimization in 5G mmWave Networks via
    Multi-Agent Deep Reinforcement Learning" (arXiv:2511.22105v1), Sec. III-B,
    following Hsu et al. (INFOCOM 2007).

    Timeline per episode: NMP (Tn) -> CMP (Tc), repeating. Communities are
    redefined at every episode (NMP) start. Within each period every UE goes
    through NE epochs; durations ~ Exp(mean_epoch), normalized to the period
    length. At each epoch boundary the UE redraws speed ~ U[vmin, vmax] and
    heading ~ U[0, 2*pi) and switches local <-> roaming according to a
    two-state Markov chain with period-dependent probabilities. Roaming UEs
    are reflected at the service-area bounding box; local UEs at their
    community circle (specular reflection).
    """

    ROAMING, LOCAL = 0, 1

    def __init__(self,
                 service_area_bounds,
                 service_area_grid_xy=None,
                 num_communities: int = 7,        # paper: C = 7
                 area_nmp_m2: float = 500.0,      # paper: An = 500 m^2
                 area_cmp_m2: float = 250.0,      # paper: Ac = 250 m^2
                 nmp_duration_s: float = 3600.0,  # paper: Tn = 3600 s
                 cmp_duration_s: float = 1800.0,  # paper: Tc = 1800 s
                 num_epochs: int = 10,            # paper: NE = 10
                 mean_epoch_s: float = 340.0,     # paper: E-bar = 340 s
                 p_local_nmp: float = 0.5,        # pl,n (paper adopts from [31])
                 p_local_cmp: float = 0.5,        # pl,c
                 p_roam: float = 0.5,             # pr
                 vmin_mps: float = 5.0,           # paper: vmin = vmax = 5 m/s
                 vmax_mps: float = 5.0,
                 dt_s: float = 0.01,              # stepped once per 10 ms
                 seed: int = 24) -> None:
        self.bounds = np.asarray(service_area_bounds, dtype=np.float64)  # minx, miny, maxx, maxy
        self.grid_xy = None if service_area_grid_xy is None else np.asarray(service_area_grid_xy)[:, :2]
        self.C = num_communities
        self.area_n, self.area_c = area_nmp_m2, area_cmp_m2
        self.t_n, self.t_c = nmp_duration_s, cmp_duration_s
        self.ne = num_epochs
        self.mean_epoch = mean_epoch_s
        self.p_local = (p_local_nmp, p_local_cmp)
        self.p_roam = p_roam
        self.vmin, self.vmax = vmin_mps, vmax_mps
        self.dt = dt_s
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self, device_manager: DeviceManager) -> None:
        """Start a new episode: redefine communities and randomize UE state."""
        self.num_devices = device_manager.get_all_positions().shape[0]
        self.sim_time_s = 0.0
        self.current_period_idx = 0
        self.mode = self.rng.integers(0, 2, self.num_devices)   # initial epoch mode
        self._define_communities()
        self._start_period(0, initial=True)

        pos = np.empty((self.num_devices, 3), dtype=np.float64)
        roam = self.mode == CommunityBasedMobilityHelper.ROAMING
        pos[roam, :2] = self.rng.uniform(self.bounds[[0, 1]], self.bounds[[2, 3]],
                                         size=(int(roam.sum()), 2))
        u = self.rng.uniform(0.0, 1.0, size=(int((~roam).sum()), 1))
        a = self.rng.uniform(0.0, 2 * np.pi, size=(int((~roam).sum()), 1))
        c = self.device_center[~roam]
        pos[~roam, :2] = c + self.current_radius * np.sqrt(u) * np.hstack([np.cos(a), np.sin(a)])
        pos[:, 2] = 1.5   # paper: UE height 1.5 m

        device_manager.device_position_matrix = pos.astype(np.float32)
        vel = self._velocity_vectors()
        device_manager.device_velocity_matrix = np.column_stack([vel * self.dt, np.zeros(self.num_devices)]).astype(np.float32)

    def _define_communities(self) -> None:
        """C random SA grid points as community centers; UEs evenly assigned."""
        if self.grid_xy is not None and len(self.grid_xy) >= self.C:
            idx = self.rng.choice(len(self.grid_xy), self.C, replace=False)
            self.community_centers = self.grid_xy[idx]
        else:
            self.community_centers = self.rng.uniform(self.bounds[[0, 1]], self.bounds[[2, 3]],
                                                      size=(self.C, 2))
        counts = np.full(self.C, self.num_devices // self.C)
        counts[: self.num_devices % self.C] += 1
        assignment = np.concatenate([np.full(n, k, dtype=np.int64) for k, n in enumerate(counts)])
        self.rng.shuffle(assignment)
        self.community_of_device = assignment
        self.device_center = self.community_centers[assignment]

    def _start_period(self, period_idx: int, initial: bool = False) -> None:
        """Enter period `period_idx`: redraw epoch schedule for every UE."""
        is_nmp = (period_idx % 2 == 0)
        period_duration = self.t_n if is_nmp else self.t_c
        self.current_radius = np.sqrt((self.area_n if is_nmp else self.area_c) / np.pi)
        self.period_start_s = self.sim_time_s
        if is_nmp:
            self._define_communities()      # episode start -> communities redefined

        dur = self.rng.exponential(self.mean_epoch, size=(self.num_devices, self.ne))
        dur *= period_duration / dur.sum(axis=1, keepdims=True)   # normalized to Tn/Tc
        self.epoch_durations = dur
        self.epoch_idx = np.zeros(self.num_devices, dtype=np.int64)
        self._draw_kinematics(np.arange(self.num_devices))
        if initial:
            # random phase within the first epoch so transitions aren't synchronous
            self.epoch_end_time = self.rng.uniform(0.0, dur[:, 0])
        else:
            self.epoch_end_time = dur[:, 0].copy()

    def _draw_kinematics(self, idxs: np.ndarray) -> None:
        k = idxs.size
        if k == 0:
            return
        if not hasattr(self, 'speed'):
            self.speed = np.zeros(self.num_devices)
            self.angle = np.zeros(self.num_devices)
        self.speed[idxs] = self.rng.uniform(self.vmin, self.vmax, k)
        self.angle[idxs] = self.rng.uniform(0.0, 2 * np.pi, k)

    def _advance_epochs(self, idxs: np.ndarray) -> None:
        """UEs in `idxs` hit an epoch boundary: transition mode, redraw speed/heading."""
        k = idxs.size
        self.epoch_idx[idxs] += 1
        ei = np.minimum(self.epoch_idx[idxs], self.ne - 1)   # clamp (float-safety wrap)
        self.epoch_end_time[idxs] += self.epoch_durations[idxs, ei]

        pl = self.p_local[self.current_period_idx % 2]
        m = self.mode[idxs]
        to_local = (m == CommunityBasedMobilityHelper.ROAMING) & (self.rng.random(k) < pl)
        to_roam = (m == CommunityBasedMobilityHelper.LOCAL) & (self.rng.random(k) < self.p_roam)
        self.mode[idxs[to_local]] = CommunityBasedMobilityHelper.LOCAL
        self.mode[idxs[to_roam]] = CommunityBasedMobilityHelper.ROAMING
        self._draw_kinematics(idxs)

    def _velocity_vectors(self) -> np.ndarray:
        return np.column_stack([np.cos(self.angle), np.sin(self.angle)]) * self.speed[:, None]

    # ------------------------------------------------------------------
    def step_devices(self, device_manager: DeviceManager) -> None:
        self.sim_time_s += self.dt

        # 1. global period (and possibly episode) boundaries
        while self.sim_time_s >= self.period_start_s + (self.t_n if self.current_period_idx % 2 == 0 else self.t_c):
            self.current_period_idx += 1
            self._start_period(self.current_period_idx)

        # 2. per-UE epoch boundaries (Markov mode transitions)
        crossed = np.nonzero(self.sim_time_s >= self.epoch_end_time)[0]
        if crossed.size:
            self._advance_epochs(crossed)

        # 3. straight-line motion for this step
        pos = device_manager.get_all_positions().astype(np.float64)
        vel = self._velocity_vectors()
        step = vel * self.dt
        p = pos[:, :2] + step

        # 4a. roaming: reflect inside the service-area bounding box
        roam = self.mode == CommunityBasedMobilityHelper.ROAMING
        if roam.any():
            pr, vr = p[roam], vel[roam]
            lo, hi = self.bounds[[0, 1]], self.bounds[[2, 3]]
            for ax in range(2):
                below = pr[:, ax] < lo[ax]
                pr[below, ax] = 2 * lo[ax] - pr[below, ax]
                vr[below, ax] *= -1
                above = pr[:, ax] > hi[ax]
                pr[above, ax] = 2 * hi[ax] - pr[above, ax]
                vr[above, ax] *= -1
            p[roam], vel[roam] = pr, vr

        # 4b. local: specular reflection at the community circle boundary
        if (~roam).any():
            li = np.nonzero(~roam)[0]
            pl_, vl = p[li], vel[li]
            c = self.device_center[li]
            d = pl_ - c
            dist = np.linalg.norm(d, axis=1)
            out = dist > self.current_radius
            if out.any():
                n = d[out] / dist[out, None]                       # outward unit normal
                pl_[out] = c[out] + n * self.current_radius        # back to boundary
                v = vl[out]
                vl[out] = v - 2.0 * np.sum(v * n, axis=1)[:, None] * n
            p[li], vel[li] = pl_, vl

        # 5. write back (speed/angle may have changed via reflection)
        self.speed = np.linalg.norm(vel, axis=1)
        self.angle = np.arctan2(vel[:, 1], vel[:, 0])

        new_pos = np.empty_like(pos)
        new_pos[:, :2] = p
        new_pos[:, 2] = 1.5
        device_manager.device_position_matrix = new_pos.astype(np.float32)
        device_manager.device_velocity_matrix = np.column_stack([step, np.zeros(self.num_devices)]).astype(np.float32)


class RandomWalkMobilityHelper(MobilityHelper):
    def __init__(self, seed=24) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)

    def step_devices(self, device_manager: DeviceManager) -> None:
        positions = device_manager.get_all_positions().astype(np.float32)
        velocities = device_manager.get_all_velocities().astype(np.float32)

        perturbation = self.rng.uniform(-0.5, 0.5, size=velocities.shape).astype(np.float32)
        velocities[:, :2] = velocities[:, :2] + perturbation[:, :2]

        max_speed = 3.0
        speed = np.linalg.norm(velocities[:, :2], axis=1)
        speed = np.maximum(speed, 1e-6)
        scale = np.minimum(max_speed / speed, 1.0)
        velocities[:, :2] *= scale[:, None]

        new_positions = positions + velocities
        new_positions[:, 2] = np.maximum(new_positions[:, 2], 1.0)

        device_manager.device_velocity_matrix = velocities
        device_manager.device_position_matrix = new_positions


class LinearWalkMobilityHelper(MobilityHelper):
    def __init__(self, seed=24):
        super().__init__()

    def step_devices(self, device_manager: DeviceManager) -> None:
        device_manager.device_position_matrix += device_manager.get_all_velocities()