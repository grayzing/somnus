import numpy as np
import geopandas as gpd

from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from geometry_helper import GeometryHelper, parse_building_heights

from abc import ABC, abstractmethod

DEFAULT_UE_SPEED_VECTOR = np.array([0.01388, 0.01388, 0.0])   # 5 km/h per 10 ms
ROOFTOP_MOUNT_OFFSET_M = 2.0                                  # antenna height above the roof


class NetworkTopologyHelper(ABC):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int) -> None:
        self.base_station_manager = BaseStationManager(num_base_stations=num_base_stations)
        self.sector_manager = SectorManager(num_sectors=num_sectors_per_base_station * num_base_stations)
        self.device_manager = DeviceManager(num_devices=num_devices)

        self.sector_manager.sector_parent_base_station_vector = np.repeat(
            np.arange(num_base_stations, dtype=np.int16),
            num_sectors_per_base_station,
        )

        reuse_frequencies_ghz = np.array([6.0, 6.5, 7.0], dtype=np.float32)
        base_station_center_freqs = reuse_frequencies_ghz[np.arange(num_base_stations) % reuse_frequencies_ghz.shape[0]]
        self.sector_manager.center_freq_ghz_matrix = np.repeat(base_station_center_freqs, num_sectors_per_base_station)

        self.num_sectors = num_sectors_per_base_station * num_base_stations
        self.num_base_stations = num_base_stations
        self.num_sectors_per_base_station = num_sectors_per_base_station
        self.num_devices = num_devices

    @abstractmethod
    def generate_base_station_positions(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def generate_device_positions(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def intialize_network_topology(self):
        self.generate_base_station_positions()
        self.generate_device_positions()

    # ------------------------------------------------------------------ neighbours
    def _populate_neighboring_sectors(self, num_neighbor_base_stations: int = 6) -> None:
        """
        neighboring_sectors_indices_matrix[s] = all sectors of the `num_neighbor_base_stations` base stations
        nearest to s's own site (own site first).  6 sites x 3 sectors = 18 columns.  With fewer sites the
        remaining columns repeat the sector's own site's first sector.
        """
        spr = self.num_sectors_per_base_station
        pos = np.asarray(self.base_station_manager.get_all_positions(), dtype=np.float64)[:, :2]
        B = pos.shape[0]
        k = min(num_neighbor_base_stations, B)
        width = self.sector_manager.neighboring_sectors_indices_matrix.shape[1]
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

        matrix = np.zeros((self.num_sectors, width), dtype=np.int16)
        for b in range(B):
            order = np.argsort(dist[b], kind="stable")
            order = np.concatenate([[b], order[order != b]])[:k]
            row = (order[:, None] * spr + np.arange(spr)).ravel()
            if row.size < width:
                row = np.concatenate([row, np.full(width - row.size, row[0])])
            matrix[b * spr:(b + 1) * spr] = row[:width]
        self.sector_manager.neighboring_sectors_indices_matrix = matrix

    def _default_velocities(self, rng) -> None:
        multipliers = rng.integers(-1, 2, size=(self.num_devices, 3))
        self.device_manager.device_velocity_matrix = (DEFAULT_UE_SPEED_VECTOR[None, :] * multipliers).astype(np.float32)


class HexagonalNetworkTopologyHelperWithRandomDevicePlacements(NetworkTopologyHelper):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int, inter_site_distance_m: float = 500.0, seed: int = 24) -> None:
        super().__init__(num_base_stations=num_base_stations, num_sectors_per_base_station=num_sectors_per_base_station, num_devices=num_devices)
        self.inter_site_distance_m = inter_site_distance_m
        self.rng = np.random.default_rng(seed)

    def generate_base_station_positions(self):
        """Hexagonal grid spiralling outwards in rings."""
        base_station_positions = [[0.0, 0.0, 10.0]]
        axial_coords = []
        for ring in [1, 2]:
            q, r = ring, 0
            for direction in range(6):
                for step in range(ring):
                    axial_coords.append((q, r))
                    if direction == 0: r -= 1
                    elif direction == 1: q -= 1
                    elif direction == 2: q -= 1; r += 1
                    elif direction == 3: r += 1
                    elif direction == 4: q += 1
                    elif direction == 5: q += 1; r -= 1

        for q, r in axial_coords:
            if len(base_station_positions) >= self.base_station_manager.base_station_position_matrix.shape[0]:
                break
            x = self.inter_site_distance_m * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
            y = self.inter_site_distance_m * (3.0 / 2 * r)
            base_station_positions.append([x, y, 10.0])

        self.base_station_manager.base_station_position_matrix = np.array(base_station_positions[:self.num_base_stations], dtype=np.float32)
        # Previously np.argpartition(d, -18)[:, -18:] -> the 18 FARTHEST sectors. Now the nearest sites.
        self._populate_neighboring_sectors()

    def generate_device_positions(self):
        bs = self.base_station_manager.base_station_position_matrix
        x_min, x_max, y_min, y_max = bs[:, 0].min(), bs[:, 0].max(), bs[:, 1].min(), bs[:, 1].max()
        self._default_velocities(self.rng)
        device_positions = self.rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(self.num_devices, 2))
        device_positions = np.hstack((device_positions, np.full((self.num_devices, 1), 1.5)))
        self.device_manager.device_position_matrix = device_positions.astype(np.float32)


class HeterogenousHexagonalNetworkTopologyHelperWithRandomDevicePlacements(HexagonalNetworkTopologyHelperWithRandomDevicePlacements):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int, inter_site_distance_m: float = 200.0, seed: int = 24) -> None:
        super().__init__(num_base_stations=num_base_stations, num_sectors_per_base_station=num_sectors_per_base_station, num_devices=num_devices, inter_site_distance_m=inter_site_distance_m, seed=seed)
        self.reuse_frequencies_ghz = np.array([35.0, 37.0, 39.0], dtype=np.float32)
        self.sector_manager.tx_power_dbm_matrix = np.full((self.sector_manager.num_sectors,), 35.0, dtype=np.float32)
        self.sector_manager.bandwidth_mhz_matrix = np.full((self.sector_manager.num_sectors,), 100.0, dtype=np.float32)
        self.sector_manager.sector_numerology_matrix = np.full((self.sector_manager.num_sectors,), 3, dtype=np.int8)

    def generate_base_station_positions(self):
        super().generate_base_station_positions()
        base_station_center_freqs = self.reuse_frequencies_ghz[np.arange(self.base_station_manager.base_station_position_matrix.shape[0]) % self.reuse_frequencies_ghz.shape[0]]
        self.sector_manager.center_freq_ghz_matrix = np.repeat(base_station_center_freqs, self.num_sectors_per_base_station)
        self.base_station_manager.base_station_position_matrix[0] = [0.0, 0.0, 25.0]
        self.sector_manager.center_freq_ghz_matrix[0:3] = [6.0, 6.5, 7.0]
        self.sector_manager.tx_power_dbm_matrix[0:3] = 49.0
        self.sector_manager.bandwidth_mhz_matrix[0:3] = 20.0
        self.sector_manager.sector_numerology_matrix[0:3] = 1


class VisibilityBasedNetworkTopologyHelper(NetworkTopologyHelper):
    """
    BS placement by greedy max-coverage over a (candidate x service-area-point) LoS matrix.
    Rooftop candidates are building representative points (guaranteed inside the footprint), mounted
    ROOFTOP_MOUNT_OFFSET_M above the roof, and a candidate's own building never blocks its own LoS.
    """
    def __init__(self, num_base_stations, num_sectors_per_base_station, num_devices,
                 candidate_bs_positions, service_area_grid, geometry_helper, seed=24):
        super().__init__(num_base_stations=num_base_stations,
                         num_sectors_per_base_station=num_sectors_per_base_station,
                         num_devices=num_devices)
        self.candidate_bs_positions = candidate_bs_positions
        self.service_area_grid = service_area_grid
        self.geometry_helper = geometry_helper
        self.rng = np.random.default_rng(seed)

    def _extract_building_candidates(self, obstacles: gpd.GeoDataFrame) -> np.ndarray:
        heights = parse_building_heights(obstacles) + ROOFTOP_MOUNT_OFFSET_M
        pts = obstacles.geometry.representative_point()
        return np.column_stack([np.asarray(pts.x), np.asarray(pts.y), heights]).astype(np.float32)

    def generate_base_station_positions(self):
        obstacles = self.geometry_helper.obstacles_gdf
        host_idx = None
        if obstacles is not None and not obstacles.empty:
            candidates = self._extract_building_candidates(obstacles)
            if candidates.size:
                self.candidate_bs_positions = candidates
                host_idx = np.arange(len(candidates))          # candidate i sits on building i

        candidates = self.candidate_bs_positions
        grid = self.service_area_grid
        num_candidates, num_sa_points = candidates.shape[0], grid.shape[0]

        visibility_matrix = self.geometry_helper.line_of_sight_matrix(candidates, grid, from_host_idx=host_idx)

        selected: list = []
        selected_mask = np.zeros(num_candidates, dtype=bool)
        covered_sa_mask = np.zeros(num_sa_points, dtype=bool)
        for _ in range(self.num_base_stations):
            new_coverage = np.count_nonzero(visibility_matrix & ~covered_sa_mask, axis=1)
            new_coverage[selected_mask] = 0
            best = int(np.argmax(new_coverage))
            if new_coverage[best] == 0:
                break
            selected.append(best)
            selected_mask[best] = True
            covered_sa_mask |= visibility_matrix[best]

        if len(selected) < self.num_base_stations:
            print(f"Warning: Only {len(selected)} BSs provided visibility improvements. "
                  f"Target was {self.num_base_stations}.")
            remaining = np.nonzero(~selected_mask)[0]
            pad = min(self.num_base_stations - len(selected), remaining.size)
            if pad > 0:
                selected.extend(self.rng.choice(remaining, pad, replace=False).tolist())

        self.base_station_manager.base_station_position_matrix = candidates[selected[:self.num_base_stations]].astype(np.float32)
        self.coverage_fraction = float(covered_sa_mask.mean())
        self._populate_neighboring_sectors()

    def generate_device_positions(self):
        selected_indices = self.rng.choice(self.service_area_grid.shape[0], self.num_devices, replace=True)
        self.device_manager.device_position_matrix = self.service_area_grid[selected_indices].astype(np.float32)
        self._default_velocities(self.rng)