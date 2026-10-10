import numpy as np

from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from geometry_helper import GeometryHelper

from abc import ABC, abstractmethod

from scipy.spatial.distance import cdist

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

class HexagonalNetworkTopologyHelperWithRandomDevicePlacements(NetworkTopologyHelper):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int, inter_site_distance_m: float = 500.0, seed: int = 24) -> None:
        super().__init__(num_base_stations=num_base_stations, num_sectors_per_base_station=num_sectors_per_base_station, num_devices=num_devices)
        self.inter_site_distance_m = inter_site_distance_m
        self.rng = np.random.default_rng(seed)

    def generate_base_station_positions(self):
        """
        Generates a true mathematical hexagonal grid spiraling outwards in rings.
        This completely eliminates row/column box alignment issues.
        """
        base_station_positions = []
        
        base_station_positions.append([0.0, 0.0, 10.0]) # Z=10 default for micros
        
        axial_coords = []
        
        for ring in [1,2]:
            q = ring
            r = 0
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
            x = self.inter_site_distance_m * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
            y = self.inter_site_distance_m * (3.0/2 * r)
            base_station_positions.append([x, y, 10.0]) # Default Z=10 for micros

        self.base_station_manager.base_station_position_matrix = np.array(base_station_positions[:self.num_base_stations], dtype=np.float32)
        # Now we also populate the neighboring sectors.
        sector_positions: np.ndarray = self.base_station_manager.get_all_positions()[self.sector_manager.sector_parent_base_station_vector] # M x 3
        sector_distance_matrix: np.ndarray = cdist(sector_positions, sector_positions).astype(np.float32) + 1e-9
        # ...Then sort the columns. 
        self.sector_manager.neighboring_sectors_indices_matrix = np.argpartition(sector_distance_matrix, -18, axis=1)[:, -18:]

    def generate_device_positions(self):
        x_min = np.min(self.base_station_manager.base_station_position_matrix[:, 0])
        x_max = np.max(self.base_station_manager.base_station_position_matrix[:, 0])
        y_min = np.min(self.base_station_manager.base_station_position_matrix[:, 1])
        y_max = np.max(self.base_station_manager.base_station_position_matrix[:, 1])

        velocity_multipliers = self.rng.integers(-1, 2, size=(self.num_devices, 3))
        self.device_manager.device_velocity_matrix *= velocity_multipliers

        device_positions = self.rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(self.device_manager.device_position_matrix.shape[0], 2))
        device_positions = np.hstack((device_positions, np.zeros((device_positions.shape[0], 1))))  
        self.device_manager.device_position_matrix = device_positions.astype(np.float32)


class HeterogenousHexagonalNetworkTopologyHelperWithRandomDevicePlacements(HexagonalNetworkTopologyHelperWithRandomDevicePlacements):
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int, inter_site_distance_m: float = 200.0, seed: int = 24) -> None:
        super().__init__(num_base_stations=num_base_stations, num_sectors_per_base_station=num_sectors_per_base_station, num_devices=num_devices, inter_site_distance_m=inter_site_distance_m, seed=seed)
        self.reuse_frequencies_ghz = np.array([35.0, 37.0, 39.0], dtype=np.float32)
        self.sector_manager.tx_power_dbm_matrix = np.full((self.sector_manager.num_sectors, ), 35.0, dtype=np.float32) 
        self.sector_manager.bandwidth_mhz_matrix = np.full((self.sector_manager.num_sectors, ), 100.0, dtype=np.float32) 
        self.sector_manager.sector_numerology_matrix = np.full((self.sector_manager.num_sectors, ), 3, dtype=np.int8) 

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
    Implements the BS placement algorithm using a voxel viewshed/visibility approach 
    for Urban Macro (UMa) environments as described in arXiv:2511.22105v1.
    """
    def __init__(self, num_base_stations: int, num_sectors_per_base_station: int, num_devices: int, 
                 candidate_bs_positions: np.ndarray, service_area_grid: np.ndarray, 
                 geometry_helper: GeometryHelper, seed: int = 24) -> None:
        super().__init__(num_base_stations=num_base_stations, 
                         num_sectors_per_base_station=num_sectors_per_base_station, 
                         num_devices=num_devices)
        self.candidate_bs_positions = candidate_bs_positions
        self.service_area_grid = service_area_grid
        self.geometry_helper = geometry_helper
        self.rng = np.random.default_rng(seed)

    def generate_base_station_positions(self):
        """
        Extracts building centroids and heights to act as candidate BS positions,
        then iteratively selects the BS with the maximum visibility (LoS) to uncovered 
        Service Area (SA) grid points.
        """
        obstacles = self.geometry_helper.obstacles_gdf
        
        # Override initial random candidates with building rooftops if GIS data is present
        if obstacles is not None and not obstacles.empty:
            building_candidates = []
            for idx, row in obstacles.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                
                # Determine building height from OSM metadata (fallback to 15.0m)
                height = 15.0
                if 'height' in row and row['height']:
                    try:
                        height = float(str(row['height']).replace('m', '').replace(',', '').split(';')[0])
                    except (ValueError, TypeError):
                        pass
                elif 'building:levels' in row and row['building:levels']:
                    try:
                        height = float(str(row['building:levels']).split(';')[0]) * 3.0
                    except (ValueError, TypeError):
                        pass
                
                centroid = geom.centroid
                building_candidates.append([centroid.x, centroid.y, height])
                
            if building_candidates:
                self.candidate_bs_positions = np.array(building_candidates, dtype=np.float32)

        num_candidates = self.candidate_bs_positions.shape[0]
        num_sa_points = self.service_area_grid.shape[0]
        
        # Precompute visibility matrix (Candidates x SA Points)
        visibility_matrix = np.zeros((num_candidates, num_sa_points), dtype=bool)
        for c in range(num_candidates):
            for p in range(num_sa_points):
                visibility_matrix[c, p] = self.geometry_helper.check_line_of_sight_pairwise(
                    self.candidate_bs_positions[c], 
                    self.service_area_grid[p]
                )
        
        selected_bs_indices = []
        covered_sa_mask = np.zeros(num_sa_points, dtype=bool)
        
        # Greedily select BSs that maximize new visibility
        while len(selected_bs_indices) < self.num_base_stations:
            best_candidate = -1
            max_new_coverage = 0
            
            for c in range(num_candidates):
                if c in selected_bs_indices:
                    continue
                
                # Calculate how many currently uncovered points this candidate can see
                new_coverage = np.sum(visibility_matrix[c] & ~covered_sa_mask)
                
                if new_coverage > max_new_coverage:
                    max_new_coverage = new_coverage
                    best_candidate = c
            
            # Stop if no further visibility improvement is possible
            if best_candidate == -1 or max_new_coverage == 0:
                break
                
            selected_bs_indices.append(best_candidate)
            covered_sa_mask |= visibility_matrix[best_candidate]

        # If we couldn't find enough BSs to hit the target, fallback or warn
        if len(selected_bs_indices) < self.num_base_stations:
            print(f"Warning: Only {len(selected_bs_indices)} BSs provided visibility improvements. "
                  f"Target was {self.num_base_stations}.")
            # Pad with random unselected candidates if strict count is required
            remaining_candidates = list(set(range(num_candidates)) - set(selected_bs_indices))
            pad_count = self.num_base_stations - len(selected_bs_indices)
            selected_bs_indices.extend(self.rng.choice(remaining_candidates, min(pad_count, len(remaining_candidates)), replace=False))

        # Assign selected positions to the Base Station Manager
        self.base_station_manager.base_station_position_matrix = self.candidate_bs_positions[selected_bs_indices[:self.num_base_stations]].astype(np.float32)

    def generate_device_positions(self):
        """
        Distributes devices randomly across the valid Service Area (SA) grid points.
        """
        num_sa_points = self.service_area_grid.shape[0]
        selected_indices = self.rng.choice(num_sa_points, self.num_devices, replace=True)
        device_positions = self.service_area_grid[selected_indices]
        
        self.device_manager.device_position_matrix = device_positions.astype(np.float32)
        
        velocity_multipliers = self.rng.integers(-1, 2, size=(self.num_devices, 3))
        self.device_manager.device_velocity_matrix *= velocity_multipliers