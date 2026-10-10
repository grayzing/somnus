import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from device_manager import DeviceManager
from sector_manager import SectorManager
from base_station_manager import BaseStationManager
from scipy.spatial.distance import cdist

class GeometryHelper:
    def __init__(self, num_sectors: int, num_devices: int, obstacles_gdf: gpd.GeoDataFrame) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices
        self.obstacles_gdf = obstacles_gdf

        self.relative_azimuth_angle_deg_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.relative_zenith_angle_deg_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.distance_matrix_meters_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.is_line_of_sight_mask_matrix: np.ndarray = np.ones((num_sectors, num_devices), dtype=bool)
        self.is_indoor_mask_vector: np.ndarray = np.zeros(num_devices, dtype=bool)

    def set_obstacles(self, obstacles_gdf: gpd.GeoDataFrame):
        """Load or update the GeoPandas GeoDataFrame containing GIS obstacles (e.g., buildings)."""
        self.obstacles_gdf = obstacles_gdf

    def check_line_of_sight_pairwise(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Check if two points (p1, p2) have a clear line of sight based on GIS geometry.
        p1, p2: array-like of shape (2,) or (3,) representing [x, y, (z)] coordinates.
        """
        if self.obstacles_gdf is None or self.obstacles_gdf.empty:
            return True
        
        line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
        
        # Use GeoPandas spatial index (sindex) for fast intersection queries
        if hasattr(self.obstacles_gdf, 'sindex'):
            possible_matches_index = list(self.obstacles_gdf.sindex.intersection(line.bounds))
            possible_matches = self.obstacles_gdf.iloc[possible_matches_index]
            intersecting = possible_matches.geometry.intersects(line)
            return not intersecting.any()
        else:
            return not self.obstacles_gdf.geometry.intersects(line).any()

    def update_spatial_geometry(self, device_manager: DeviceManager, sector_manager: SectorManager, base_station_manager: BaseStationManager):
        device_positions: np.ndarray = device_manager.get_all_positions() # U x 3
        sector_positions: np.ndarray = base_station_manager.get_all_positions()[sector_manager.sector_parent_base_station_vector] # M x 3

        self.distance_matrix_meters_matrix = cdist(sector_positions, device_positions).astype(np.float32) + 1e-9 # M x U
        displacement_matrix_meters: np.ndarray = device_positions[np.newaxis, :, :] - sector_positions[:, np.newaxis, :] # M x U x 3
        dz_meters: np.ndarray = displacement_matrix_meters[:, :, 2] # M x U
        cosine_zenith = np.clip(dz_meters / self.distance_matrix_meters_matrix, -1.0, 1.0)
        absolute_zenith_deg = np.clip(np.degrees(np.arccos(cosine_zenith)), 0.0, 180.0)
        
        sector_downtilt = sector_manager.downtilt_deg_matrix[:, np.newaxis]
        self.relative_zenith_angle_deg_matrix = absolute_zenith_deg - (90.0 + sector_downtilt)

        dx_meters: np.ndarray = displacement_matrix_meters[:, :, 0]
        dy_meters: np.ndarray = displacement_matrix_meters[:, :, 1]

        azimuth_abs = np.degrees(np.arctan2(dy_meters, dx_meters))

        azimuth_raw = azimuth_abs - sector_manager.sector_azimuth_angle_deg_matrix[:, np.newaxis]
        self.relative_azimuth_angle_deg_matrix = (azimuth_raw + 180) % 360 - 180

        # Check GIS-based Indoor Mask and Pairwise Line-of-Sight Mask
        indoor_mask = np.zeros(self.num_devices, dtype=bool)
        lo_s_mask = np.ones((self.num_sectors, self.num_devices), dtype=bool)

        if self.obstacles_gdf is not None and not self.obstacles_gdf.empty:
            for u in range(self.num_devices):
                # 1. Check Indoor Mask
                pt = Point(device_positions[u, 0], device_positions[u, 1])
                if hasattr(self.obstacles_gdf, 'sindex'):
                    possible_matches_index = list(self.obstacles_gdf.sindex.intersection(pt.bounds))
                    possible_matches = self.obstacles_gdf.iloc[possible_matches_index]
                    if possible_matches.geometry.contains(pt).any():
                        indoor_mask[u] = True
                else:
                    if self.obstacles_gdf.geometry.contains(pt).any():
                        indoor_mask[u] = True
                
                # 2. Check pairwise LoS
                for m in range(self.num_sectors):
                    # If device is indoor, it is strictly NLOS. Otherwise check the line string.
                    if indoor_mask[u]:
                        lo_s_mask[m, u] = False
                    else:
                        lo_s_mask[m, u] = self.check_line_of_sight_pairwise(sector_positions[m], device_positions[u])
        
        self.is_indoor_mask_vector = indoor_mask
        self.is_line_of_sight_mask_matrix = lo_s_mask