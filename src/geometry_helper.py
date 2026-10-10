import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from shapely import contains_xy, intersects, linestrings, prepare, union_all


class GeometryHelper:
    def __init__(self, num_sectors: int, num_devices: int, obstacles_gdf: gpd.GeoDataFrame) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices
        self.obstacles_gdf = obstacles_gdf

        self.relative_azimuth_angle_deg_matrix = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.relative_zenith_angle_deg_matrix = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.distance_matrix_meters_matrix = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.is_line_of_sight_mask_matrix = np.ones((num_sectors, num_devices), dtype=bool)
        self.is_indoor_mask_vector = np.zeros(num_devices, dtype=bool)

        # Cached prepared union of obstacles (rebuilt only when obstacles change)
        self._prep_obstacles = None

    # ------------------------------------------------------------------
    def set_obstacles(self, obstacles_gdf: gpd.GeoDataFrame):
        """Load or update obstacles; invalidates the cached prepared union."""
        self.obstacles_gdf = obstacles_gdf
        self._prep_obstacles = None

    def _get_prepared_obstacles(self):
        """Union of all obstacles, prepared once for fast predicate queries."""
        if self._prep_obstacles is None:
            if self.obstacles_gdf is None or self.obstacles_gdf.empty:
                return None
            geom = union_all(self.obstacles_gdf.geometry.values)
            prepare(geom)                     # speeds up contains/intersects hugely
            self._prep_obstacles = geom
        return self._prep_obstacles

    # ------------------------------------------------------------------
    def check_line_of_sight_pairwise(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Single-pair check kept for API compatibility (fast path for MxU loop elsewhere)."""
        prep = self._get_prepared_obstacles()
        if prep is None:
            return True
        line = linestrings([p1[0], p2[0]], [p1[1], p2[1]])   # shape (1,)
        return not bool(intersects(prep, line))

    def line_of_sight_matrix(self, from_positions: np.ndarray, to_positions: np.ndarray,
                             chunk_size: int = 512) -> np.ndarray:
        """
        Vectorized pairwise LoS: returns a (F, T) bool mask, True = clear LoS.
        from_positions: (F, 3), to_positions: (T, 3). Only x,y are used (2D LoS,
        matching check_line_of_sight_pairwise).
        """
        f, t = from_positions.shape[0], to_positions.shape[0]
        mask = np.ones((f, t), dtype=bool)
        prep = self._get_prepared_obstacles()
        if prep is None:
            return mask

        dest_xy = to_positions[:, :2]
        for f0 in range(0, f, chunk_size):
            f1 = min(f0 + chunk_size, f)
            starts = np.repeat(from_positions[f0:f1, :2], t, axis=0)
            ends = np.tile(dest_xy, (f1 - f0, 1))
            lines = linestrings(starts, ends)
            mask[f0:f1] = (~intersects(prep, lines)).reshape(f1 - f0, t)
        return mask

    # ------------------------------------------------------------------
    def update_spatial_geometry(self, device_manager, sector_manager, base_station_manager):
        device_positions = device_manager.get_all_positions()                                    # U x 3
        sector_positions = base_station_manager.get_all_positions()[
            sector_manager.sector_parent_base_station_vector]                                    # M x 3

        # --- distances -------------------------------------------------
        self.distance_matrix_meters_matrix = (
            cdist(sector_positions, device_positions).astype(np.float32) + 1e-9
        )                                                                                        # M x U

        displacement = device_positions[np.newaxis, :, :] - sector_positions[:, np.newaxis, :]   # M x U x 3

        # --- zenith ----------------------------------------------------
        cosine_zenith = np.clip(displacement[..., 2] / self.distance_matrix_meters_matrix, -1.0, 1.0)
        absolute_zenith_deg = np.clip(np.degrees(np.arccos(cosine_zenith)), 0.0, 180.0)
        self.relative_zenith_angle_deg_matrix = (
            absolute_zenith_deg - (90.0 + sector_manager.downtilt_deg_matrix[:, np.newaxis])
        ).astype(np.float32)

        # --- azimuth ---------------------------------------------------
        azimuth_abs = np.degrees(np.arctan2(displacement[..., 1], displacement[..., 0]))
        azimuth_raw = azimuth_abs - sector_manager.sector_azimuth_angle_deg_matrix[:, np.newaxis]
        self.relative_azimuth_angle_deg_matrix = (
            (azimuth_raw + 180.0) % 360.0 - 180.0
        ).astype(np.float32)

        # --- indoor + LoS, vectorized ----------------------------------
        indoor_mask = np.zeros(self.num_devices, dtype=bool)
        los_mask = np.ones((self.num_sectors, self.num_devices), dtype=bool)

        prep = self._get_prepared_obstacles()
        if prep is not None:
            # 1. Indoor test for ALL devices in one vectorized call
            indoor_mask = contains_xy(
                prep, device_positions[:, 0], device_positions[:, 1]
            )

            # 2. LoS only for outdoor devices (indoor => strictly NLOS)
            outdoor_idx = np.nonzero(~indoor_mask)[0]
            if outdoor_idx.size:
                m_idx = np.arange(self.num_sectors)
                # Build all M x len(outdoor) segments at once:
                xs = np.concatenate([
                    np.repeat(sector_positions[:, 0], outdoor_idx.size),
                    np.tile(device_positions[outdoor_idx, 0], self.num_sectors),
                ])
                ys = np.concatenate([
                    np.repeat(sector_positions[:, 1], outdoor_idx.size),
                    np.tile(device_positions[outdoor_idx, 1], self.num_sectors),
                ])
                # Pair them: segment i connects (xs[i], ys[i]) -> (xs[i+n], ys[i+n])
                n = xs.size // 2
                lines = linestrings(
                    np.column_stack([xs[:n], xs[n:]]),
                    np.column_stack([ys[:n], ys[n:]]),
                )
                blocked = intersects(prep, lines).reshape(self.num_sectors, outdoor_idx.size)
                los_mask[:, outdoor_idx] = ~blocked

        self.is_indoor_mask_vector = indoor_mask
        self.is_line_of_sight_mask_matrix = los_mask