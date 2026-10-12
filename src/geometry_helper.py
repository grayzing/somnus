import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from shapely import STRtree, linestrings, points

DEFAULT_BUILDING_HEIGHT_M = 15.0


def parse_building_heights(obstacles: gpd.GeoDataFrame, default: float = DEFAULT_BUILDING_HEIGHT_M) -> np.ndarray:
    """Per-building height in metres from OSM 'height' (or 'building:levels' x 3 m), else `default`."""
    heights = np.full(len(obstacles), default, dtype=np.float64)

    def _parse(col, scale=1.0):
        s = (obstacles[col].astype(str).str.split(';').str[0]
             .str.replace('m', '', regex=False).str.replace(',', '', regex=False))
        return pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float64) * scale

    if 'height' in obstacles.columns:
        h = _parse('height')
        heights = np.where(np.isfinite(h) & (h > 0), h, heights)
    elif 'building:levels' in obstacles.columns:
        lv = _parse('building:levels', scale=3.0)
        heights = np.where(np.isfinite(lv) & (lv > 0), lv, heights)
    return heights


class GeometryHelper:
    """
    Distances, antenna angles, indoor flags and 2.5D line-of-sight.

    LoS model: a link is blocked if its 2D segment intersects a building footprint AND the straight 3D
    ray is lower than that building's roof (altitude evaluated above the footprint centroid projected onto
    the segment).  A rooftop base station's own host building never blocks its own links.
    All coordinates are expected in a LOCAL metric frame (small numbers, float32-safe).
    """
    def __init__(self, num_sectors: int, num_devices: int, obstacles_gdf: gpd.GeoDataFrame,
                 los_update_period: int = 1) -> None:
        """los_update_period: recompute LoS for ALL outdoor UEs only every N-th update_spatial_geometry() call
        (UEs that just moved from indoor to outdoor are always recomputed). 1 = exact."""
        self.los_update_period = max(1, int(los_update_period))
        self.num_sectors = num_sectors
        self.num_devices = num_devices
        self.obstacles_gdf = obstacles_gdf

        self.relative_azimuth_angle_deg_matrix = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.relative_zenith_angle_deg_matrix = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.distance_matrix_meters_matrix = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.is_line_of_sight_mask_matrix = np.ones((num_sectors, num_devices), dtype=bool)
        self.is_indoor_mask_vector = np.zeros(num_devices, dtype=bool)

        self._tree = None
        self._heights = None
        self._centroids = None
        self._index_built = False
        self._host_key = None
        self._host_idx = None
        self._los_cache = np.ones((num_sectors, num_devices), dtype=bool)
        self._los_valid = np.zeros(num_devices, dtype=bool)
        self._geom_calls = 0

    def reset_los_cache(self) -> None:
        self._los_valid[:] = False
        self._geom_calls = 0

    # ------------------------------------------------------------------ spatial index
    def set_obstacles(self, obstacles_gdf: gpd.GeoDataFrame):
        self.obstacles_gdf = obstacles_gdf
        self._tree, self._heights, self._centroids = None, None, None
        self._index_built, self._host_key, self._host_idx = False, None, None

    def _ensure_index(self):
        if self._index_built:
            return
        self._index_built = True
        gdf = self.obstacles_gdf
        if gdf is None or gdf.empty:
            return
        geoms = np.asarray(gdf.geometry.values)
        self._tree = STRtree(geoms)
        self._heights = parse_building_heights(gdf)
        c = gdf.geometry.centroid
        self._centroids = np.column_stack([np.asarray(c.x), np.asarray(c.y)]).astype(np.float64)

    @property
    def building_heights(self) -> np.ndarray:
        self._ensure_index()
        return self._heights

    def find_host_buildings(self, xy: np.ndarray) -> np.ndarray:
        """Index of the building whose footprint contains each (x, y) point, or -1."""
        self._ensure_index()
        host = np.full(len(xy), -1, dtype=np.int64)
        if self._tree is not None and len(xy):
            pairs = self._tree.query(points(xy[:, 0], xy[:, 1]), predicate='intersects')
            host[pairs[0]] = pairs[1]
        return host

    # ------------------------------------------------------------------ line of sight
    def line_of_sight_matrix(self, from_positions: np.ndarray, to_positions: np.ndarray,
                             chunk_size: int = 256, from_host_idx=None) -> np.ndarray:
        """
        (F, T) bool mask, True = clear LoS.  from_positions: (F, 3), to_positions: (T, 3).
        `from_host_idx` (F,) gives, per origin, a building index that must be ignored (rooftop host).
        """
        f, t = from_positions.shape[0], to_positions.shape[0]
        mask = np.ones((f, t), dtype=bool)
        self._ensure_index()
        if self._tree is None or f == 0 or t == 0:
            return mask

        src = np.asarray(from_positions, dtype=np.float64)
        dst = np.asarray(to_positions, dtype=np.float64)
        host_all = None if from_host_idx is None else np.asarray(from_host_idx)

        for f0 in range(0, f, chunk_size):
            f1 = min(f0 + chunk_size, f)
            n = f1 - f0
            starts = np.repeat(src[f0:f1, :2], t, axis=0)               # (n*t, 2), row-major (from i, to j)
            ends = np.tile(dst[:, :2], (n, 1))
            z_start = np.repeat(src[f0:f1, 2], t)
            z_end = np.tile(dst[:, 2], n)

            lines = linestrings(np.stack([starts, ends], axis=1))       # coords array (N, 2, 2)
            line_idx, b_idx = self._tree.query(lines, predicate='intersects')

            visible = np.ones(n * t, dtype=bool)
            if line_idx.size:
                if host_all is not None:
                    keep = b_idx != host_all[f0 + line_idx // t]
                    line_idx, b_idx = line_idx[keep], b_idx[keep]
                s, e, c = starts[line_idx], ends[line_idx], self._centroids[b_idx]
                d = e - s
                frac = np.clip(((c - s) * d).sum(axis=1) / np.maximum((d ** 2).sum(axis=1), 1e-9), 0.0, 1.0)
                z_ray = z_start[line_idx] + frac * (z_end[line_idx] - z_start[line_idx])
                blocked = z_ray < self._heights[b_idx]
                visible[line_idx[blocked]] = False
            mask[f0:f1] = visible.reshape(n, t)
        return mask

    def check_line_of_sight_pairwise(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        p1 = np.asarray(p1, dtype=np.float64).reshape(1, -1)
        p2 = np.asarray(p2, dtype=np.float64).reshape(1, -1)
        if p1.shape[1] == 2:
            p1 = np.hstack([p1, [[25.0]]])
        if p2.shape[1] == 2:
            p2 = np.hstack([p2, [[1.5]]])
        return bool(self.line_of_sight_matrix(p1, p2)[0, 0])

    # ------------------------------------------------------------------ per-step geometry
    def update_spatial_geometry(self, device_manager, sector_manager, base_station_manager):
        device_positions = device_manager.get_all_positions()                                     # U x 3
        sector_positions = base_station_manager.get_all_positions()[
            sector_manager.sector_parent_base_station_vector]                                    # M x 3

        self.distance_matrix_meters_matrix = (
            cdist(sector_positions, device_positions).astype(np.float32) + 1e-9)                 # M x U

        displacement = device_positions[np.newaxis, :, :] - sector_positions[:, np.newaxis, :]   # M x U x 3

        cosine_zenith = np.clip(displacement[..., 2] / self.distance_matrix_meters_matrix, -1.0, 1.0)
        absolute_zenith_deg = np.clip(np.degrees(np.arccos(cosine_zenith)), 0.0, 180.0)
        self.relative_zenith_angle_deg_matrix = (
            absolute_zenith_deg - (90.0 + sector_manager.downtilt_deg_matrix[:, np.newaxis])
        ).astype(np.float32)

        azimuth_abs = np.degrees(np.arctan2(displacement[..., 1], displacement[..., 0]))
        azimuth_raw = azimuth_abs - sector_manager.sector_azimuth_angle_deg_matrix[:, np.newaxis]
        self.relative_azimuth_angle_deg_matrix = ((azimuth_raw + 180.0) % 360.0 - 180.0).astype(np.float32)

        # ---- indoor + LoS -------------------------------------------------
        indoor_mask = np.zeros(self.num_devices, dtype=bool)
        self._ensure_index()
        if self._tree is not None:
            pairs = self._tree.query(points(device_positions[:, 0], device_positions[:, 1]), predicate='intersects')
            indoor_mask[pairs[0]] = True                              # inside any footprint -> indoor, NLoS

            outdoor = ~indoor_mask
            full_refresh = (self._geom_calls % self.los_update_period) == 0
            todo = np.flatnonzero(outdoor if full_refresh else (outdoor & ~self._los_valid))
            if todo.size:
                # Sectors of one site share a position: evaluate LoS once per site, then expand.
                site_pos, inverse = np.unique(sector_positions, axis=0, return_inverse=True)
                key = site_pos.tobytes()
                if key != self._host_key:
                    self._host_idx = self.find_host_buildings(site_pos[:, :2])
                    self._host_key = key
                visible = self.line_of_sight_matrix(site_pos, device_positions[todo], from_host_idx=self._host_idx)
                self._los_cache[:, todo] = visible[np.asarray(inverse).reshape(-1)]
                self._los_valid[todo] = True
            self._los_valid[indoor_mask] = False
        self._geom_calls += 1

        self.is_indoor_mask_vector = indoor_mask
        self.is_line_of_sight_mask_matrix = self._los_cache