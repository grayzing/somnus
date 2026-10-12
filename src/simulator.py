import gc
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from geometry_helper import GeometryHelper
from radio_channel_model import RadioChannelModel
from handover_manager import RSRPBasedHandoverManager
from mobility_helper import CommunityBasedMobilityHelper
from network_topology_helper import VisibilityBasedNetworkTopologyHelper
from scheduler import QueueAwareProportionalFairPhysicalResourceBlockScheduler
from traffic_generator import BurstyTrafficGenerator
from sleep_mode_manager import SleepModeManager
from simulation_kpis_handler import SimulationKPIHandler
from power_consumption_helper import RadioUnitPowerConsumptionHelper

DOWNTOWN_LA_BBOX = (-118.252308, 34.045850, -118.246892, 34.050350)   # (left, bottom, right, top)


def load_obstacles(bbox=DOWNTOWN_LA_BBOX, cache_path="osm_buildings_cache.pkl") -> gpd.GeoDataFrame:
    """
    Building footprints in a LOCAL metric frame (UTM, shifted so the bbox corner is the origin; float32-safe).
    Cached on disk after the first download so every run / ablation arm sees the identical map.
    NOTE: the cache is a pickle - only load files you created.
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    import osmnx as ox
    gdf = ox.features_from_bbox(bbox, tags={'building': True})
    if gdf.crs is not None and not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    minx, miny = gdf.total_bounds[:2]
    gdf = gdf.set_geometry(gdf.geometry.translate(xoff=-minx, yoff=-miny))
    if cache_path:
        gdf.to_pickle(cache_path)
    return gdf


class Simulator:
    """
    One simulator step = `step_ms` ms (default 1).  Order inside step():
        1. sleep-mode timers tick (wake-ups complete)
        2. every 10 ms: UEs move; geometry, path loss, antenna gain and received power are refreshed
           (line of sight itself only every `los_update_period_ms`)
        3. association (handover) - recomputed ONLY if received power or the sleep state changed; this is exact
           because handover is idempotent for unchanged inputs
        4. serving-link SINR / spectral efficiency (co-channel interference from awake sectors, current load)
        5. arrivals join the backlog, the scheduler drains it, KPIs / power / energy are accumulated
    Speed: only serving-link quantities are computed per step (nothing else is consumed), power is a cached
    affine function of load, and association is event driven.  `los_update_period_ms=10` gives exact LoS.
    Seeding: `seed` controls the episode (mobility, traffic, shadowing); `topology_seed` (default 0, fixed)
    controls base-station placement, so all runs and ablation arms share one network unless you change it.
    """
    GEOMETRY_UPDATE_MS = 10.0

    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000,
                 seed: int = 72288026, *, topology_seed: int = 0, step_ms: float = 1.0,
                 obstacles_gdf: gpd.GeoDataFrame = None, osm_cache_path: str = "osm_buildings_cache.pkl",
                 sectors_per_ru: int = 3, los_update_period_ms: float = 100.0) -> None:
        self.num_base_stations = num_base_stations
        self.num_devices = num_devices
        self.num_sectors = num_base_stations * 3
        self.simulation_length_ms = simulation_length_ms
        self.step_ms = step_ms
        self.geometry_update_period = max(1, int(round(self.GEOMETRY_UPDATE_MS / step_ms)))
        self.window_length = int(round(simulation_length_ms / step_ms))
        self.rng = np.random.default_rng(seed)
        los_period_calls = max(1, int(round(los_update_period_ms / self.GEOMETRY_UPDATE_MS)))
        topology_rng = np.random.default_rng(topology_seed)

        self.sleep_mode_manager = SleepModeManager(num_sectors=self.num_sectors)

        # 1. Map (local metric frame) ------------------------------------------------------------
        if obstacles_gdf is None:
            obstacles_gdf = load_obstacles(cache_path=osm_cache_path)
        self.geometry_helper = GeometryHelper(num_sectors=self.num_sectors, num_devices=self.num_devices, obstacles_gdf=obstacles_gdf,
                                        los_update_period=los_period_calls)

        # 2. Candidate BS positions / service-area grid ------------------------------------------
        minx, miny, maxx, maxy = obstacles_gdf.total_bounds
        candidate_bs_positions = np.column_stack((
            topology_rng.uniform(minx, maxx, 100), topology_rng.uniform(miny, maxy, 100), np.full(100, 25.0)))
        xx, yy = np.meshgrid(np.linspace(minx, maxx, 20), np.linspace(miny, maxy, 20))
        service_area_grid = np.column_stack((xx.ravel(), yy.ravel(), np.full(400, 1.5)))

        # 3. Topology (fixed for the whole run) ---------------------------------------------------
        self.network_topology_helper = VisibilityBasedNetworkTopologyHelper(
            num_base_stations=self.num_base_stations, num_sectors_per_base_station=3, num_devices=self.num_devices,
            candidate_bs_positions=candidate_bs_positions, service_area_grid=service_area_grid,
            geometry_helper=self.geometry_helper, seed=topology_seed)
        self.network_topology_helper.intialize_network_topology()

        self.base_station_manager = self.network_topology_helper.base_station_manager
        self.sector_manager = self.network_topology_helper.sector_manager
        self.device_manager = self.network_topology_helper.device_manager

        # 4. Mobility (Masrur et al., Sec. III-B), one episode == one simulation run --------------
        episode_duration_s = self.simulation_length_ms / 1000.0
        self.mobility_helper = CommunityBasedMobilityHelper(
            service_area_bounds=(minx, miny, maxx, maxy), service_area_grid_xy=service_area_grid,
            num_communities=7, area_nmp_m2=500.0, area_cmp_m2=250.0,
            nmp_duration_s=episode_duration_s * 2.0 / 3.0, cmp_duration_s=episode_duration_s / 3.0,
            num_epochs=10, mean_epoch_s=episode_duration_s / 10.0,
            p_local_nmp=0.5, p_local_cmp=0.5, p_roam=0.5, vmin_mps=5.0, vmax_mps=5.0,
            dt_s=0.01, seed=seed)

        # 5. Remaining components -------------------------------------------------------------------
        self.radio_channel_model = RadioChannelModel(num_sectors=self.num_sectors, num_devices=self.num_devices, seed=seed)
        self.handover_manager = RSRPBasedHandoverManager(num_sectors=self.num_sectors, num_devices=self.num_devices)
        self.traffic_generator = BurstyTrafficGenerator(num_devices=self.num_devices, window_length=self.window_length,
                                                        seed=seed, step_ms=step_ms)
        self.scheduler = QueueAwareProportionalFairPhysicalResourceBlockScheduler(
            num_sectors=self.num_sectors, num_devices=self.num_devices, step_ms=step_ms)
        self.kpi_handler = SimulationKPIHandler(self.num_devices, self.num_sectors)
        self.power_helper = RadioUnitPowerConsumptionHelper(self.num_sectors, sectors_per_ru=sectors_per_ru)

        self._seen_sleep_version = -1
        self._serving = None
        self._start_episode(seed)

    # ------------------------------------------------------------------ episode lifecycle
    def _reseed(self, seed) -> None:
        """Independent child streams so the seed really controls every random process."""
        children = np.random.SeedSequence(seed).spawn(4)
        self.rng = np.random.default_rng(children[0])
        self.radio_channel_model.reset(children[1])
        self.traffic_generator.rng = np.random.default_rng(children[2])
        self.mobility_helper.rng = np.random.default_rng(children[3])

    def _start_episode(self, seed) -> None:
        self._reseed(seed)
        self.sleep_mode_manager.reset()
        self.mobility_helper.reset(self.device_manager)
        self.sector_manager.sector_physical_resource_block_utilization = np.full((self.num_sectors,), 0.05, dtype=np.float32)
        self.handover_manager.reset()
        self.scheduler.reset()
        self.traffic_generator.generate_device_downlink_bits_matrix()
        self.kpi_handler.reset()
        self.geometry_helper.reset_los_cache()
        self._seen_sleep_version = -1
        self.refresh_radio_state()

    def reset(self, seed):
        self._start_episode(42 if seed is None else seed)
        gc.collect()

    # ------------------------------------------------------------------ physics
    def _update_geometry_and_channel(self) -> None:
        self.geometry_helper.update_spatial_geometry(self.device_manager, self.sector_manager, self.base_station_manager)
        self.radio_channel_model.update_path_loss_matrix(self.geometry_helper, self.sector_manager, self.base_station_manager, self.device_manager)
        self.radio_channel_model.update_directional_gain_matrix(self.geometry_helper, self.sector_manager, self.device_manager)
        self.radio_channel_model.update_received_power_matrix_per_resource_element(self.sector_manager)

    def _update_association(self) -> None:
        self.handover_manager.handover(self.sector_manager, self.device_manager, self.radio_channel_model, self.sleep_mode_manager)
        self._serving = self.handover_manager.get_serving_sector_indices()
        self._seen_sleep_version = self.sleep_mode_manager.version

    def _update_serving_link(self) -> None:
        self.radio_channel_model.update_serving_link(self._serving, self.sector_manager, self.sleep_mode_manager)

    def refresh_radio_state(self) -> None:
        """Populate geometry / channel / association / serving link for the CURRENT state without advancing time,
        scheduling traffic or touching KPIs (use it to get a valid first observation)."""
        self._update_geometry_and_channel()
        self._update_association()
        self._update_serving_link()

    def refresh_full_link_matrices(self) -> None:
        """Diagnostics only: the full (sectors x UEs) SINR and spectral-efficiency matrices."""
        self.radio_channel_model.update_snr_dbm_matrix_per_slot(self.sector_manager, self.sleep_mode_manager)
        self.radio_channel_model.update_spectral_efficiency_matrix(self.sleep_mode_manager)

    def sleep_xapp(self, step_number):
        pass

    def step(self, step_number):
        sm = self.sleep_mode_manager
        sm.tick()
        dirty = sm.version != self._seen_sleep_version
        if step_number % self.geometry_update_period == 0:
            self.mobility_helper.step_devices(self.device_manager)
            self._update_geometry_and_channel()
            dirty = True
        if dirty:
            self._update_association()
        self._update_serving_link()

        arrivals = self.traffic_generator.arrivals_at(step_number)
        self.traffic_generator.backlog_bits += arrivals
        _, transmitted_bits = self.scheduler.schedule(
            self.sector_manager, self.radio_channel_model, self.traffic_generator, self.handover_manager,
            self.device_manager, sm, step_number)

        power_w = self.power_helper.calculate_total_power_fast(sm, self.sector_manager.sector_physical_resource_block_utilization)
        self.kpi_handler.update_kpis(transmitted_bits)
        self.kpi_handler.update_window(arrivals, transmitted_bits, power_w, self.step_ms)

        if step_number % 100 == 0:
            self.sleep_xapp(step_number)

    # ------------------------------------------------------------------ RL helpers
    def pop_interval_kpis(self, sla_mbps: float = 8.0) -> dict:
        """KPIs accumulated since the previous call (one control interval); resets the window."""
        kpis = self.kpi_handler.pop_window(sla_mbps, self.traffic_generator.backlog_bits)
        kpis["energy_saving"] = self.power_helper.energy_saving_ratio(kpis["avg_power_w"])
        return kpis

    def run_simulation(self):
        self.kpi_handler.start_clock()
        for step in range(0, self.window_length):
            self.step(step_number=step)
        self.kpi_handler.end_clock()
        return self.kpi_handler.get_elapsed_time()