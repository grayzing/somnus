import time
import gc

import numpy as np
import osmnx as ox
import geopandas as gpd

from geometry_helper import GeometryHelper
from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel
from handover_manager import RSRPBasedHandoverManager
from mobility_helper import RandomWalkMobilityHelper, LinearWalkMobilityHelper
from network_topology_helper import HeterogenousHexagonalNetworkTopologyHelperWithRandomDevicePlacements, VisibilityBasedNetworkTopologyHelper
from scheduler import QueueAwareProportionalFairPhysicalResourceBlockScheduler
from traffic_generator import TrafficGenerator, BurstyTrafficGenerator
from sleep_mode_manager import SleepModeManager
from simulation_kpis_handler import SimulationKPIHandler

class Simulator:
    def __init__(self, num_base_stations: int, num_devices: int, simulation_length_ms: int = 600_000, seed: int = 72288026) -> None:
        self.num_base_stations = num_base_stations
        self.num_devices = num_devices
        self.num_sectors = num_base_stations * 3
        self.simulation_length_ms = simulation_length_ms
        self.rng = np.random.default_rng(seed=seed)
        
        self.sleep_mode_manager = SleepModeManager(num_sectors=self.num_sectors)

        # 1. Fetch Downtown LA building footprints via OSMnx from bounding box: (left, bottom, right, top)
        bbox = (-118.252308, 34.045850, -118.246892, 34.050350)
        obstacles_gdf = ox.features_from_bbox(bbox, tags={'building': True})
        
        # Project to local UTM coordinate system (meters) using GeoPandas
        if obstacles_gdf.crs is not None and not obstacles_gdf.crs.is_projected:
            obstacles_gdf = obstacles_gdf.to_crs(obstacles_gdf.estimate_utm_crs())

        # 2. Instantiate GeometryHelper FIRST so the topology helper can use it for LoS checks
        self.geometry_helper = GeometryHelper(num_sectors=self.num_sectors, num_devices=self.num_devices, obstacles_gdf=obstacles_gdf)

        # 3. Generate candidate BS positions and service area grid based on the map bounds
        minx, miny, maxx, maxy = obstacles_gdf.total_bounds
        
        # Example: 100 random candidate locations at 25m height
        candidate_bs_positions = np.column_stack((
            self.rng.uniform(minx, maxx, 100),
            self.rng.uniform(miny, maxy, 100),
            np.full(100, 25.0) 
        ))

        # Example: 20x20 service area grid for devices at 1.5m height
        xx, yy = np.meshgrid(np.linspace(minx, maxx, 20), np.linspace(miny, maxy, 20))
        service_area_grid = np.column_stack((xx.ravel(), yy.ravel(), np.full(400, 1.5)))

        # 4. Initialize Network Topology with required arguments
        self.network_topology_helper = VisibilityBasedNetworkTopologyHelper(
            num_base_stations=self.num_base_stations, 
            num_sectors_per_base_station=3, 
            num_devices=self.num_devices, 
            candidate_bs_positions=candidate_bs_positions,
            service_area_grid=service_area_grid,
            geometry_helper=self.geometry_helper,
            seed=seed
        )
        self.network_topology_helper.intialize_network_topology()

        self.base_station_manager = self.network_topology_helper.base_station_manager
        self.sector_manager = self.network_topology_helper.sector_manager
        self.device_manager = self.network_topology_helper.device_manager

        # 5. Initialize remaining components
        self.radio_channel_model = RadioChannelModel(num_sectors=self.num_sectors, num_devices=self.num_devices, seed=seed)
        self.handover_manager = RSRPBasedHandoverManager(num_sectors=self.num_sectors, num_devices=self.num_devices)
        self.mobility_helper = RandomWalkMobilityHelper()
        self.traffic_generator = BurstyTrafficGenerator(num_devices=self.num_devices, window_length=self.simulation_length_ms, seed=seed)
        self.traffic_generator.generate_device_downlink_bits_matrix()

        self.scheduler = QueueAwareProportionalFairPhysicalResourceBlockScheduler(num_sectors=self.num_sectors, num_devices=self.num_devices)
        self.kpi_handler = SimulationKPIHandler(self.num_devices, self.num_sectors)

    def reset(self, seed):
        if seed is None:
            seed = 42
        self.rng = np.random.default_rng(seed=seed)
        self.sleep_mode_manager.sector_sleep_mode_countdown = np.zeros((self.num_sectors, ), dtype=np.int8)
        self.sleep_mode_manager.sector_sleep_mode_matrix = np.zeros((self.num_sectors, ), dtype=np.int32)

        self.network_topology_helper.generate_device_positions()
        self.sector_manager.sector_physical_resource_block_utilization: np.ndarray = np.full((self.num_sectors, ), 0.05, dtype=np.float32)

        self.geometry_helper.relative_azimuth_angle_deg_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.geometry_helper.relative_zenith_angle_deg_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.geometry_helper.distance_matrix_meters_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)

        self.radio_channel_model.path_loss_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.directional_gain_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.received_power_dbm_matrix_per_resource_element: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.snr_dbm_matrix_per_slot: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)
        self.radio_channel_model.spectral_efficiency_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)

        self.handover_manager.sector_device_association_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.int8)
        self.traffic_generator.generate_device_downlink_bits_matrix()

        self.kpi_handler.total_transmitted_bits_per_device = np.zeros(self.num_devices, dtype=np.float64)

        gc.collect()

    def sleep_xapp(self, step_number):
        pass

    def step(self, step_number):
        if step_number % 10 == 0:
            self.mobility_helper.step_devices(self.device_manager)
            self.geometry_helper.update_spatial_geometry(self.device_manager,self.sector_manager,self.base_station_manager)

        self.radio_channel_model.update_path_loss_matrix(self.geometry_helper, self.sector_manager, self.base_station_manager, self.device_manager)
        self.radio_channel_model.update_directional_gain_matrix(self.geometry_helper, self.sector_manager, self.device_manager)
        self.radio_channel_model.update_received_power_matrix_per_resource_element(self.sector_manager)
        self.radio_channel_model.update_snr_dbm_matrix_per_slot(self.sector_manager)
        self.radio_channel_model.update_spectral_efficiency_matrix(self.sleep_mode_manager)

        self.sleep_mode_manager.tick()
        self.handover_manager.handover(self.sector_manager, self.device_manager, self.radio_channel_model, self.sleep_mode_manager)

        prb_allocation_matrix, transmitted_bits_per_device = self.scheduler.schedule(
            self.sector_manager,
            self.radio_channel_model,
            self.traffic_generator,
            self.handover_manager,
            self.device_manager,
            self.sleep_mode_manager,
            step_number,
        )

        self.kpi_handler.update_kpis(transmitted_bits_per_device)
        self.sleep_xapp(step_number)

    def run_simulation(self):
        self.kpi_handler.start_clock()
        for step in range(0, self.simulation_length_ms):
            self.step(step_number=step)
            
        self.kpi_handler.end_clock()
        self.kpi_handler.get_elapsed_time()