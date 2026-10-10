import numpy as np

from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from geometry_helper import GeometryHelper
from sleep_mode_manager import SleepModeManager

from scipy.spatial.distance import cdist

class RadioChannelModel:
    def __init__(self, num_sectors: int, num_devices: int, seed=24) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

        self.path_loss_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.directional_gain_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.received_power_dbm_matrix_per_resource_element: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.snr_dbm_matrix_per_slot: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)
        self.spectral_efficiency_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float32)

        self.rng = np.random.default_rng(seed)

        self._3gpp_prb_table = {
            'FR1': {
                15: {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 40: 216, 50: 270},
                30: {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106, 50: 133, 60: 162, 70: 189, 80: 217, 90: 245, 100: 273},
                60: {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 40: 51, 50: 65, 60: 79, 70: 93, 80: 107, 90: 121, 100: 135}
            },
            'FR2': {
                60:  {50: 66, 100: 132, 200: 264},
                120: {50: 32, 100: 66,  200: 132, 400: 264}
            }
        }

    def _get_3gpp_prbs(self, center_freq_ghz: float, numerology: int, bandwidth_mhz: float) -> int:
        fr = 'FR2' if center_freq_ghz > 24.0 else 'FR1'
        scs = 15 * (2 ** numerology)
        bw = int(round(bandwidth_mhz))
        
        try:
            return self._3gpp_prb_table[fr][scs][bw]
        except KeyError:
            import warnings
            fallback_prb = int(bandwidth_mhz * 5)
            warnings.warn(f"Config combination (FR: {fr}, SCS: {scs}kHz, BW: {bw}MHz) not found in 3GPP spec tables. Falling back to linear calculation ({fallback_prb} PRBs).")
            return fallback_prb

    def update_spectral_efficiency_matrix(self, sleep_mode_manager: SleepModeManager) -> None:
        snr_thresholds = np.array([-6.0, -4.0, -2.0, 1.0, 3.0, 5.5, 8.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0, 29.0])
        spectral_efficiencies = np.array([
            0.0000, 0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 
            2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063
        ])
        cqi_matrix = np.digitize(self.snr_dbm_matrix_per_slot, snr_thresholds)
        allocated_spectral_efficiency = spectral_efficiencies[cqi_matrix]
        self.spectral_efficiency_matrix = allocated_spectral_efficiency
        sleeping_indices = sleep_mode_manager.get_sector_sleep_mode_indices()
        if len(sleeping_indices) > 0:
            self.spectral_efficiency_matrix[sleeping_indices, :] = 0.0

    def update_directional_gain_matrix(self, geometry_helper: GeometryHelper, sector_manager: SectorManager, device_manager: DeviceManager):
        vertical_attenuation_db: np.ndarray = np.minimum(
            12*np.square(geometry_helper.relative_zenith_angle_deg_matrix / sector_manager.vertical_beamwidth_deg_matrix[:, np.newaxis]),
            30
        )

        horizontal_attenuation_db: np.ndarray = np.minimum(
            12*np.square(geometry_helper.relative_azimuth_angle_deg_matrix / sector_manager.horizontal_beamwidth_deg_matrix[:, np.newaxis]),
            sector_manager.front_to_back_ratio_matrix[:, np.newaxis]
        )

        self.directional_gain_matrix = -1 * np.minimum(
            vertical_attenuation_db + horizontal_attenuation_db,
            sector_manager.front_to_back_ratio_matrix[:, np.newaxis]
        )

    def update_path_loss_matrix(self, geometry_helper: GeometryHelper, sector_manager: SectorManager, base_station_manager: BaseStationManager, device_manager: DeviceManager):
        device_positions: np.ndarray = device_manager.get_all_positions() # U x 3
        sector_positions: np.ndarray = base_station_manager.get_all_positions()[sector_manager.sector_parent_base_station_vector] # M x 3
        
        breakpoint_distance_matrix: np.ndarray = 4 * device_positions[:, 2][np.newaxis, :] * sector_positions[:, 2][:, np.newaxis] * sector_manager.center_freq_ghz_matrix[:, np.newaxis].astype(np.float32) * 1e9 / 3e8

        breakpoint_distance_mask_matrix: np.ndarray = (10 <= geometry_helper.distance_matrix_meters_matrix) & (geometry_helper.distance_matrix_meters_matrix < breakpoint_distance_matrix)

        base_pathloss_line_of_sight_matrix_one: np.ndarray = 28.0 + 22 * np.log10(geometry_helper.distance_matrix_meters_matrix) + 20 * np.log10(sector_manager.center_freq_ghz_matrix[:, np.newaxis])
        
        pathloss_line_of_sight_matrix: np.ndarray = base_pathloss_line_of_sight_matrix_one + breakpoint_distance_mask_matrix * (18 + np.log10(geometry_helper.distance_matrix_meters_matrix) - 9 * np.log10(np.square(breakpoint_distance_matrix) + np.square(sector_positions[:, 2][:, np.newaxis] - 1 - device_positions[:, 2][np.newaxis, :] - 1))) + self.rng.normal(0, 4.4, size=(self.num_sectors, self.num_devices)).astype(np.float32)

        pathloss_nlos_matrix: np.ndarray = 13.54 + 39.08 * np.log10(geometry_helper.distance_matrix_meters_matrix) + 20 * np.log10(sector_manager.center_freq_ghz_matrix[:, np.newaxis]) - 9.5 * np.log10(np.square(breakpoint_distance_matrix) + np.square(sector_positions[:, 2][:, np.newaxis] - 1 - device_positions[:, 2][np.newaxis, :] - 1)) + self.rng.normal(0, 6, size=(self.num_sectors, self.num_devices)).astype(np.float32)

        # Use GIS-computed line-of-sight mask from GeometryHelper
        is_line_of_sight_mask_matrix: np.ndarray = geometry_helper.is_line_of_sight_mask_matrix
        
        # Bring in 2D indoor mask and broadcast to (num_sectors, num_devices)
        is_indoor_mask_matrix: np.ndarray = geometry_helper.is_indoor_mask_vector[np.newaxis, :]

        # 3GPP 38.901 O2I Penetration loss model
        # Base building penetration loss + inside penetration distance loss (~20-25 dB total average)
        # Added alongside the standard deviation for log-normal shadow fading of penetration (sigma = ~6.5)
        o2i_penetration_loss: np.ndarray = 25.0 + self.rng.normal(0, 6.5, size=(self.num_sectors, self.num_devices)).astype(np.float32)

        # Vectorized assignment: If indoor -> NLOS + O2I Loss. Else If LoS -> LoS Loss. Else -> NLOS Loss
        self.path_loss_matrix = np.where(
            is_indoor_mask_matrix,
            pathloss_nlos_matrix + o2i_penetration_loss,
            np.where(
                is_line_of_sight_mask_matrix,
                pathloss_line_of_sight_matrix,
                pathloss_nlos_matrix
            )
        ) 

    def update_received_power_matrix_per_resource_element(self, sector_manager: SectorManager):
        num_subcarriers = np.zeros((self.num_sectors, 1), dtype=np.float32)
        for sector_idx in range(self.num_sectors):
            num_subcarriers[sector_idx] = 10*np.log10(12 * self._get_3gpp_prbs(sector_manager.center_freq_ghz_matrix[sector_idx], sector_manager.sector_numerology_matrix[sector_idx], sector_manager.bandwidth_mhz_matrix[sector_idx]))
        self.received_power_dbm_matrix_per_resource_element = sector_manager.tx_power_dbm_matrix[:, np.newaxis] - self.path_loss_matrix + self.directional_gain_matrix - num_subcarriers

    def update_snr_dbm_matrix_per_slot(self, sector_manager: SectorManager):
        bandwidth_hz = sector_manager.bandwidth_mhz_matrix[:, np.newaxis] * 1e6
        thermal_noise_dbm = -174.0 + 10 * np.log10(bandwidth_hz) + 9.0  # Noise Figure = 9
        thermal_noise_mw = 10 ** (thermal_noise_dbm / 10)

        noise_denominator_mw = thermal_noise_mw
        noise_denominator_dbm = 10 * np.log10(noise_denominator_mw)

        self.snr_dbm_matrix_per_slot = np.minimum(
            35.0,
            self.received_power_dbm_matrix_per_resource_element - noise_denominator_dbm,
        )