import warnings
import numpy as np

from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from geometry_helper import GeometryHelper
from sleep_mode_manager import SleepModeManager


SINR_THRESHOLDS_DB = np.array([-6.0, -4.0, -2.0, 1.0, 3.0, 5.5, 8.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0, 29.0])
SPECTRAL_EFFICIENCIES = np.array([
    0.0000, 0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063,
    2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063
])


class RadioChannelModel:
    """
    Link model following 3GPP TR 38.901 UMa:
      * LoS: PL1 for d2D <= d_BP', PL2 beyond (d_BP' = 4 (h_BS-1)(h_UT-1) fc / c), sigma_SF = 4 dB
      * NLoS: max(PL_LoS, PL'_NLoS), sigma_SF = 6 dB;  indoor UEs: NLoS + O2I penetration loss
      * Shadow fading is spatially correlated over time (AR(1) in UE displacement, d_corr = 37 m) instead of
        being redrawn i.i.d. every step; O2I loss is drawn once per UE.
      * `snr_dbm_matrix_per_slot` holds the per-resource-element SINR in dB (name kept for compatibility):
        S_RE / (N_RE + I_RE) where N_RE is the thermal noise in ONE subcarrier and I_RE is co-channel
        interference from other AWAKE sectors on the same carrier, weighted by their PRB utilisation.
    """
    SHADOW_DECORRELATION_M = 37.0
    NOISE_FIGURE_DB = 9.0
    SINR_CAP_DB = 35.0

    def __init__(self, num_sectors: int, num_devices: int, seed=24, interference: bool = True) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices
        self.interference = interference

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
        self.reset(seed)

    # ------------------------------------------------------------------ state
    def reset(self, seed=None) -> None:
        """Zero all link matrices and restart the shadowing process (optionally reseeding)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        shape = (self.num_sectors, self.num_devices)
        self.path_loss_matrix = np.zeros(shape, dtype=np.float32)
        self.directional_gain_matrix = np.zeros(shape, dtype=np.float32)
        self.received_power_dbm_matrix_per_resource_element = np.zeros(shape, dtype=np.float32)
        self.snr_dbm_matrix_per_slot = np.zeros(shape, dtype=np.float32)
        self.interference_dbm_matrix_per_resource_element = np.full(shape, -np.inf, dtype=np.float32)
        self.spectral_efficiency_matrix = np.zeros(shape, dtype=np.float32)
        # Serving-link quantities (what the scheduler / observations actually consume), shape (U,)
        self.serving_sinr_db = np.zeros(self.num_devices, dtype=np.float32)
        self.serving_spectral_efficiency = np.zeros(self.num_devices, dtype=np.float32)
        self._rx_mw = None
        self._ue_index = np.arange(self.num_devices)
        self._carrier_key = None
        self._same_carrier = None
        self._noise_key = None
        self._noise_re_mw = None
        self._shadow_z = None
        self._o2i_z = None
        self._last_ue_xy = None
        self._prb_cache_key = None
        self._prb_cache = None

    # ------------------------------------------------------------------ PRBs
    def _get_3gpp_prbs(self, center_freq_ghz: float, numerology: int, bandwidth_mhz: float) -> int:
        fr = 'FR2' if center_freq_ghz > 24.0 else 'FR1'
        scs = 15 * (2 ** int(numerology))
        bw = int(round(float(bandwidth_mhz)))
        try:
            return self._3gpp_prb_table[fr][scs][bw]
        except KeyError:
            fallback_prb = int(bandwidth_mhz * 5)
            warnings.warn(f"Config combination (FR: {fr}, SCS: {scs}kHz, BW: {bw}MHz) not found in 3GPP spec tables. "
                          f"Falling back to linear calculation ({fallback_prb} PRBs).")
            return fallback_prb

    def get_prbs_per_sector(self, sector_manager: SectorManager) -> np.ndarray:
        """Number of PRBs per sector (cached; recomputed only if the sector configuration changes)."""
        key = (sector_manager.center_freq_ghz_matrix.tobytes(),
               sector_manager.sector_numerology_matrix.tobytes(),
               sector_manager.bandwidth_mhz_matrix.tobytes())
        if key != self._prb_cache_key:
            self._prb_cache = np.array([
                self._get_3gpp_prbs(sector_manager.center_freq_ghz_matrix[i],
                                    sector_manager.sector_numerology_matrix[i],
                                    sector_manager.bandwidth_mhz_matrix[i])
                for i in range(self.num_sectors)], dtype=np.float32)
            self._prb_cache_key = key
        return self._prb_cache

    # ------------------------------------------------------------------ SE
    def update_spectral_efficiency_matrix(self, sleep_mode_manager: SleepModeManager) -> None:
        cqi_matrix = np.digitize(self.snr_dbm_matrix_per_slot, SINR_THRESHOLDS_DB)
        se = SPECTRAL_EFFICIENCIES[cqi_matrix].astype(np.float32)
        sleeping = np.asarray(sleep_mode_manager.sector_sleep_mode_matrix) > 0
        se[sleeping, :] = 0.0
        self.spectral_efficiency_matrix = se

    # ------------------------------------------------------------------ antenna
    def update_directional_gain_matrix(self, geometry_helper: GeometryHelper, sector_manager: SectorManager, device_manager: DeviceManager):
        vertical_attenuation_db = np.minimum(
            12 * np.square(geometry_helper.relative_zenith_angle_deg_matrix / sector_manager.vertical_beamwidth_deg_matrix[:, np.newaxis]),
            30
        )
        horizontal_attenuation_db = np.minimum(
            12 * np.square(geometry_helper.relative_azimuth_angle_deg_matrix / sector_manager.horizontal_beamwidth_deg_matrix[:, np.newaxis]),
            sector_manager.front_to_back_ratio_matrix[:, np.newaxis]
        )
        self.directional_gain_matrix = (-1 * np.minimum(
            vertical_attenuation_db + horizontal_attenuation_db,
            sector_manager.front_to_back_ratio_matrix[:, np.newaxis]
        )).astype(np.float32)

    # ------------------------------------------------------------------ shadowing
    def _update_shadow_state(self, ue_xy: np.ndarray) -> None:
        shape = (self.num_sectors, self.num_devices)
        if self._shadow_z is None or self._shadow_z.shape != shape:
            self._shadow_z = self.rng.standard_normal(shape)
            self._o2i_z = self.rng.standard_normal(self.num_devices)
        elif self._last_ue_xy is not None:
            disp = np.linalg.norm(ue_xy - self._last_ue_xy, axis=1)
            rho = np.exp(-disp / self.SHADOW_DECORRELATION_M)[np.newaxis, :]
            self._shadow_z = rho * self._shadow_z + np.sqrt(1.0 - rho ** 2) * self.rng.standard_normal(shape)
        self._last_ue_xy = ue_xy.copy()

    # ------------------------------------------------------------------ path loss
    def update_path_loss_matrix(self, geometry_helper: GeometryHelper, sector_manager: SectorManager, base_station_manager: BaseStationManager, device_manager: DeviceManager):
        ue_pos = device_manager.get_all_positions().astype(np.float64)                    # U x 3
        sec_pos = base_station_manager.get_all_positions()[
            sector_manager.sector_parent_base_station_vector].astype(np.float64)          # M x 3
        h_bs = sec_pos[:, 2][:, None]
        h_ue = ue_pos[:, 2][None, :]

        d3 = np.maximum(geometry_helper.distance_matrix_meters_matrix.astype(np.float64), 1.0)
        dz = h_bs - h_ue
        d2 = np.sqrt(np.maximum(d3 ** 2 - dz ** 2, 0.0))
        d3 = np.maximum(d3, 10.0)                                                         # model validity: d >= 10 m
        fc = sector_manager.center_freq_ghz_matrix.astype(np.float64)[:, None]

        # Breakpoint uses effective heights h' = h - 1 m (38.901 Table 7.4.1-1, note 1)
        d_bp = 4.0 * np.maximum(h_bs - 1.0, 0.1) * np.maximum(h_ue - 1.0, 0.1) * fc * 1e9 / 3e8

        pl1 = 28.0 + 22.0 * np.log10(d3) + 20.0 * np.log10(fc)
        pl2 = 28.0 + 40.0 * np.log10(d3) + 20.0 * np.log10(fc) - 9.0 * np.log10(d_bp ** 2 + dz ** 2)
        pl_los = np.where(d2 <= d_bp, pl1, pl2)
        pl_nlos_prime = 13.54 + 39.08 * np.log10(d3) + 20.0 * np.log10(fc) - 0.6 * (h_ue - 1.5)
        pl_nlos = np.maximum(pl_los, pl_nlos_prime)

        los = geometry_helper.is_line_of_sight_mask_matrix
        indoor = geometry_helper.is_indoor_mask_vector[np.newaxis, :]
        use_los = los & ~indoor

        self._update_shadow_state(ue_pos[:, :2])
        sigma = np.where(use_los, 4.0, 6.0)
        o2i = np.where(indoor, 25.0 + 6.5 * self._o2i_z[np.newaxis, :], 0.0)

        self.path_loss_matrix = (np.where(use_los, pl_los, pl_nlos) + sigma * self._shadow_z + o2i).astype(np.float32)

    # ------------------------------------------------------------------ power / SINR
    def update_received_power_matrix_per_resource_element(self, sector_manager: SectorManager):
        num_subcarriers_db = 10 * np.log10(12.0 * self.get_prbs_per_sector(sector_manager))[:, np.newaxis]
        self.received_power_dbm_matrix_per_resource_element = (
            sector_manager.tx_power_dbm_matrix[:, np.newaxis] - self.path_loss_matrix
            + self.directional_gain_matrix - num_subcarriers_db
        ).astype(np.float32)
        self._rx_mw = 10 ** (self.received_power_dbm_matrix_per_resource_element.astype(np.float64) / 10)

    def update_serving_link(self, serving: np.ndarray, sector_manager: SectorManager,
                            sleep_mode_manager: SleepModeManager = None) -> None:
        """
        Per-UE SINR and spectral efficiency on the SERVING link only (O(U*M) instead of O(M*M*U)).
        Same model as update_snr_dbm_matrix_per_slot + update_spectral_efficiency_matrix, but nothing
        outside the serving entries is ever read by the scheduler or the observations.
        """
        if self._rx_mw is None:
            self._rx_mw = 10 ** (self.received_power_dbm_matrix_per_resource_element.astype(np.float64) / 10)
        ue = self._ue_index
        signal_mw = self._rx_mw[serving, ue]

        noise_key = sector_manager.sector_numerology_matrix.tobytes()
        if noise_key != self._noise_key:
            scs_hz = 15e3 * (2.0 ** sector_manager.sector_numerology_matrix.astype(np.float64))
            self._noise_re_mw = 10 ** ((-174.0 + 10 * np.log10(scs_hz) + self.NOISE_FIGURE_DB) / 10)
            self._noise_key = noise_key
        denom = self._noise_re_mw[serving]

        sleeping = None
        if sleep_mode_manager is not None:
            sleeping = np.asarray(sleep_mode_manager.sector_sleep_mode_matrix) > 0
        if self.interference:
            carrier_key = sector_manager.center_freq_ghz_matrix.tobytes()
            if carrier_key != self._carrier_key:
                f = sector_manager.center_freq_ghz_matrix
                self._same_carrier = ((f[:, None] == f[None, :]) & ~np.eye(self.num_sectors, dtype=bool)).astype(np.float64)
                self._carrier_key = carrier_key
            load = np.clip(np.asarray(sector_manager.sector_physical_resource_block_utilization, dtype=np.float64), 0.0, 1.0)
            active_load = load if sleeping is None else load * ~sleeping
            weights = self._same_carrier[serving] * active_load[None, :]                 # (U, M)
            denom = denom + np.einsum('um,mu->u', weights, self._rx_mw)

        sinr_db = np.minimum(self.SINR_CAP_DB, 10 * np.log10(signal_mw / denom))
        se = SPECTRAL_EFFICIENCIES[np.searchsorted(SINR_THRESHOLDS_DB, sinr_db, side='right')]
        if sleeping is not None:
            se = np.where(sleeping[serving], 0.0, se)
        self.serving_sinr_db = sinr_db.astype(np.float32)
        self.serving_spectral_efficiency = se.astype(np.float32)

    def update_snr_dbm_matrix_per_slot(self, sector_manager: SectorManager, sleep_mode_manager: SleepModeManager = None):
        """Per-RE SINR (dB). Sleeping sectors neither serve nor interfere."""
        scs_hz = 15e3 * (2.0 ** sector_manager.sector_numerology_matrix.astype(np.float64))
        noise_re_mw = 10 ** ((-174.0 + 10 * np.log10(scs_hz) + self.NOISE_FIGURE_DB) / 10)[:, np.newaxis]

        p_mw = 10 ** (self.received_power_dbm_matrix_per_resource_element.astype(np.float64) / 10)
        interference_mw = np.zeros_like(p_mw)
        if self.interference:
            awake = np.ones(self.num_sectors) if sleep_mode_manager is None else \
                (np.asarray(sleep_mode_manager.sector_sleep_mode_matrix) == 0).astype(np.float64)
            load = np.clip(np.asarray(sector_manager.sector_physical_resource_block_utilization, dtype=np.float64), 0.0, 1.0)
            freq = sector_manager.center_freq_ghz_matrix
            same_carrier = (freq[:, None] == freq[None, :]) & ~np.eye(self.num_sectors, dtype=bool)
            weights = same_carrier * (awake * load)[None, :]                    # (victim i, interferer j)
            interference_mw = weights @ p_mw                                    # (M, U)

        with np.errstate(divide='ignore'):
            self.interference_dbm_matrix_per_resource_element = (10 * np.log10(interference_mw)).astype(np.float32)
        sinr_db = 10 * np.log10(p_mw / (noise_re_mw + interference_mw))
        self.snr_dbm_matrix_per_slot = np.minimum(self.SINR_CAP_DB, sinr_db).astype(np.float32)