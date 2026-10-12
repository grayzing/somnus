import numpy as np

from sleep_mode_manager import SleepModeManager


class RadioUnitPowerConsumptionHelper:
    """
    O-RU power model (Usman et al. profile scales), vectorised and generalised:

      * `sectors_per_ru` consecutive sectors share one physical RU (default 3 = one site).  The digital
        unit runs at the shallowest sleep mode among the sectors of its RU, and its power is split evenly
        across those sectors (same rule as before; the old hard-wired "sectors 0-2 are the macro" is gone).
      * `ru_type`: "macro" (64T64R tables) or "micro" (4T4R tables) for all RUs.
      * Load dependence: in ACTIVE mode the RF power is
            rf_active * ((1 - f) + f * utilisation),   f = load_proportional_fraction
        so an idle-but-awake sector costs (1-f) of full RF power.  Sleep modes are load independent.
      * float32 throughout (float16 had ~8-16 W resolution at the 20-70 kW totals).
    Columns of every table: [Active, SM1, SM2, SM3, SM4]
    """
    TABLES = {
        "micro": {"rf": [365.0, 61.0, 25.0, 18.0, 10.0], "digital": [32.0, 27.0, 15.0, 10.0, 5.0]},
        "macro": {"rf": [1200.0, 270.0, 130.0, 75.0, 25.0], "digital": [150.0, 50.0, 30.0, 15.0, 10.0]},
    }

    def __init__(self, num_sectors: int, sectors_per_ru: int = 3, ru_type: str = "macro",
                 load_proportional_fraction: float = 0.6):
        assert ru_type in self.TABLES
        self.num_sectors = num_sectors
        self.load_fraction = float(load_proportional_fraction)
        self.power_consumption_sector_matrix = np.zeros((num_sectors,), dtype=np.float32)

        self.sector_to_ru_map = (np.arange(num_sectors) // sectors_per_ru).astype(np.int32)
        self.num_physical_rus = int(self.sector_to_ru_map.max() + 1)
        self.sectors_per_ru_count = np.bincount(self.sector_to_ru_map, minlength=self.num_physical_rus)

        self.rf_power = np.array(self.TABLES[ru_type]["rf"], dtype=np.float32)
        self.digital_power = np.array(self.TABLES[ru_type]["digital"], dtype=np.float32)
        self._max_power_w = None
        self._fast_key = None
        self._fast_const_w = 0.0
        self._fast_coeff = np.zeros(num_sectors)

    # ------------------------------------------------------------------ components
    def _modes(self, sleep_mode_manager: SleepModeManager) -> np.ndarray:
        return np.clip(np.asarray(sleep_mode_manager.sector_sleep_mode_matrix, dtype=np.int64), 0, 4)

    def calculate_digital_components_power(self, sleep_mode_manager: SleepModeManager) -> np.ndarray:
        modes = self._modes(sleep_mode_manager)
        ru_modes = np.full(self.num_physical_rus, 4, dtype=np.int64)
        np.minimum.at(ru_modes, self.sector_to_ru_map, modes)       # shallowest mode per RU
        per_ru = self.digital_power[ru_modes]
        return (per_ru[self.sector_to_ru_map] / self.sectors_per_ru_count[self.sector_to_ru_map]).astype(np.float32)

    def calculate_rf_processing_power(self, sleep_mode_manager: SleepModeManager, utilization=None) -> np.ndarray:
        modes = self._modes(sleep_mode_manager)
        rf = self.rf_power[modes].copy()
        if utilization is not None and self.load_fraction > 0:
            util = np.clip(np.asarray(utilization, dtype=np.float32), 0.0, 1.0)
            scale = (1.0 - self.load_fraction) + self.load_fraction * util
            rf = np.where(modes == 0, rf * scale, rf)
        return rf.astype(np.float32)

    def calculate_analog_components_power(self, sleep_mode_manager: SleepModeManager, utilization=None) -> np.ndarray:
        return 0.25 * self.calculate_rf_processing_power(sleep_mode_manager, utilization)

    def calculate_power_amplifier_power(self, sleep_mode_manager: SleepModeManager, utilization=None) -> np.ndarray:
        return 0.75 * self.calculate_rf_processing_power(sleep_mode_manager, utilization)

    # ------------------------------------------------------------------ totals
    def calculate_total_power(self, sleep_mode_manager: SleepModeManager, utilization=None) -> np.float32:
        analog = self.calculate_analog_components_power(sleep_mode_manager, utilization)
        digital = self.calculate_digital_components_power(sleep_mode_manager)
        pa = self.calculate_power_amplifier_power(sleep_mode_manager, utilization)
        self.power_consumption_sector_matrix = (analog + digital + pa).astype(np.float32)
        return np.float32(np.sum(self.power_consumption_sector_matrix, dtype=np.float64))

    def calculate_total_power_fast(self, sleep_mode_manager: SleepModeManager, utilization) -> float:
        """
        Same total as calculate_total_power (analog 25 % + PA 75 % = 100 % of RF), but affine in utilisation:
            P = const(sleep state) + coeff(sleep state) . utilisation
        The const/coeff pair is rebuilt only when the sleep-mode vector changes, so the per-step cost is one
        dot product.  `utilization` must lie in [0, 1] (the scheduler guarantees it).
        """
        key = np.asarray(sleep_mode_manager.sector_sleep_mode_matrix).tobytes()
        if key != self._fast_key:
            modes = self._modes(sleep_mode_manager)
            rf = self.rf_power[modes].astype(np.float64)
            active = modes == 0
            f = self.load_fraction
            self._fast_const_w = float(np.where(active, rf * (1.0 - f), rf).sum()
                                       + self.calculate_digital_components_power(sleep_mode_manager).sum(dtype=np.float64))
            self._fast_coeff = np.where(active, rf * f, 0.0)
            self._fast_key = key
        return self._fast_const_w + float(np.dot(self._fast_coeff, utilization))

    @property
    def max_power_w(self) -> float:
        """Power with every sector ACTIVE at 100 % utilisation (reference for the saving ratio)."""
        if self._max_power_w is None:
            class _AllActive:
                sector_sleep_mode_matrix = np.zeros(self.num_sectors, dtype=np.int8)
            self._max_power_w = float(self.calculate_total_power(_AllActive(), np.ones(self.num_sectors)))
        return self._max_power_w

    def energy_saving_ratio(self, power_w: float) -> float:
        return float(1.0 - power_w / self.max_power_w)

    def calculate_energy_efficiency(self, sleep_mode_manager: SleepModeManager, kpi_handler, step: int) -> float:
        throughput_mbps = np.sum(kpi_handler.calculate_throughput_mbps(step))
        total_power_watts = self.calculate_total_power(sleep_mode_manager)
        return 0.0 if total_power_watts == 0 else float(throughput_mbps / total_power_watts)