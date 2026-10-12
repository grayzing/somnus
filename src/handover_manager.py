import numpy as np

from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel
from sleep_mode_manager import SleepModeManager

from abc import ABC, abstractmethod


class HandoverManager(ABC):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices
        self.reset()

    def reset(self) -> None:
        # [i][j] = 1 if sector i is giving downlink data to device j
        self.sector_device_association_matrix: np.ndarray = np.zeros((self.num_sectors, self.num_devices), dtype=np.int8)
        self.serving_sector_vector: np.ndarray = np.zeros(self.num_devices, dtype=np.int64)
        self._initialized = False

    def get_serving_sector_indices(self) -> np.ndarray:
        return self.serving_sector_vector

    @abstractmethod
    def handover(self, sector_manager: SectorManager, device_manager: DeviceManager, radio_channel_model: RadioChannelModel, sleep_mode_manager: SleepModeManager):
        raise NotImplementedError("Subclasses must implement this method.")

    def manual_handover(self, sector_id: int, device_id: int, sector_manager: SectorManager, device_manager: DeviceManager, radio_channel_model: RadioChannelModel, sleep_mode_manager: SleepModeManager):
        self.sector_device_association_matrix[:, device_id] = 0
        self.sector_device_association_matrix[sector_id][device_id] = 1
        self.serving_sector_vector[device_id] = sector_id


class RSRPBasedHandoverManager(HandoverManager):
    """
    Strongest-RSRP association among AWAKE sectors, with hysteresis: a UE keeps its serving sector unless
    that sector is asleep/waking, or another awake sector is better by more than `hysterisis` dB.
    `handover()` is idempotent for unchanged inputs, so callers only need to invoke it when received power or
    the sleep state changed.
    If every sector is asleep the UE attaches to its strongest sector anyway (it then gets zero spectral
    efficiency), instead of the old hard-wired "switch sectors 0-2 back on".
    """
    def __init__(self, num_sectors: int, num_devices: int, hysterisis: float = 3.0) -> None:
        super().__init__(num_sectors, num_devices)
        self.hysterisis = hysterisis

    def handover(self, sector_manager: SectorManager, device_manager: DeviceManager, radio_channel_model: RadioChannelModel, sleep_mode_manager: SleepModeManager):
        rx = radio_channel_model.received_power_dbm_matrix_per_resource_element
        ue = np.arange(self.num_devices)
        sleeping = np.asarray(sleep_mode_manager.sector_sleep_mode_matrix) > 0

        masked = np.where(sleeping[:, None], -np.inf, rx)
        best_awake = np.argmax(masked, axis=0)
        best_val = masked[best_awake, ue]
        none_awake = ~np.isfinite(best_val)
        best = np.where(none_awake, np.argmax(rx, axis=0), best_awake)

        prev = self.serving_sector_vector
        prev_ok = ~sleeping[prev] if self._initialized else np.zeros(self.num_devices, dtype=bool)
        stay = prev_ok & ~none_awake & (best_val - rx[prev, ue] < self.hysterisis)
        new_serving = np.where(stay, prev, best)

        self.sector_device_association_matrix = np.zeros((self.num_sectors, self.num_devices), dtype=np.int8)
        self.sector_device_association_matrix[new_serving, ue] = 1
        self.serving_sector_vector = new_serving.astype(np.int64)
        self._initialized = True