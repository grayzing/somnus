import numpy as np
from sector_manager import SectorManager


class SleepModeManager:
    """
    0 - ACTIVE, 1 - SM1, 2 - SM2, 3 - SM3.  One "step" = one simulator step (1 ms by default).

    Semantics
      * Entering a sleep mode (from active or from another sleep mode) is immediate and starts a minimum
        dwell lock: the sector cannot change mode again until the lock expires.
      * Waking up (sleep -> ACTIVE) is NOT immediate for SM2/SM3: the sector stays in its sleep mode (so it
        serves nobody and creates no interference) for WAKE_DELAY_STEPS[mode] steps, and only then becomes
        ACTIVE inside tick().  Requests made while a sector is waking or locked are ignored.
    The totals (SM3: 5 wake + 10 dwell = 15) match the old hard-coded "5 + 10" constants.
    """
    MIN_DWELL_STEPS = {1: 1, 2: 2, 3: 10}
    WAKE_DELAY_STEPS = {1: 0, 2: 1, 3: 5}
    MAX_LOCK_STEPS = 15

    def __init__(self, num_sectors: int):
        self.num_sectors = num_sectors
        self.reset()

    def reset(self) -> None:
        self.sector_sleep_mode_matrix = np.zeros((self.num_sectors,), dtype=np.int8)
        self.sector_sleep_mode_countdown = np.zeros((self.num_sectors,), dtype=np.int32)   # dwell lock
        self.sector_wake_countdown = np.zeros((self.num_sectors,), dtype=np.int32)         # remaining wake delay
        self._waking = np.zeros((self.num_sectors,), dtype=bool)
        self.num_transitions = 0
        self.version = 0        # bumped whenever sector_sleep_mode_matrix changes (lets callers cache)

    def lock_remaining(self) -> np.ndarray:
        """Steps until the sector can accept a new command (max of dwell lock and wake delay)."""
        return np.maximum(self.sector_sleep_mode_countdown, self.sector_wake_countdown)

    def tick(self):
        self.sector_sleep_mode_countdown = np.maximum(self.sector_sleep_mode_countdown - 1, 0)
        if self._waking.any():
            self.sector_wake_countdown[self._waking] -= 1
            done = self._waking & (self.sector_wake_countdown <= 0)
            if done.any():
                self.version += 1
            self.sector_sleep_mode_matrix[done] = 0
            self.sector_wake_countdown[done] = 0
            self._waking[done] = False

    def set_sleep_mode(self, sector_id: int, sleep_mode: int, sector_manager: SectorManager = None) -> None:
        assert sleep_mode in (0, 1, 2, 3), "Invalid sleep mode"
        current = int(self.sector_sleep_mode_matrix[sector_id])
        if self._waking[sector_id] or current == sleep_mode:
            return
        if self.sector_sleep_mode_countdown[sector_id] > 0:
            return                                    # still inside the minimum dwell time

        if sleep_mode == 0:
            delay = self.WAKE_DELAY_STEPS[current]
            if delay == 0:
                self.sector_sleep_mode_matrix[sector_id] = 0
                self.version += 1
            else:
                self._waking[sector_id] = True
                self.sector_wake_countdown[sector_id] = delay
        else:
            self.sector_sleep_mode_matrix[sector_id] = sleep_mode
            self.sector_sleep_mode_countdown[sector_id] = self.MIN_DWELL_STEPS[sleep_mode]
            self.version += 1
        self.num_transitions += 1

    def get_sector_sleep_mode_indices(self) -> np.ndarray:
        """Sectors that are not serving (sleeping or still waking up)."""
        return np.where(self.sector_sleep_mode_matrix > 0)