import numpy as np
from abc import ABC, abstractmethod


class TrafficGenerator(ABC):
    """
    device_downlink_bits_matrix[u, t] = NEW bits arriving for device u during step t.
    backlog_bits[u]                   = bits queued at the base station (owned/updated by the scheduler).
    """
    def __init__(self, num_devices: int, window_length: int, seed: int = 24, step_ms: float = 1.0) -> None:
        self.num_devices = num_devices
        self.window_length = window_length      # in steps
        self.step_ms = step_ms
        self.device_downlink_bits_matrix: np.ndarray = np.zeros((num_devices, window_length), dtype=np.float32)
        self.backlog_bits: np.ndarray = np.zeros(num_devices, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self._arrivals_T = None
        self._zero_arrivals = np.zeros(num_devices, dtype=np.float64)

    def refresh_arrivals_cache(self) -> None:
        """(steps, devices) float64 copy so per-step access is a contiguous row. Call after editing the matrix by hand."""
        self._arrivals_T = np.ascontiguousarray(self.device_downlink_bits_matrix.T, dtype=np.float64)

    def arrivals_at(self, step: int) -> np.ndarray:
        """New bits for each device at `step`. Returns a READ-ONLY view; do not modify it."""
        if step < 0 or step >= self.window_length:
            return self._zero_arrivals
        if self._arrivals_T is None:
            self.refresh_arrivals_cache()
        return self._arrivals_T[step]

    @abstractmethod
    def generate_device_downlink_bits_matrix(self):
        raise NotImplementedError("Subclasses must implement this method.")


class BurstyTrafficGenerator(TrafficGenerator):
    """
    ON/OFF source with heavy-tailed (Pareto) burst/idle durations, specified in MILLISECONDS and
    rates in BITS PER SECOND (converted to bits per step internally).

      duration_ms = min_ms + scale_ms * Pareto(shape)        (numpy pareto == Lomax, mean 1/(shape-1))

    Defaults: ON ~ 50 ms + 100 ms*Pareto(2.5)  (mean ~117 ms), OFF ~ 100 ms + 300 ms*Pareto(1.5)
    (mean ~700 ms), i.e. roughly 15-25 % duty cycle at 5 Mbps -> ~1 Mbps mean offered load per UE
    (the old 5e6 was bits per STEP, i.e. 5 Gbps; it was clearly meant as bits per second).
    Every UE starts in a random state with a random residual duration so sources are not synchronised.
    """
    def __init__(self, num_devices, window_length,
                 on_shape=2.5, off_shape=1.5,
                 high_rate_bps=5e6, low_rate_bps=0.0,
                 on_min_ms=50.0, on_scale_ms=100.0,
                 off_min_ms=100.0, off_scale_ms=300.0,
                 seed=24, step_ms=1.0):
        super().__init__(num_devices, window_length, seed, step_ms)
        self.on_shape, self.off_shape = on_shape, off_shape
        self.high_rate_bps, self.low_rate_bps = high_rate_bps, low_rate_bps
        self.on_min_ms, self.on_scale_ms = on_min_ms, on_scale_ms
        self.off_min_ms, self.off_scale_ms = off_min_ms, off_scale_ms

    def _mean_ms(self, minimum, scale, shape):
        return minimum + (scale / (shape - 1.0) if shape > 1.0 else 10 * scale)

    def generate_device_downlink_bits_matrix(self):
        self.device_downlink_bits_matrix[:] = 0.0
        self.backlog_bits[:] = 0.0
        high = self.high_rate_bps * self.step_ms / 1000.0       # bits per step while ON
        low = self.low_rate_bps * self.step_ms / 1000.0
        mean_on = self._mean_ms(self.on_min_ms, self.on_scale_ms, self.on_shape)
        mean_off = self._mean_ms(self.off_min_ms, self.off_scale_ms, self.off_shape)
        p_on = mean_on / (mean_on + mean_off)

        for i in range(self.num_devices):
            t = 0
            is_on = self.rng.random() < p_on
            first = True
            while t < self.window_length:
                if is_on:
                    dur_ms = self.on_min_ms + self.on_scale_ms * self.rng.pareto(self.on_shape)
                else:
                    dur_ms = self.off_min_ms + self.off_scale_ms * self.rng.pareto(self.off_shape)
                steps = max(1, int(round(dur_ms / self.step_ms)))
                if first:                                        # random residual of the first period
                    steps = max(1, int(steps * self.rng.random()))
                    first = False
                end = min(t + steps, self.window_length)
                self.device_downlink_bits_matrix[i, t:end] = high if is_on else low
                t = end
                is_on = not is_on
        self.refresh_arrivals_cache()
        return self.device_downlink_bits_matrix