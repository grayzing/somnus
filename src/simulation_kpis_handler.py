import numpy as np
from time import time


class SimulationKPIHandler:
    """
    Cumulative KPIs (since reset) plus a resettable *window* used for RL control intervals.

    SLA satisfaction (window): a UE with demand (bits queued at window start + new arrivals) is satisfied
    if it received >= `min_delivery_fraction` of that demand OR averaged >= `sla_mbps`.  UEs with no demand
    are ignored, so idle UEs no longer count as violations.
    """
    def __init__(self, num_devices: int, num_sectors: int):
        self.num_devices = num_devices
        self.num_sectors = num_sectors
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_time = 0.0
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self):
        self.total_transmitted_bits_per_device = np.zeros(self.num_devices, dtype=np.float64)
        self.window_start_backlog_bits = np.zeros(self.num_devices, dtype=np.float64)
        self._clear_window()
        self.last_window_rate_mbps = np.zeros(self.num_devices, dtype=np.float32)
        self.last_window_satisfied = np.ones(self.num_devices, dtype=np.float32)

    def _clear_window(self):
        self.window_offered_bits = np.zeros(self.num_devices, dtype=np.float64)
        self.window_delivered_bits = np.zeros(self.num_devices, dtype=np.float64)
        self.window_energy_j = 0.0
        self.window_ms = 0.0

    def start_clock(self):
        self.start_time = time()

    def end_clock(self):
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        return self.elapsed_time

    # ---------------------------------------------------------------- cumulative
    def update_kpis(self, total_transmitted_bits_per_device_vector):
        self.total_transmitted_bits_per_device += total_transmitted_bits_per_device_vector

    def calculate_throughput_mbps(self, total_simulated_ms):
        seconds = total_simulated_ms / 1000.0
        return (self.total_transmitted_bits_per_device / 1e6) / (seconds + 1e-6)

    def calculate_average_throughput_mbps(self, total_ms):
        return np.mean(self.calculate_throughput_mbps(total_ms))

    def calculate_throughput_percentile(self, percentile, total_ms):
        return np.percentile(self.calculate_throughput_mbps(total_ms), percentile)

    # ---------------------------------------------------------------- window
    def update_window(self, offered_bits, delivered_bits, power_w, step_ms):
        self.window_offered_bits += offered_bits
        self.window_delivered_bits += delivered_bits
        self.window_energy_j += float(power_w) * step_ms / 1000.0
        self.window_ms += step_ms

    def pop_window(self, sla_mbps: float, end_backlog_bits: np.ndarray, min_delivery_fraction: float = 0.95) -> dict:
        window_s = max(self.window_ms, 1e-9) / 1000.0
        demand = self.window_start_backlog_bits + self.window_offered_bits
        delivered = self.window_delivered_bits
        active = demand > 0
        rate_mbps = delivered / 1e6 / window_s
        satisfied = (~active) | (delivered >= min_delivery_fraction * demand) | (rate_mbps >= sla_mbps)

        out = {
            "sla_satisfaction": float(satisfied[active].mean()) if active.any() else 1.0,
            "num_active_ues": int(active.sum()),
            "mean_throughput_mbps": float(rate_mbps[active].mean()) if active.any() else 0.0,
            "avg_power_w": self.window_energy_j / window_s if self.window_ms > 0 else 0.0,
            "energy_j": self.window_energy_j,
            "backlog_bits": float(np.sum(end_backlog_bits)),
        }
        self.last_window_rate_mbps = rate_mbps.astype(np.float32)
        self.last_window_satisfied = satisfied.astype(np.float32)
        self.window_start_backlog_bits = np.array(end_backlog_bits, dtype=np.float64, copy=True)
        self._clear_window()
        return out

    def print_kpis(self, total_ms):
        avg = self.calculate_average_throughput_mbps(total_ms)
        p10 = self.calculate_throughput_percentile(10, total_ms)
        print("--- Simulation Performance ---")
        print(f"Wall-clock time: {self.get_elapsed_time():.2f} seconds")
        print(f"Average throughput (Mbps): {avg:.4f}")
        print(f"10th percentile throughput (Mbps): {p10:.4f}")