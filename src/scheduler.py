import numpy as np

from sector_manager import SectorManager
from radio_channel_model import RadioChannelModel
from device_manager import DeviceManager
from traffic_generator import TrafficGenerator
from handover_manager import HandoverManager
from sleep_mode_manager import SleepModeManager

from abc import ABC, abstractmethod

RESOURCE_ELEMENTS_PER_PRB_PER_SLOT = int(12 * 14 * 0.80)   # 12 subcarriers x 14 symbols, 20 % overhead


class PhysicalResourceBlockScheduler(ABC):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

    @abstractmethod
    def schedule(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")


class QueueAwareProportionalFairPhysicalResourceBlockScheduler(PhysicalResourceBlockScheduler):
    """
    Per sector: (1) one PRB to every backlogged attached UE in PF-metric order (while PRBs last), then
    (2) remaining PRBs greedily by PF metric (R/T_hist) * backlog^alpha.

    Queue model: `traffic_generator.backlog_bits` is the only queue.  The simulator adds each step's
    arrivals to it before calling schedule(); the scheduler drains it in place.  (The old design carried
    leftovers by mutating the arrival matrix one column ahead.)

    Capacity: one PRB carries SE * RE_per_PRB bits per slot, and a step of `step_ms` contains
    2^mu * step_ms slots for numerology mu (the old code counted exactly one slot per step regardless of mu).
    """
    def __init__(self, num_sectors: int, num_devices: int, step_ms: float = 1.0) -> None:
        super().__init__(num_sectors, num_devices)
        self.step_ms = step_ms
        self.alpha = 1.0
        self.ema_beta = 0.95
        self.reset()

    def reset(self) -> None:
        self.historical_throughput = np.ones(self.num_devices, dtype=np.float64)   # bits per step (EMA)

    def schedule(self, sector_manager: SectorManager, radio_channel_model: RadioChannelModel,
                 traffic_generator: TrafficGenerator, handover_manager: HandoverManager,
                 device_manager: DeviceManager, sleep_mode_manager: SleepModeManager, step: int):
        """Returns (allocated PRBs per sector (M,), transmitted bits per device (U,))."""
        M, U = self.num_sectors, self.num_devices
        backlog = traffic_generator.backlog_bits
        serving = handover_manager.get_serving_sector_indices()

        slots_per_step = (2.0 ** sector_manager.sector_numerology_matrix.astype(np.float64)) * self.step_ms
        rate_of_serving = (radio_channel_model.serving_spectral_efficiency
                           * (RESOURCE_ELEMENTS_PER_PRB_PER_SLOT * slots_per_step[serving]))      # bits per PRB, (U,)

        allocated = np.zeros(M, dtype=np.int64)
        tx_bits = np.zeros(U, dtype=np.float64)
        dev_alloc = np.zeros(U, dtype=np.int16)

        cand = np.flatnonzero((backlog > 0) & (rate_of_serving > 0))
        total_prbs = radio_channel_model.get_prbs_per_sector(sector_manager)
        if cand.size:
            sc = serving[cand]
            rate = rate_of_serving[cand]
            q = backlog[cand]
            metric = rate / self.historical_throughput[cand] * (q if self.alpha == 1.0 else np.power(q, self.alpha))

            # Rank UEs inside each sector by PF metric (descending): sort by (sector, -metric).
            order = np.lexsort((-metric, sc))
            cand, sc, rate, q = cand[order], sc[order], rate[order], q[order]
            K = cand.size
            is_start = np.empty(K, dtype=bool)
            is_start[0] = True
            is_start[1:] = sc[1:] != sc[:-1]
            start_idx = np.flatnonzero(is_start)
            grp = np.cumsum(is_start) - 1
            rank = np.arange(K) - start_idx[grp]
            size = np.diff(np.append(start_idx, K))                     # UEs per sector group
            prbs = total_prbs[sc[start_idx]].astype(np.int64)           # PRBs per sector group

            # Phase 1: one PRB to each UE in rank order while PRBs last.
            g1 = rank < prbs[grp]
            sent = np.where(g1, np.minimum(q, rate), 0.0)
            alloc = g1.astype(np.int64)
            rem = q - sent

            # Phase 2: remaining PRBs greedily in rank order.  Sequential "n_j = min(need_j, left)" equals
            # clip(left0 - sum_{i<j} need_i, 0, need_j).
            left0 = np.maximum(prbs - size, 0)
            need = np.where(rem > 0, np.ceil(rem / rate), 0.0).astype(np.int64)
            excl = np.cumsum(need) - need
            before = excl - excl[start_idx][grp]
            n2 = np.clip(left0[grp] - before, 0, need)
            sent += np.minimum(n2 * rate, rem)
            alloc += n2

            allocated = np.bincount(sc, weights=alloc, minlength=M).astype(np.int64)
            dev_alloc[cand] = alloc
            tx_bits[cand] = sent
            backlog -= tx_bits
            np.maximum(backlog, 0.0, out=backlog)

        self.historical_throughput = np.maximum(
            self.ema_beta * self.historical_throughput + (1 - self.ema_beta) * tx_bits, 1.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            utilization = np.where(total_prbs > 0, allocated / total_prbs, 0.05)
        utilization = np.maximum(0.05, utilization).astype(np.float32)
        utilization[np.asarray(sleep_mode_manager.sector_sleep_mode_matrix) > 0] = 0.01
        sector_manager.sector_physical_resource_block_utilization = utilization

        device_manager.device_physical_resource_block_allocation_vector = dev_alloc
        return allocated, tx_bits