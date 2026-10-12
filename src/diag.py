"""
Sanity checks for the patched simulator. Run on your real map:   python sim_diagnostics.py
Expected on a healthy setup (see comments); anything far off means a modelling problem upstream of the RL.
"""
import numpy as np
from simulator import Simulator

INT = 10
sim = Simulator(19, 200, 1000, seed=5, topology_seed=0)
gh, rcm, ue = sim.geometry_helper, sim.radio_channel_model, np.arange(200)
bs = sim.base_station_manager.get_all_positions()
out = ~gh.is_indoor_mask_vector

print(f"BS heights {bs[:,2].min():.0f}-{bs[:,2].max():.0f} m | all on rooftops: {(gh.find_host_buildings(bs[:, :2]) >= 0).all()} "
      f"| placement coverage {sim.network_topology_helper.coverage_fraction:.2f}")
print(f"indoor UEs {gh.is_indoor_mask_vector.mean():.2f} | outdoor LoS fraction {gh.is_line_of_sight_mask_matrix[:, out].mean():.2f}   (should NOT be 0)")

d, pl = gh.distance_matrix_meters_matrix, rcm.path_loss_matrix
m = gh.is_line_of_sight_mask_matrix & out[None, :] & (d > 80) & (d < 120)
if m.any():
    print(f"LoS path loss near 100 m: {np.median(pl[m]):.1f} dB   (free space @6 GHz = 88 dB, expect ~85-95)")
srv = sim.handover_manager.get_serving_sector_indices()
sinr = rcm.snr_dbm_matrix_per_slot[srv, ue]
print(f"serving SINR median {np.median(sinr):.1f} dB, p10 {np.percentile(sinr,10):.1f}, at 35 dB cap {np.mean(sinr>=35):.2f}   (cap fraction should be small)")

def episode(sleep_mode, frac, seed=7):
    s = Simulator(19, 200, 1000, seed=seed, topology_seed=0)
    for k in np.random.default_rng(0).permutation(57)[:int(57 * frac)]:
        s.sleep_mode_manager.set_sleep_mode(int(k), sleep_mode, s.sector_manager)
    rows = []
    for k in range(100):
        for j in range(INT): s.step(k * INT + j)
        rows.append(s.pop_interval_kpis(8.0))
    g = lambda key: np.mean([r[key] for r in rows])
    return g("sla_satisfaction"), g("energy_saving"), g("avg_power_w"), g("mean_throughput_mbps")

print("\nsleep policy            | SLA sat | saving | power W | thr Mbps   (SLA sat must fall and saving rise as more sleeps)")
for name, mode, frac in [("all active", 0, 0.0), ("25% in SM1", 1, 0.25), ("50% in SM1", 1, 0.5), ("50% in SM3", 3, 0.5), ("90% in SM3", 3, 0.9)]:
    print(f"{name:24s}| {episode(mode, frac)[0]:.3f}   | {episode(mode, frac)[1]:.3f}  | {episode(mode, frac)[2]:8.0f} | {episode(mode, frac)[3]:.2f}")

a = Simulator(19, 200, 1000, seed=1, topology_seed=0); b = Simulator(19, 200, 1000, seed=1, topology_seed=0)
for s in range(300): a.step(s); b.step(s)
print("\nsame seed -> identical trajectory:", np.array_equal(a.kpi_handler.total_transmitted_bits_per_device, b.kpi_handler.total_transmitted_bits_per_device))