import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch
from simulator import Simulator
from power_consumption_helper import RadioUnitPowerConsumptionHelper  # Import our helper

NUM_AGENTS = 7

# --- Simulator Classes with Step-by-Step Logging ---

class BaseSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleeping_history = []
        self.energy_efficiency_history = []  # Track EE per step
        
        # Instantiate the power consumption helper
        self.power_helper = RadioUnitPowerConsumptionHelper(num_sectors=self.num_sectors)
    
    def step(self, step_number):
        super().step(step_number)
        # Record number of sectors in a sleep mode (> 0)
        num_sleeping = np.count_nonzero(self.sleep_mode_manager.sector_sleep_mode_matrix)
        self.sleeping_history.append(num_sleeping)
        
        # Calculate energy efficiency (Mbits/Joule) for the current step
        step_ee = self.power_helper.calculate_energy_efficiency(
            self.sleep_mode_manager, 
            self.kpi_handler, 
            step=step_number
        )
        self.energy_efficiency_history.append(step_ee)

class EnhancedSleepSimulator(BaseSimulator):
    def sleep_xapp(self, curr_step):
        for sector in range(self.num_sectors):
            if self.sleep_mode_manager.sector_sleep_mode_matrix[sector] == 0:
                attached_devices_indices = np.where(self.handover_manager.sector_device_association_matrix[sector,:] == 0)
                average_throughput_of_sector = self.kpi_handler.calculate_throughput_mbps(curr_step)[attached_devices_indices].mean()
                if self.sector_manager.sector_physical_resource_block_utilization[sector] < 0.5 and average_throughput_of_sector < 20.0:
                    for device in attached_devices_indices:
                        rsrp_device_slice = self.radio_channel_model.received_power_dbm_matrix_per_resource_element[:,device]
                        prb_utilization = self.sector_manager.sector_physical_resource_block_utilization[:,np.newaxis]
                        rsrp_device_mask = (rsrp_device_slice >= -90) & (prb_utilization <= 0.9)
                        rsrp_device_mask[sector, :] = False
                        first_sector = np.argmax(np.any(rsrp_device_mask, axis=1))
                        self.handover_manager.manual_handover(first_sector, device, self.sector_manager, self.device_manager, self.radio_channel_model, self.sleep_mode_manager)
            if np.all(self.handover_manager.sector_device_association_matrix[sector, :] == 0):
                self.sleep_mode_manager.set_sleep_mode(sector_id=sector, sleep_mode=1, sector_manager=self.sector_manager)

class BasicSleepSimulator(BaseSimulator):
    def sleep_xapp(self, curr_step):
        for sector in range(self.num_sectors):
            if np.all(self.handover_manager.sector_device_association_matrix[sector, :] == 0) and self.sleep_mode_manager.sector_sleep_mode_matrix[sector] == 0:
                self.sleep_mode_manager.set_sleep_mode(sector_id=sector, sleep_mode=1, sector_manager=self.sector_manager)

class RandomSleepSimulator(BaseSimulator):
    def sleep_xapp(self, curr_step):
        self.sleep_mode_manager.set_sleep_mode(sector_id=self.rng.integers(0,self.num_sectors), sleep_mode=self.rng.integers(0,4), sector_manager=self.sector_manager)

class VDNSleepSimulator(BaseSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = torch.load("q_net.pth", weights_only=False, map_location=torch.device('cpu'))

    def sleep_xapp(self, curr_step):
        for agent in range(0,NUM_AGENTS):
            obs = self.take_observation(agent, self)
            action = int(torch.argmax(self.q_net(obs)))
            sector_id = int(agent * 3 + (action // 4))
            self.sleep_mode_manager.set_sleep_mode(sector_id=sector_id, sleep_mode=int(action % 4), sector_manager=self.sector_manager)

    def take_observation(self, agent_id, simulator):
        return torch.from_numpy(np.concatenate(
        [np.stack(
            [
                simulator.radio_channel_model.received_power_dbm_matrix_per_resource_element[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]],
                simulator.radio_channel_model.snr_dbm_matrix_per_slot[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]]
            ],
            axis=0
            ).flatten(),
            simulator.sleep_mode_manager.sector_sleep_mode_matrix[simulator.sector_manager.neighboring_sectors_indices_matrix[agent_id]]
        ]
        )).to(torch.float32)

class NormalSimulator(BaseSimulator):
    def sleep_xapp(self, curr_step):
        pass  # Baseline without sleeping

class ITQoSLBSleepSimulator(BaseSimulator):
    def __init__(self, *args, alpha_u: float = 0.7, beta: float = 0.7, rsrp_threshold_dbm: float = -90.0, **kwargs):
        """
        Iterative QoS-Aware Load-Based (IT-QOS-LB) Sleep Mode Simulator.
        
        Args:
            alpha_u (float): Throughput satisfaction parameter (GBR fraction of All On rate)[cite: 9].
            beta (float): Minimum service reliability parameter (QoS threshold ratio)[cite: 9].
            rsrp_threshold_dbm (float): RSRP threshold to consider a UE within a sector's coverage.
        """
        super().__init__(*args, **kwargs)
        self.alpha_u = alpha_u
        self.beta = beta
        self.rsrp_threshold_dbm = rsrp_threshold_dbm
        
        # Keep track of the baseline "All On" throughputs for QoS evaluation[cite: 9]
        self.all_on_throughputs = None

    def _compute_qos_satisfaction_ratio(self, current_throughputs: np.ndarray) -> float:
        """
        Calculates the fraction of UEs satisfying their GBR constraint (Eq. 13 & 14)[cite: 9].
        """
        if self.all_on_throughputs is None:
            # Fallback if baseline is not established
            return 1.0
        
        # Identify satisfied UEs: achieved rate under SM >= alpha_u * All On rate[cite: 9]
        satisfied_ue_mask = current_throughputs >= (self.alpha_u * self.all_on_throughputs)
        qos_satisfied_ratio = np.sum(satisfied_ue_mask) / self.num_devices
        return qos_satisfied_ratio

    def sleep_xapp(self, curr_step: int):
        """
        Executes the Iterative QoS-Aware Load-Based SMO Strategy (Algorithm 1)[cite: 9].
        """
        # --- Step 0: Establish "All On" reference rates if not done already ---
        # Note: In a real run, this evaluates baseline capacity. We can estimate or track it.
        if self.all_on_throughputs is None:
            # Estimate using the current step throughput as baseline if no All-On step is recorded
            self.all_on_throughputs = self.kpi_handler.calculate_throughput_mbps(curr_step)
            
        # Get active/serving connections (u_ij)
        # Association matrix shape: (num_sectors, num_devices). 1 if served, 0 otherwise.
        u_matrix = (self.handover_manager.sector_device_association_matrix == 1).astype(np.float32)
        
        # Determine coverage footprint (s_ij) based on RSRP threshold (excluding serving sector)
        rsrp_matrix = self.radio_channel_model.received_power_dbm_matrix_per_resource_element
        coverage_matrix = (rsrp_matrix >= self.rsrp_threshold_dbm).astype(np.float32)
        
        # s_ij is 1 if in coverage but NOT served[cite: 9]
        s_matrix = np.clip(coverage_matrix - u_matrix, 0, 1)
        
        # --- Step 1: Compute UE-centric load contribution (Eq. 16) ---
        # Sum of u_ij + s_ij per UE across all sectors[cite: 9]
        num_covering_sectors = np.sum(u_matrix + s_matrix, axis=0) 
        
        # Avoid division by zero for unserved/out-of-range UEs
        with np.errstate(divide='ignore', invalid='ignore'):
            l_ue = np.where(num_covering_sectors > 0, 1.0 / num_covering_sectors, 0.0)
            
        # --- Step 2: Compute overall Sector Loads (Eq. 17) ---
        # Sum of load contributions from primarily associated UEs[cite: 9]
        sector_loads = np.sum(u_matrix * l_ue[np.newaxis, :], axis=1)
        
        # Only target active sectors for sleep optimization
        active_sectors = np.where(self.sleep_mode_manager.sector_sleep_mode_matrix == 0)[0]
        if len(active_sectors) == 0:
            return
            
        # Sort candidate active sectors by load in ascending order[cite: 9]
        sorted_candidates = active_sectors[np.argsort(sector_loads[active_sectors])]
        
        # Maintain a rollback backup of sleep states and sector utilization
        original_sleep_states = self.sleep_mode_manager.sector_sleep_mode_matrix.copy()
        
        # --- Step 3: Iteratively deactivate sectors and check QoS (Algorithm 1) ---
        for sector in sorted_candidates:
            # Temporarily put the sector to sleep (SM1)[cite: 7, 9]
            self.sleep_mode_manager.sector_sleep_mode_matrix[sector] = 1
            
            # Recalculate immediate channel properties & throughput under the test state
            # Force Handover to find alternative sectors for affected UEs
            self.handover_manager.handover(
                self.sector_manager, self.device_manager, 
                self.radio_channel_model, self.sleep_mode_manager
            )
            
            # Predict throughput for the current step
            test_throughputs = self.kpi_handler.calculate_throughput_mbps(curr_step)
            
            # Evaluate system-wide QoS metric[cite: 9]
            psi_qos = self._compute_qos_satisfaction_ratio(test_throughputs)
            
            # If the QoS threshold is violated, roll back and stop further deactivations[cite: 9]
            if psi_qos < self.beta:
                # Reactivate sector and revert sleep states[cite: 9]
                self.sleep_mode_manager.sector_sleep_mode_matrix = original_sleep_states.copy()
                self.handover_manager.handover(
                    self.sector_manager, self.device_manager, 
                    self.radio_channel_model, self.sleep_mode_manager
                )
                break  # Stop further deactivations[cite: 9]
            else:
                # Deactivation is safe, commit and update original backup state
                original_sleep_states[sector] = 1
                self.sleep_mode_manager.set_sleep_mode(
                    sector_id=sector, sleep_mode=1, sector_manager=self.sector_manager
                )

# --- Batch Execution & Export ---
def run_experiment(SimulatorClass, method_name, n=20):
    all_data = []
    for i in range(n):
        sim = SimulatorClass(NUM_AGENTS, 200, 100, seed=5000+i)
        sim.run_simulation()
        
        tp = sim.kpi_handler.calculate_throughput_mbps(100)
        all_data.append({
            'Method': method_name,
            'Trial': i,
            'P10_Throughput': np.percentile(tp, 10),
            'Avg_Throughput': np.mean(tp),
            'Avg_Sleeping_Sectors': np.mean(sim.sleeping_history),
            'Avg_Energy_Efficiency': np.mean(sim.energy_efficiency_history)  # Add EE metric
        })
        del sim; gc.collect()
    return all_data

methods = {
    "ALL-ON": NormalSimulator, 
    "Liang et al.": EnhancedSleepSimulator, 
    "IT-QoS-LB": ITQoSLBSleepSimulator,
    "SM1": BasicSleepSimulator, 
    "VDN": VDNSleepSimulator
}

results = []
for name, cls in methods.items():
    print(f"Running 20 simulations for {name}...")
    results.extend(run_experiment(cls, name))

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("simulation_results.csv", index=False)
print("Data saved to simulation_results.csv")

# --- Accessible Visualization with Energy Efficiency ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
styles = [('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'D'), ('-', 'x')]

metrics_to_plot = ['P10_Throughput', 'Avg_Throughput', 'Avg_Energy_Efficiency']
titles = [
    "CDF: 10th Percentile Throughput (Mbps)", 
    "CDF: Average Throughput (Mbps)", 
    "CDF: Network Energy Efficiency (Mbits/Joule)"
]

for (name, cls), (ls, marker) in zip(methods.items(), styles):
    subset = df[df['Method'] == name]
    for i, col in enumerate(metrics_to_plot):
        data = np.sort(subset[col].values)
        axs[i].plot(data, np.linspace(0, 1, len(data)), label=name, ls=ls, marker=marker, markevery=2)

for i, ax in enumerate(axs):
    ax.set_title(titles[i])
    ax.legend()
    ax.grid(True)

plt.tight_layout()

# --- Sleeping Sectors Boxplot ---
plt.figure(figsize=(10, 5))
bp = plt.boxplot([df[df['Method']==m]['Avg_Sleeping_Sectors'] for m in methods.keys()], 
                 labels=methods.keys(), patch_artist=True)
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_hatch('//')

plt.title("Average Number of Sleeping Sectors per Method")
plt.show()