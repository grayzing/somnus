from simulator import Simulator

if __name__ == "__main__":
    sim: Simulator = Simulator(19, 200, 5000)
    sim.run_simulation()

    sim.kpi_handler.print_kpis(5000)