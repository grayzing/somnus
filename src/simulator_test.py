from simulator import Simulator

if __name__ == "__main__":
    sim: Simulator = Simulator(19, 200, 60 * 1000)
    sim.run_simulation()

    sim.kpi_handler.print_kpis(60 * 1000)