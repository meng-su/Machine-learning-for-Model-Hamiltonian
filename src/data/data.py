from Ising import IsingLattice
import MC
import numpy as np
import pandas as pd


def cycle_(lat_size,Jij,hii,data_set):
    temperature = np.random.uniform(high=10,low=1e-5,size=1)[0]
    ising_model = IsingLattice(lattice_size=lat_size,J=Jij,h=hii)
    lattice_configs, energy_records, magnetization_records \
        = MC.monte_carlo_simulation(ising_model, T = temperature, num_scans = 1000, num_scans_4_equilibrium = 1000, frequency_sweeps_to_collect_magnetization = 35, plot_result = False,print_info=False)
    lattice_configs = lattice_configs.tolist()
    energy_records = energy_records.tolist()
    magnetization_records = magnetization_records.tolist()
    data_set["lattice_configs"] = lattice_configs
    data_set["energy_records"] = energy_records
    data_set["T"] = temperature
    data_set["magnetization_records"] = magnetization_records


import multiprocessing

if __name__ == "__main__":
    # Define the number of tasks
    num_of_data = 5

    lat_size = 8
    Jij = 1.0
    hii = 0.0
    
    items1 = [lat_size for x in range(num_of_data)]
    items2 = [Jij for x in range(num_of_data)]
    items3 = [hii for x in range(num_of_data)]


    # Creating a shared dictionary
    manager = multiprocessing.Manager()
    data_set = manager.dict()

    # data_set = {"lattice_configs": [],"energy_records": [],"T": [],"magnetization_records": []}
    data_set["lattice_configs"] = []
    data_set["energy_records"] = []
    data_set["T"] = []
    data_set["magnetization_records"] = []

    # Create a process pool, using the number of cores of the CPU as the number of processes
    pool = multiprocessing.Pool()

    # Adds a task to the process pool
    results = []
    for i in range(num_of_data):
        results.append(pool.apply_async(cycle_, args=(items1[i], items2[i], items3[i], data_set)))

    # Wait for all tasks to complete
    pool.close()
    pool.join()
    

    # print(data_set)

    dataname = "./source/dataset.csv"

    dataset = pd.DataFrame(data_set,columns=["id","lattice_configs","energy_records","T","magnetization_records"])

    print(dataset.columns)

    dataset.to_csv(dataname,columns=dataset.columns,sep=',')
