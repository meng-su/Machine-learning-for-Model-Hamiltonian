import time
import numpy as np

def scan_lattice(ising_lattice, T):
    
    for k in np.arange(ising_lattice.num_sites):
        # We choose a random site
        i = round(np.random.uniform(high = ising_lattice.lattice_size - 1, low = 0, size = 1)[0])
        j = round(np.random.uniform(high = ising_lattice.lattice_size - 1, low = 0, size = 1)[0])
        
        # We calculate the energy difference if we flip
        E_initial = ising_lattice.spin_energy(i,j)
        ising_lattice.flip_spin(i,j)
        E_final = ising_lattice.spin_energy(i,j)
        delta_E = E_final - E_initial
        
        # For convenience we flip it back to the original
        ising_lattice.flip_spin(i,j)
        
        # Then we should flip the spin
        
        if delta_E<0 or np.random.rand()<np.exp(-delta_E/T):
            # If the Metropolis Criteria holds, swap. 
            ising_lattice.flip_spin(i,j)

def monte_carlo_simulation(ising_lattice, T, num_scans, num_scans_4_equilibrium, frequency_sweeps_to_collect_magnetization, plot_result = False, print_info=False):
    start_time = time.time()
    
    # The first three arguments are self-explanatory. 
    # The last one is the number of scans we need to do
    # Before we reach equilibrium. Therefore we do not
    # need to collect data at these steps. 
    if print_info:
        ising_lattice.print_info()
    
    # We start by collecting <E> and <m> data. In order to 
    # calculate these, we record energy and magnetization 
    # after we reach equilibrium.
    
    # The total number of records, both first and last point included
    TOTAL_NUM_RECORDS = int(num_scans/frequency_sweeps_to_collect_magnetization)+1
    energy_records = np.zeros(TOTAL_NUM_RECORDS)
    magnetization_records = np.zeros(TOTAL_NUM_RECORDS)
    increment_records = 0
    
    # We will return this n-dimensional 
    lattice_configs = np.zeros((TOTAL_NUM_RECORDS,\
                               ising_lattice.lattice_size,\
                               ising_lattice.lattice_size))
    
    ei = ising_lattice.energy
    ef = ising_lattice.energy + 10.0
    for equ in np.arange(num_scans_4_equilibrium):
        if abs(ei - ef) > 1e-6:
        #     continue
            scan_lattice(ising_lattice,T)
            ef = ising_lattice.energy

    for k in np.arange(num_scans):
        scan_lattice(ising_lattice, T)
        if k%frequency_sweeps_to_collect_magnetization==0:
            energy_records[increment_records] = ising_lattice.energy()
            magnetization_records[increment_records] = ising_lattice.magnetization()
            lattice_configs[increment_records] = ising_lattice.lattice_state
            increment_records += 1
    
    # Now we can get the <E> and <m>
    
  
    
    
    print("For T = ", T, "Simulation is executed in: ", " %s seconds " % round(time.time() - start_time,2))
    
    if plot_result:
        ising_lattice.plot_lattice()
    
    
    return lattice_configs, energy_records, magnetization_records