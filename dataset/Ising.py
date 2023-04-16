import numpy as np
import matplotlib.pyplot as plt 

class IsingLattice:
    
    # Initializer. Parameter n corresponds to the lattice size. 
    
    def __init__(self,lattice_size,J,h):
        
        # In order to easily access the parameters: 
        self.lattice_size = lattice_size
        self.num_sites = lattice_size*lattice_size
        self.J = J
        self.h = h
        
        # We randomly initialize the lattice with 0's and 1's
        lattice_state = np.random.choice([1,-1],size=(self.lattice_size,self.lattice_size))
        
        # We store the configuration 
        self.lattice_state = lattice_state
    
    # The Methods 
    # Plot function. This will help us easily see the lattice configuration.
    
    def plot_lattice(self, print_info=False): # print_info is Boolean. If it is true then we print info.
        
        plt.figure()
        plt.imshow(self.lattice_state)
        plt.show()
        if print_info:
            self.print_info()
    
    # Now we define print_info() method. It will print all the information about the lattice.
    
    def print_info(self):
        
        print("Lattice size: ", self.lattice_size , "x", self.lattice_size, ". J: ", self.J, " h: ", self.h )
    
    # A spin flipper at site (i,j) method
    
    def flip_spin(self,i,j):
        self.lattice_state[i,j] *= -1
        
    # Calculating energy of one spin at site (i,j)
        
    def spin_energy(self,i,j):
        
        # Spin at (i,j)
        spin_ij = self.lattice_state[i,j]
        
        # Now we need to deal with the boundary spins. 
        # We apply periodic boundary conditions.  
        sum_neighbouring_spins = self.lattice_state[(i+1)%self.lattice_size, j] + \
                                 self.lattice_state[i, (j+1)%self.lattice_size] + \
                                 self.lattice_state[(i-1)%self.lattice_size, j] + \
                                 self.lattice_state[i, (j-1)%self.lattice_size]
        
        # We calculate the energy terms for site 
        interaction_term = (- self.J * spin_ij * sum_neighbouring_spins)
        
        # This part is added so that in case 
        # there is no external magnetic field, i.e. h = 0
        # then we do not need the computer to do the computation
        # for the magnetic term. 
        if self.h == 0:
            return interaction_term
        else:
            magnetic_field_term = - (self.h * spin_ij)
            return magnetic_field_term + interaction_term
    
    # Calculating Total Lattice Energy
    
    def energy(self):
        
        # Initialize energy as 0.
        E = 0.0
        
        # We iterate through the lattice
        for i in np.arange(self.lattice_size):
            for j in np.arange(self.lattice_size):
                E = E + self.spin_energy(i,j)
                
        # But we counted neighbours twice here. So we need to correctly return. 
        # We divide by two 
        E = E / (2.0) / self.num_sites
        if self.h==0:
            return E
        else: 
            # We add the magnetic field term |IS THERE A 1/2 FACTOR HERE?|
            E = (E - self.h * np.sum(self.lattice_state)) / self.num_sites
            return E
    
    # Net magnetization
    
    def magnetization(self):
        return  np.abs(np.sum(self.lattice_state))/ (self.num_sites)