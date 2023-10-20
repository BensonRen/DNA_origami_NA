
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from numpy.linalg import norm
import math
import freud
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable  
save_folder =  '/home/pl201/2023_dataset/'  #save npy destination
mother_folder = '/home/pl201/2023_dataset/simulation_data' #data folder


if __name__ == '__main__':
        for file in os.listdir(mother_folder):
                file_id = file[:8] #the rep id, e.g. 1181.30
                file_pos = '{}/{}'.format(mother_folder,file)
                save_file_pos = '{}/{}'.format(save_folder,file_id) #for saving the image
                #load simulation data file and generate numpy file
                with warnings.catch_warnings(): 
                        warnings.simplefilter("ignore")
                        # We read the number of particles, the system box, and the
                        # particle positions into 3 separate arrays.
                        N = int(np.genfromtxt(file_pos, skip_header=2, max_rows=1)[0]) #array([7600.,   nan])[0] = 7600
                        box_data = np.genfromtxt(file_pos, skip_header=5, max_rows=2)
                        data_unprocessed = np.genfromtxt(file_pos, skip_header=23, invalid_raise=False)
                        data =data_unprocessed[:,2:6] #keep atom_type, x, y,z
                        # Remove the unwanted text rows, if exists
                        data = data[~np.isnan(data).all(axis=1)]
                        box_xy = box_data[:,1]-box_data[:, 0]
                typeid = data[:,0] #array contains type of atoms
                r = data[:,1:]
                box = box_xy[0]

                ###
                #Depends on how you want to store the data as what kinds of format, either way the above process can produce the numpy array specifying the simulation box size as variable 'box' (not important for now), 
                # Atom coordinates as variable "r"(what we need) and id list specifying which atom is a patch (attractive stuff) and which one is a core (DNA origami) 
                ###
                #np.save("Box",box)
                #np.save("Type",typeid)
                #np.save("Trajectory",r)

def Simulation_data_reader(file_pos):
        '''
        This is used to extract position, box, and atom types of a given simulation file 
        '''
        #load simulation data file and generate numpy file
        with warnings.catch_warnings(): 
                        warnings.simplefilter("ignore")
                        # We read the number of particles, the system box, and the
                        # particle positions into 3 separate arrays.
                        #N = int(np.genfromtxt(file_pos, skip_header=2, max_rows=1)[0]) #array([7600.,   nan])[0] = 7600
                        box_data = np.genfromtxt(file_pos, skip_header=5, max_rows=2)
                        data_unprocessed = np.genfromtxt(file_pos, skip_header=23, invalid_raise=False)
                        data =data_unprocessed[:,:6] #keep atom_type, x, y,z
                        # Remove the unwanted text rows, if exists
                        data = data[~np.isnan(data).all(axis=1)]
                        box_xy = box_data[:,1]-box_data[:, 0]
        box = box_xy[0]
        atomid =    data[:,0]
        molid =     data[:,1]
        typeid =    data[:,2] #array contains type of atoms
        r =         data[:,3:]
        return r, box, atomid, molid, typeid

def show_bond_orientational_order_metrics(system):
    voro = freud.locality.Voronoi()
    voro.compute(system)
    voro.plot()
    for k in [ 2, 3, 4, 5, 6, 7, 8]:

        psi = freud.order.Hexatic(k=k, weighted=False)
        psi.compute(system, neighbors=voro.nlist.filter_r(9))
        order = np.absolute(psi.particle_order)
        
        ax = voro.plot()
        patches = ax.collections[0]
        patches.set_array(order)
        patches.set_cmap("viridis")
        patches.set_clim(0, 1)
        patches.set_alpha(0.7)
        # Remove old colorbar coloring by number of sides
        ax.figure.delaxes(ax.figure.axes[-1])
        ax_divider = make_axes_locatable(ax)
        # Add a new colorbar to the right of the main axes.
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cbar = Colorbar(cax, patches)
        cbar.set_label(rf"$\psi'_{k}$", size=20)
        ax
def count_orientational_order_metrics(system):
    '''
    distribution of bond orientational order. for the sake of comparison
    '''
    voro = freud.locality.Voronoi()
    voro.compute(system)
    bond_order_list  = []
    for k in [2, 3, 4, 5, 6, 7, 8]:

        psi = freud.order.Hexatic(k=k,weighted=False)
        psi.compute(system, neighbors=voro.nlist.filter_r(9))
        order = np.absolute(psi.particle_order)
        #counts, bins = np.histogram(order)
        counts, bins = np.histogram(order,20,[0,1]) #10 bins or 100 bins the data distribution looks drastically different...
        bond_order_list.append(counts)
    return np.array(bond_order_list)

def show_bond_orientational_order_metrics(system):
    voro = freud.locality.Voronoi()
    voro.compute(system)
    voro.plot()
    for k in [ 2, 3, 4, 5, 6, 7, 8]:
   
        psi = freud.order.Hexatic(k=k, weighted=False)
        psi.compute(system, neighbors=voro.nlist.filter_r(9))
        order = np.absolute(psi.particle_order)
        
        ax = voro.plot()
        patches = ax.collections[0]
        patches.set_array(order)
        patches.set_cmap("viridis")
        patches.set_clim(0, 1)
        patches.set_alpha(0.7)
        # Remove old colorbar coloring by number of sides
        ax.figure.delaxes(ax.figure.axes[-1])
        ax_divider = make_axes_locatable(ax)
        # Add a new colorbar to the right of the main axes.
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cbar = Colorbar(cax, patches)
        cbar.set_label(rf"$\psi'_{k}$", size=20)
        ax

def RDF_calc_1frame(Position_table, Box_length, dr, n):
  '''
  input: location information of partcles at 1 frame, number of atoms, thickness of shell (dr), number density
  output: RDF value for a single frame
  Modified Allen & Tildesky's code
  ''' 
  # dr =   thickness of shell
  # n  =   number of atoms

  #Parameter setup
 
  r = Position_table.copy() 
  r /= Box_length #convert r to box unit
  dr /= Box_length # convert dr to box unit
  n_k = math.floor(0.5/dr) # Accumulate out to half box length
  r_max = n_k*dr           # Actual r_max (box=1 units)
  h     = np.zeros(n_k,dtype=np.int_) # initialize


      # Simple approach calculating all pairs at once
  rij        = r[:,np.newaxis,:] - r[np.newaxis,:,:]           # Set of all distance vectors
  rij        = rij - np.rint(rij)                              # Apply periodic boundaries
  rij_mag    = np.sqrt(np.sum(rij**2,axis=-1))                 # Separation distances
  rij_mag    = rij_mag[np.triu_indices_from(rij_mag,k=1)]      # Extract upper triangle
  hist,edges = np.histogram(rij_mag,bins=n_k,range=(0.0,r_max)) # Accumulate histogram of separations
  h          = h + 2*hist                                      # Accumulate histogram
  
  rho  = float(n) # Our calculation is done in box=1 units
  #h_id = ( 4/3 * np.pi * rho ) * ( edges[1:n_k+1]**3 - edges[0:n_k]**3 ) # Ideal number #3D
  h_id = (np.pi * rho ) * ( edges[1:n_k+1]**2 - edges[0:n_k]**2 )#2d #2d 2D so the particle number in the shell = pi*r^2 *global density, assuming particle distributes uniformly
  g    = h / h_id / n # Average number
  edges = edges*Box_length                       # Convert bin edges back to sigma=1 units
  r_mid = 0.5*(edges[0:n_k]+edges[1:n_k+1]) # Mid points of bins
  
  return r_mid, g  
def Pairwise_Distance_calculator(r_center):
    '''
    calculate from a position array, whic is the position of the center of the molecule, return a pairwise distance matrix ( a symmetric matrix) for finding out which intermolecular distance is smaller than a given cutoff 
    '''
    rij        = r_center[:,np.newaxis,:] - r_center[np.newaxis,:,:]           # Set of all distance vectors
    rij        = rij - np.rint(rij)                                            # Apply periodic boundaries
    rij = np.sqrt(np.sum(rij**2,axis=-1))
    return rij
      
def select_neighbors_cutoff_style(rij,molid_center,cutoff,count_yourself=False):
    '''
    Based on the central point (typeid==3), select the nearest neightbor, return their mol id. if count_yourself is false, that means that given a molecule, that molecule will not be included in the neighbor list
    '''
    neighbor_list = []
    if not count_yourself:
        for idx in range(rij.shape[0]):                       #it's a symmetric matrix
            neighbor_idx = np.where((rij[idx]<cutoff) & (rij[idx]>0.))[0] #this >0 selection works because atoms don't overlap
            neighbor = molid_center[neighbor_idx] # [0] is because the output is a tuple like this (array_we_want,), so use [0] to select array_we_want
            neighbor_list.append(neighbor) 
    else:
        print("Not implemented/suitable in this situation, but I still keep this, so fuck you")
        for idx in range(rij.shape[0]):                       #it's a symmetric matrix
            neighbor = molid_center[np.where((rij[idx]<cutoff))[0]] # [0] is because the output is a tuple like this (array_we_want,), so use [0] to select array_we_want
            neighbor_list.append(neighbor)

    return neighbor_list

def diagonal_vector_calculator(r,selected_mol,molid,atomid,normalization=True):
    '''
    compute a diagonal vector of a give cubic molecule (atoma with the same molid), where that vector is from the position of a atom with the smallest id to the largest id within the molecule
    '''
    rigid_particle_idx = np.where(molid==selected_mol)
    min_id = int(min(atomid[rigid_particle_idx])) 
    max_id = int(max(atomid[rigid_particle_idx]))
    min_idx = np.where(atomid==min_id)[0][0] 
    max_idx = np.where(atomid==max_id)[0][0]
    diagonal_vector = r[max_idx,:]  - r[min_idx,:]
    
    if normalization==True:
        diagonal_vector /= np.linalg.norm(diagonal_vector)
    return diagonal_vector

def angle_between_vectors(a, b):
    '''
    input vectors, return angles between vectors
    https://stackoverflow.com/questions/64501805/dot-product-and-angle-in-degrees-between-two-numpy-arrays
    '''
    dot_product = np.dot(a, b)
    prod_of_norms = np.linalg.norm(a) * np.linalg.norm(b)
    angle = round(np.degrees(np.arccos(dot_product / prod_of_norms)), 1)
    if angle>90:
         angle-=90
    elif angle>180:
         angle-=180
    elif angle>270:
         angle-=270

        
    return  angle

def Angle_calc_1frame(Position_table, Box_length, atomid, typeid, molid, cutoff, hist_width = 6, neighbor_style="Is_cutoff"):
    '''
    extract angular distribution
    '''
    r = Position_table 
    r /= Box_length #convert r to box unit
    cutoff /= Box_length #convert cutoff to box unit
    angle_tensor = []
    if neighbor_style=="Is_cutoff":
        center_particle_idx = np.where(typeid==3)
        r_center = r[center_particle_idx]
        molid_center = molid[center_particle_idx]
        rij = Pairwise_Distance_calculator(r_center) #distance matrix
        neighbor_list = select_neighbors_cutoff_style(rij,molid_center,cutoff)
        for idx, mol in enumerate(molid_center):
            neighbor = neighbor_list[idx]

            diagonal_vector = diagonal_vector_calculator(r,mol,molid,atomid)
            
            angle_list = []
            for neighbor_mol in neighbor:
                #print(neighbor_mol)
                neighbor_diagonal_vector = diagonal_vector_calculator(r,neighbor_mol,molid,atomid)
                angle  = angle_between_vectors(diagonal_vector,neighbor_diagonal_vector)
                angle_list.append(angle)
                #print(neighbor_diagonal_vector)
                #print(diagonal_vector)
                #break
            angle_tensor.append(np.array(angle_list))
            #break
    else:
        print("Implementation Error")
    
    angle_distribution = np.concatenate(angle_tensor)#convert it to a flatten array
    #hist_width
    return angle_distribution

def NN_dist(x, tol=0.01):
        '''
        extract nearest neighbor data
        '''
        angle_full_list = []
        index_list = np.arange(len(x)) # Initialize a index list
        # Calculate the pairwise distance
        dist_mat = pairwise_distances(x)
        # Remove the 0 (which is itself) 
        dist_mat[dist_mat == 0] = np.inf
        # Get the nearest neighbor from the list
        nn_dist = np.min(dist_mat, axis=0)
        # Calcualte the number of points that have the nearest neighbor distance
        num_nn = np.zeros_like(nn_dist)
        # loop over each one
        for i in range(len(nn_dist)):
                nn_indicator = np.square(dist_mat[i, :] - nn_dist[i]) < tol
                num_nn[i] = int(np.sum(nn_indicator))
                if num_nn[i] <= 1:
                        continue
                # Calculate the angle of the nearest neighbor
                nn_index_list = index_list[nn_indicator]
                first_line = x[i, :] - x[nn_index_list[0], :]
                angle_list = -10 * np.ones(int(num_nn[i] - 1))
                #         print('len of angle list', len(angle_list))
                for j in range(1, int(num_nn[i])):
                        second_line = x[i, :] - x[nn_index_list[j], :]
                        before_acrcos = np.clip(np.dot(first_line, second_line)/norm(first_line)/norm(second_line), -1 , 1)
                #             print('before_arccos', before_acrcos)
                        angle = np.arccos(before_acrcos)
                #             print(angle)
                #             print(np.shape(angle))
                        angle_list[j-1] = angle / np.pi * 180
                #         print(angle_list)
                angle_full_list.extend(list(angle_list))
                #     print(angle_list)
        return nn_dist, num_nn, angle_full_list


# plot_all(r[typeid == 3, :2])
# plot_all(r[typeid == 2, :2])
def extract_patch_nn_features(r, typeid):
        nn_dist, num_nn, angle_full_list = NN_dist(r[typeid == 2, :2])
        ## Convert the histogram into a real list of features (labels) to do inverse relationship
        num_hist, num_hist_bound = np.histogram(num_nn, density=True, bins=6, range=(0.5, 6.5))
        angle_hist, angle_hist_bound = np.histogram(angle_full_list, density=True, bins=150, range=(29.5, 180.5))
        dist_hist, dist_hist_bound = np.histogram(nn_dist, density=True, bins=50, range=(1, 2))
        return num_hist, angle_hist, dist_hist

def plot_all(x):
        nn_dist, num_nn, angle_full_list = NN_dist(x)
        f = plt.figure()
        plt.hist(num_nn, bins=50)
        plt.xlabel('num_nn')
        f = plt.figure()
        plt.hist(angle_full_list, bins=50)
        plt.xticks([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])
        plt.xlabel('nn_angle')
        f = plt.figure()
        plt.hist(nn_dist, bins=50)
        plt.xlabel('nn_dist')
