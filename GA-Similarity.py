# CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe and Alain Tchagang 
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca 
# (June-2023)


# Genetic Algorithm and Similarity Analytical Model.

# ! pip install pygad 
# Source: Gad, A. F. (2021). PyGAD: An Intuitive Genetic Algorithm Python Library. arXiv:2106.06158v1 [cs.NE]. doi:10.48550/arXiv.2106.06158 

import pygad
import numpy as np
import pandas as pd

latent_dim = 256
np.random.shuffle(sampled_vec)

AC = np.asarray(pd.read_csv('atom_coord.csv').astype('float32'))
scaler_AC = MinMaxScaler()
scaler_AC.fit(AC)
AC=scaler_AC.transform(AC)

def fitness_func(solution, solution_idx):
    
    decoded_imgs = decoder.predict((solution).reshape(1, latent_dim))

    # Energy above convex hull
    Ehull_pred = Ehull_model.predict(decoded_imgs)
    output = np.where(Ehull_pred<0, 0, Ehull_pred)
    fitness_ = 1.0 / (np.abs(float(output[0]) - desired_output) + 0.000001)

    # ICSD
    icsd_pred = icsd_model.predict(decoded_imgs)
    icsd_pred = np.rint(icsd_pred).reshape(icsd_pred.shape[0], 1)

    # Similarity analytical model
    data = []
    pred_AC = decoded_imgs[:,112:152,0:3] #Predicted atomic coordinates for novel solution
    pred_AC = scaler_AC.inverse_transform(pred_AC.reshape(decoded_imgs.shape[0],120))
    dissimilarity = (np.mean(np.abs(pred_AC[i,:] - standard[0,:]))) #'standard' is the geometrical coordination of the reference perovskite in context
           
    if icsd_pred ==1 and dissimilarity < 0.2:

      fitness = fitness_

    else:
      fitness = 0
      
    return fitness


desired_output = 0 # ideal energy above convex hull value in eV/atom.

gene_space = []
for i in range(latent_dim):
  genes = {'low': (sampled_vec[:,i].min()), 'high': (sampled_vec[:,i].max())}
  gene_space.append(genes)

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=50,
                       fitness_func=fitness_func,
                       initial_population=sampled_vec[:50,],
                       mutation_type="adaptive",
                       mutation_percent_genes=[5, 2.5]
                       gene_space = gene_space,
                       save_best_solutions=True,
                       )

ga_instance.run()
sols = ga_instance.best_solutions
decode_vec = decoder.predict((np.array(sols[:])))
