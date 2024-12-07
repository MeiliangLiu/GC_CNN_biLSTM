# GC_CNN_biLSTM

This repository contains the python code for our paper which submitted to JBHI

## Dependencies:
conda environment is provided in env.yml

## Example usage:
1. Lorenz-96: run Lorenz_96.py  
    To simulation different nonlinear strength and dimension, change the forcing term F and dimension P in line 164: avg_score = train_loz(F=10, P=10, num_epochs=8000, learning_rate=params['learning_rate'], lam=params['lam'])  

2. Dream-3: run Dream_3.py  
    To test the performance in difference gene subdataset, change the type in line 179: 'type' :[1,2,3,4,5].  
    Type 1: Elico-1  
    Type-2: Elico-2  
    Type-3: Yeast-1  
    Type-4: Yeast-2  
    Type-5: Yeast-3  

3. Whisker stimulated Rat: run wistar_rat.py


## Output
The adjacency matrix of the estimated Granger causal is exported as a txt file once the run is completed.

## Acknowledgements
The Dream-3 datasets for the gene interaction network inference experiments are taken from "http://dreamchallenges.org/project-list/dream3-2008/".  
