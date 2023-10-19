import numpy as np
import os


def read_dataset(idx):
    project_dir = os.path.dirname(os.path.dirname((__file__)))
    file_path = project_dir + f"/data/ds-{idx}.txt"
    data_array = np.loadtxt(file_path)
    targets, obs = data_array[:, 0].copy(), data_array[:, 1:].copy()
    return targets, obs


def split_data(obs, targets):
    train_obs, train_targets = obs[1::2], targets[1::2]
    test_obs, test_targets = obs[0::2], targets[0::2]
    return train_obs, test_obs, train_targets, test_targets


def measure_dist(obs_1, obs_2):
    distance = np.linalg.norm(obs_1 - obs_2)
    return distance

def nearest_neighbour(train_obs, train_targets):

    c_train_obs = np.zeros((len(train_obs), 1))

    for i in range(len(train_obs)):
        
        near_neigh = np.argmin([measure_dist(train_obs[i], train_obs[j]) for j in range(len(train_obs)) if i != j])
        c_train_obs[i] = train_targets[near_neigh]

    return c_train_obs


    
     
    
    

if __name__ == "__main__": 
    targets, obs = read_dataset(1)
    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)   

    # measure_dist(train_obs[3], train_obs[1])
    nearest_neighbour(train_obs, train_targets)



