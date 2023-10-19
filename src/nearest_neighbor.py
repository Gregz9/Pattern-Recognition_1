import numpy as np
from utils import *


def measure_dist(obs_1, obs_2):
    distance = np.linalg.norm(obs_1 - obs_2)
    return distance


def nearest_neighbour(train_obs, train_targets, test_obs):
    c_test_obs = np.zeros((len(test_obs), 1))

    for i in range(len(test_obs)):
        near_neigh = np.argmin(
            [
                measure_dist(train_obs[i], train_obs[j])
                for j in range(len(train_obs))
                if i != j
            ]
        )
        c_test_obs[i] = train_targets[near_neigh]

    return c_test_obs.flatten()


if __name__ == "__main__":
    targets, obs = read_dataset(1)
    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)

    # measure_dist(train_obs[3], train_obs[1])
    print(nearest_neighbour(train_obs, train_targets, train_obs))
