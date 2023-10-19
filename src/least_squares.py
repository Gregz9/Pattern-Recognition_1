import numpy as np
from utils import *


def least_params(train_obs, train_targets):
    bias = np.ones((len(train_obs), 1))
    ext_train_obs = np.concatenate((bias, train_obs), axis=1)

    b = np.where(train_targets == 1, 1, -1)

    params = np.linalg.inv(ext_train_obs.T @ ext_train_obs) @ ext_train_obs.T @ b
    return params


def least_discriminant(params):
    def discriminant(test_obs):
        bias = np.ones((len(train_obs), 1))
        ext_test_obs = np.concatenate((bias, test_obs), axis=1)
        return ext_test_obs @ params

    return discriminant


if __name__ == "__main__":
    targets, obs = read_dataset(1)
    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)
    print(train_obs)

    func = least_discriminant(least_params(train_obs, train_targets))

    print(np.where(func(test_obs) > 0, 1, 2))
