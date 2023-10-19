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


def least_params(train_obs):
    bias = np.ones((len(train_obs), 1))
    ext_train_obs = np.concatenate((bias, train_obs), axis=1)

    b = np.where(train_targets == 1, 1, -1)

    params = np.linalg.inv(ext_train_obs.T @ ext_train_obs) @ ext_train_obs.T @ b
    return params


def least_discriminant(params):
    def discriminant(test_obs):
        return params.T @ test_obs

    return discriminant


if __name__ == "__main__":
    targets, obs = read_dataset(1)
    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)

    least_params(train_obs)
