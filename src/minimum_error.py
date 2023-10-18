import numpy as np
import pandas as pd


def read_file(file_path):
    data_array = np.loadtxt(file_path)
    targets, obs = data_array[:, 0].copy(), data_array[:, 1:].copy()
    return targets, obs


def split_data(obs, targets):
    train_obs, train_targets = obs[1::2], targets[1::2]
    test_obs, test_targets = obs[0::2], targets[0::2]
    return train_obs, test_obs, train_targets, test_targets


def estimate_a_priori(classes):
    class_one = np.sum(classes == 1)
    prob_one = class_one / classes.shape[0]
    prob_two = 1.0 - prob_one
    return prob_one, prob_two


def estimate_class_mean(train_obs, train_targets):
    class_one_mean = train_obs[train_targets == 1].mean(axis=0)
    class_two_mean = train_obs[train_targets == 2].mean(axis=0)
    return class_one_mean, class_two_mean


def estimate_class_cov(class_one_mean, class_two_mean, train_obs, train_targets):
    N_one = train_obs[train_targets == 1].shape[0]
    N_two = train_obs.shape[0] - N_one
    class_one_dev = train_obs[train_targets == 1] - class_one_mean
    class_two_dev = train_obs[train_targets == 2] - class_two_mean
    cov_one = (class_one_dev.T @ class_one_dev) / (N_one - 1)
    cov_two = (class_two_dev.T @ class_two_dev) / (N_two - 1)

    print(cov_one)
    return cov_one, cov_two


def build_discriminant(test_obs, class_means, class_covs, a_priori_probs):
    W_one = -(1 / 2) * np.linalg.inv(class_covs[0])
    W_two = -(1 / 2) * np.linalg.inv(class_covs[1])

    w_one = np.linalg.inv(class_covs[0]) * class_means[0]
    w_two = np.linalg.inv(class_covs[1]) * class_means[1]

    w_one_0 = (
        -(1 / 2) * class_means[0] @ np.linalg(class_covs[0]) @ class_means[0].T
        - (1 / 2) * np.log(np.abs(class_covs[0]))
        + np.log(a_priori_probs[0])
    )
    w_two_0 = (
        -(1 / 2) * class_means[1] @ np.linalg(class_covs[1]) @ class_means[1].T
        - (1 / 2) * np.log(np.abs(class_covs[1]))
        + np.log(a_priori_probs[1])
    )

    



if __name__ == "__main__":
    targets, obs = read_file("/home/gregz/Programs/Pattern-Recognition_1/data/ds-1.txt")

    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)

    class_one_mean, class_two_mean = estimate_class_mean(train_obs, train_targets)
    estimate_class_cov(class_one_mean, class_two_mean, train_obs, train_targets)
