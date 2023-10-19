import numpy as np
from utils import *


def estimate_a_priori(train_targets):
    class_one = np.sum(train_targets == 1)
    prob_one = class_one / train_targets.shape[0]
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

    return cov_one, cov_two


def _class_discriminant(class_mean, class_cov, a_priori_prob):
    W = -(1 / 2) * np.linalg.inv(class_cov)

    w = np.linalg.inv(class_cov) @ class_mean

    w_0 = (
        -(1 / 2) * class_mean @ np.linalg.inv(class_cov) @ class_mean
        - (1 / 2) * np.log(np.linalg.det(class_cov))
        + np.log(a_priori_prob)
    )

    return lambda test_obs: np.sum(test_obs @ W * test_obs, axis=1) + test_obs @ w + w_0


def gen_discriminant(c1_discr, c2_discr):
    return lambda test_obs: c1_discr(test_obs) - c2_discr(test_obs)


if __name__ == "__main__":
    targets, obs = read_dataset(1)

    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)

    class_one_mean, class_two_mean = estimate_class_mean(train_obs, train_targets)
    cov_one, cov_two = estimate_class_cov(
        class_one_mean, class_two_mean, train_obs, train_targets
    )

    a_priori_one, a_priori_two = estimate_a_priori(train_targets)

    discriminant_one = _class_discriminant(class_one_mean, cov_one, a_priori_one)
    discriminant_two = _class_discriminant(class_two_mean, cov_two, a_priori_two)

    func = gen_discriminant(discriminant_one, discriminant_two)
    print(np.where(func(test_obs) > 0, 1, 2))
