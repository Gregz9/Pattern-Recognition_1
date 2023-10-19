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


def _class_discriminant(test_obs, class_mean, class_cov, a_priori_prob):
    W = -(1 / 2) * np.linalg.inv(class_cov)
    # W_two = -(1 / 2) * np.linalg.inv(class_covs[1])

    w = np.linalg.inv(class_cov) * class_mean
    # w_two = np.linalg.inv(class_covs[1]) * class_means[1]

    w_0 = (
        -(1 / 2) * class_mean @ np.linalg.inv(class_cov) @ class_mean.T
        - (1 / 2) * np.log(np.linalg.det(class_cov))
        + np.log(a_priori_prob)
    )

    # w_two_0 = (
    #     -(1 / 2) * class_means[1] @ np.linalg(class_covs[1]) @ class_means[1].T
    #     - (1 / 2) * np.log(np.linalg.det(class_covs[1]))
    #     + np.log(a_priori_probs[1])
    # )
    def discriminant_func(test_obs):
        g_x = test_obs @ W @ test_obs.T + w.T @ test_obs + w_0

        return g_x

    return discriminant_func


def gen_discriminant(c1_discr, c2_discr):
    return c1_discr - c2_discr


if __name__ == "__main__":
    targets, obs = read_dataset(1)

    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)

    class_one_mean, class_two_mean = estimate_class_mean(train_obs, train_targets)
    estimate_class_cov(class_one_mean, class_two_mean, train_obs, train_targets)
