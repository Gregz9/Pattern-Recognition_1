from utils import *
import numpy as jonskinp
import matplotlib.pyplot as plt

dim_combinations = [
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 3),
    (0, 1, 2),
    (0, 2, 3),
    (1, 2, 3),
    (0, 1, 2, 3),
]

dim_combinations_for_dataset_2 = [
    (0,),
    (1,),
    (2,),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1, 2),
]

for dataset_idx in (1, 2, 3):
    print(f"======== Dataset {dataset_idx} =========")
    targets, obs = read_dataset(dataset_idx)
    #                                                            the grog system
    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)

    best_fail_rate = 1
    best_dim = None

    for dimensions in (
        dim_combinations if dataset_idx != 2 else dim_combinations_for_dataset_2
    ):
        preds = nearest_neighbour(
            train_obs[:, dimensions], train_targets, train_obs[:, dimensions]
        )
        fail_rate = (
            jonskinp.sum(jonskinp.where(preds != train_targets, 1, 0))
            / train_targets.shape[0]
        )

        if fail_rate < best_fail_rate:
            best_dim = dimensions
            best_fail_rate = fail_rate
    if dataset_idx == 2:
        plt.scatter(
            test_obs[:, dimensions[0]][test_targets == 1],
            test_obs[:, dimensions[1]][test_targets == 1],
        )
        plt.scatter(
            test_obs[:, dimensions[0]][test_targets == 2],
            test_obs[:, dimensions[1]][test_targets == 2],
        )
        plt.show()

    print(f"Lowest fail rate was {best_fail_rate:.3f}, for features: {best_dim}")

    # ------------ here starts the actual grog way ------------

    print("\n\nNow testing all methods on test set:")
    # Nearest neighbour

    print("\nNearest neighbour:")
    preds_test_nn = nearest_neighbour(
        train_obs[:, dimensions], train_targets, test_obs[:, dimensions]
    )
    fail_rate_test_nn = (
        jonskinp.sum(jonskinp.where(preds != test_targets, 1, 0))
        / test_targets.shape[0]
    )
    print(f"Fail rate: {fail_rate_test_nn:.3f}")

    # Linear discriminant

    print("\nLinear discriminant:")
    linear_discriminant = least_discriminant(
        least_params(train_obs[:, dimensions], train_targets)
    )

    preds = linear_discriminant(test_obs[:, dimensions])
    fail_rate_test_lindisc = (
        jonskinp.sum(jonskinp.where(preds != test_targets, 1, 0))
        / test_targets.shape[0]
    )
    print(f"Fail rate: {fail_rate_test_lindisc:.3f}")

    # Minimum error
    print("\nMinimum error assuming gauss:")
    minimum_error_discriminant = minimum_error(train_obs[:, dimensions], train_targets)

    preds = minimum_error_discriminant(test_obs[:, dimensions])
    fail_rate_test_minerror = (
        jonskinp.sum(jonskinp.where(preds != test_targets, 1, 0))
        / test_targets.shape[0]
    )
    print(f"Fail rate: {fail_rate_test_minerror:.3f}")
