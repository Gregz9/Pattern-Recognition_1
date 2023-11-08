from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

dim_combinations_list = [
    [(0,), (1,), (2,), (3,)],
    [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)],
    [(0, 1, 2), (0, 2, 3), (1, 2, 3), (0, 1, 3)],
    [(0, 1, 2, 3)],
]

dim_combinations_for_dataset_2_list = [
    [(0,), (1,), (2,)],
    [(0, 1), (0, 2), (1, 2)],
    [(0, 1, 2)],
]

for dataset_idx in (1, 2, 3):
    print(f"======== Dataset {dataset_idx} =========")
    targets, obs = read_dataset(dataset_idx)
    train_obs, test_obs, train_targets, test_targets = split_data(obs, targets)

    for dim_combinations in (
        dim_combinations_list
        if dataset_idx != 2
        else dim_combinations_for_dataset_2_list
    ):
        print(
            f"========= Now testing for dimension {len(dim_combinations[0])} ============"
        )
        best_fail_rate = 1
        best_dim = None
        for dimensions in dim_combinations:
            preds = nearest_neighbour(
                train_obs[:, dimensions], test_targets, test_obs[:, dimensions]
            )
            fail_rate = (
                np.sum(np.where(preds != test_targets, 1, 0)) / test_targets.shape[0]
            )

            print(f"{dimensions=} {fail_rate=}")

            # print("\n\nNow testing all methods on test set:")
            # # Nearest neighbour
            # preds_test_nn = nearest_neighbour(
            #     train_obs[:, dimensions], train_targets, test_obs[:, dimensions]
            # )
            # fail_rate_test_nn = (
            #     np.sum(np.where(preds_test_nn != test_targets, 1, 0))
            #     / test_targets.shape[0]
            # )
            #
            # # Linear discriminant
            # linear_discriminant = least_discriminant(
            #     least_params(train_obs[:, dimensions], train_targets)
            # )
            #
            # preds_test_lindisc = linear_discriminant(test_obs[:, dimensions])
            # fail_rate_test_lindisc = (
            #     np.sum(np.where(preds_test_lindisc != test_targets, 1, 0))
            #     / test_targets.shape[0]
            # )
            # # Minimum error
            # minimum_error_discriminant = minimum_error(
            #     train_obs[:, dimensions], train_targets
            # )
            #
            # preds_test_minerr = minimum_error_discriminant(test_obs[:, dimensions])
            # fail_rate_test_minerror = (
            #     np.sum(np.where(preds_test_minerr != test_targets, 1, 0))
            #     / test_targets.shape[0]
            # )
            # print(
            #     f"Fail rates: NN: {fail_rate_test_nn:.3f} LINDISC: {fail_rate_test_lindisc:.3f} MINERROR: {fail_rate_test_minerror:.3f}"
            # )
            if fail_rate < best_fail_rate:
                best_dim = dimensions
                best_fail_rate = fail_rate

        # if dataset_idx == 2 and len(best_dim) == 2:
        #     plt.scatter(
        #         test_obs[:, best_dim[0]][test_targets == 1],
        #         test_obs[:, best_dim[1]][test_targets == 1],
        #     )
        #     plt.scatter(
        #         test_obs[:, best_dim[0]][test_targets == 2],
        #         test_obs[:, best_dim[1]][test_targets == 2],
        #     )
        #     plt.show()

        # print(f"Lowest fail rate was {best_fail_rate:.3f}, for features: {best_dim}")

        # ------------ here starts the actual grog way ------------

        print("\n\nNow testing all methods on test set:")
        # Nearest neighbour
        preds_test_nn = nearest_neighbour(
            train_obs[:, best_dim], train_targets, test_obs[:, best_dim]
        )
        fail_rate_test_nn = (
            np.sum(np.where(preds_test_nn != test_targets, 1, 0))
            / test_targets.shape[0]
        )

        # Linear discriminant
        linear_discriminant = least_discriminant(
            least_params(train_obs[:, best_dim], train_targets)
        )

        preds_test_lindisc = linear_discriminant(test_obs[:, dimensions])
        fail_rate_test_lindisc = (
            np.sum(np.where(preds_test_lindisc != test_targets, 1, 0))
            / test_targets.shape[0]
        )
        # Minimum error
        minimum_error_discriminant = minimum_error(
            train_obs[:, best_dim], train_targets
        )

        preds_test_minerr = minimum_error_discriminant(test_obs[:, best_dim])
        fail_rate_test_minerror = (
            np.sum(np.where(preds_test_minerr != test_targets, 1, 0))
            / test_targets.shape[0]
        )
        print(
            f"Fail rates: NN: {fail_rate_test_nn:.3f} LINDISC: {fail_rate_test_lindisc:.3f} MINERROR: {fail_rate_test_minerror:.3f}"
        )
