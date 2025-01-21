import numba
import numpy as np
import reclab.recommenders


class ALS(reclab.recommenders.PredictRecommender):
    def __init__(
        self,
        latent_dim,
        num_epochs=200,
        regularizer=0.04,
        init_stddev=0.1,
        eps=0.1,
        ipw=False,
        exclude=None,
    ):
        super().__init__(type="eps_greedy", eps=eps, exclude=exclude)
        self._num_epochs = num_epochs
        self._regularizer = regularizer
        self._latent_dim = latent_dim
        self._init_stddev = init_stddev
        self._user_factors = None
        self._item_factors = None
        self._weights = None
        self._ipw = ipw

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self._weights = np.zeros((len(users), len(items)))
        super().reset(users, items, ratings)

    @property
    def name(self):
        return "matrix_factorization"

    def update(self, users=None, items=None, ratings=None, retrain=True):  # noqa: D102
        super().update(users, items, ratings)

        if self._dense_predictions is None and self._user_factors is not None:
            self._dense_predictions = (self._user_factors @ self._item_factors.T).clip(
                0, 1
            )

        num_ratings = self._ratings.count_nonzero()
        max_num_ratings = len(self._users) * len(self._items) - len(self._exclude)
        if self._ipw and len(ratings) == 1:
            for user_id, item_id in ratings:
                # This is N - m.
                num_remaining = (
                    len(self._items)
                    - self._ratings[user_id].count_nonzero()
                    + 1
                    - len(self._exclude_dict[user_id])
                )
                zeros = self._ratings[user_id].toarray().flatten() == 0
                assert zeros[self._exclude_dict[user_id]].all() == True
                zeros[self._exclude_dict[user_id]] = False
                max_pred = np.max(self._dense_predictions[user_id, zeros])
                num_max = (self._dense_predictions[user_id, zeros] == max_pred).sum()
                eps = self._strategy_dict["eps"]
                prob = eps / num_remaining
                if self._dense_predictions[user_id, item_id] > max_pred:
                    prob += 1 - eps
                elif self._dense_predictions[user_id, item_id] == max_pred:
                    prob += (1 - eps) / (num_max + 1)
                prob /= len(self._users)
                self._weights[user_id, item_id] = (
                    1
                    / (max_num_ratings - num_ratings)
                    * (1 / ((max_num_ratings - num_ratings + 1) * prob) - 1)
                )

        if not retrain:
            return

        self._reset_parameters()
        if num_ratings == 0:
            return

        if ratings is not None:
            r_u = []
            nnz_u = []
            ratings = self._ratings.tocsr()
            for i in range(self._user_factors.shape[0]):
                nnz = ratings[i].nonzero()[1]
                nnz_u.append(nnz)
                r_u.append(ratings[i, nnz].toarray().flatten())
            self._r_u = numba.typed.List(r_u)
            self._nnz_u = numba.typed.List(nnz_u)

            r_i = []
            nnz_i = []
            ratings = self._ratings.tocsc()
            for i in range(self._item_factors.shape[0]):
                nnz = ratings[:, i].nonzero()[0]
                nnz_i.append(nnz)
                r_i.append(ratings[nnz, i].toarray().flatten())
            self._r_i = numba.typed.List(r_i)
            self._nnz_i = numba.typed.List(nnz_i)

        weights = self._weights * (max_num_ratings - num_ratings) + 1  # / num_ratings
        for epoch in range(self._num_epochs):
            # Optimize items
            X = self._user_factors
            self._item_factors = update_values(
                X, weights.T, self._nnz_i, self._r_i, self._regularizer
            )

            # Optimize users
            X = self._item_factors
            self._user_factors = update_values(
                X, weights, self._nnz_u, self._r_u, self._regularizer
            )

        self._dense_predictions = (self._user_factors @ self._item_factors.T).clip(0, 1)

    def _predict(self, user_item):  # noqa: D102
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(
                (self._user_factors[user_id] * self._item_factors[item_id]).sum(axis=0)
            )
        return np.array(predictions).clip(0, 1)

    def _reset_parameters(self):
        self._user_factors = (
            np.random.randn(len(self._users), self._latent_dim) * self._init_stddev
        )
        self._item_factors = (
            np.random.randn(len(self._items), self._latent_dim) * self._init_stddev
        )


@numba.njit(parallel=True, fastmath=True)
def update_values(X, weights, nnz, ratings, reg, label=None):
    factors = np.random.randn(len(nnz), X.shape[1]) * 0.1

    for i in numba.prange(len(nnz)):
        if len(nnz[i]) == 0:
            continue
        X_i = X[nnz[i]]
        w = np.diag(weights[i][nnz[i]])
        prod = X_i.T @ w
        r = ratings[i]
        factors[i] = np.linalg.solve(prod @ X_i + reg * np.eye(X.shape[1]), prod @ r)

    return factors
