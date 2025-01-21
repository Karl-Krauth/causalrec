import math
import random
import time

import numba
import numpy as np
import reclab.recommenders
import torch
from torch.linalg import norm


device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MatrixFactorization(reclab.recommenders.PredictRecommender):
    def __init__(
        self,
        latent_dim,
        num_epochs=200,
        batch_size=100,
        regularizer=0.04,
        learning_rate=0.01,
        init_stddev=0.1,
        print_loss=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._regularizer = regularizer
        self._learning_rate = learning_rate
        self._latent_dim = latent_dim
        self._init_stddev = init_stddev
        self._user_factors = None
        self._user_biases = None
        self._item_factors = None
        self._item_biases = None
        self._global_bias = None
        self._print_loss = print_loss

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        super().reset(users, items, ratings)

    @property
    def name(self):
        return "matrix_factorization"

    def update(self, users=None, items=None, ratings=None, retrain=True):  # noqa: D102
        super().update(users, items, ratings)
        if not retrain:
            return

        self._reset_parameters()
        if self._ratings.count_nonzero() == 0:
            return

        user_ids = torch.tensor(self._ratings.nonzero()[0], dtype=torch.long)
        item_ids = torch.tensor(self._ratings.nonzero()[1], dtype=torch.long)
        ratings = torch.tensor(
            self._ratings[self._ratings.nonzero()].toarray().flatten()
        )
        dataset = torch.utils.data.TensorDataset(user_ids, item_ids, ratings)
        optimizer = torch.optim.Adam(
            [
                self._user_factors,
                self._user_biases,
                self._item_factors,
                self._item_biases,
                self._global_bias,
            ],
            lr=self._learning_rate,
        )
        # momentum=0.9)
        for i in range(self._num_epochs):
            perm = torch.randperm(len(user_ids))
            user_ids = user_ids[perm].to(device)
            item_ids = item_ids[perm].to(device)
            ratings = ratings[perm].to(device)
            tot = 0.0
            start = time.time()
            for i in range(math.ceil(len(user_ids) / self._batch_size)):
                user_id = user_ids[i : i + self._batch_size]
                item_id = item_ids[i : i + self._batch_size]
                rating = ratings[i : i + self._batch_size]
                optimizer.zero_grad()
                user_biases = self._user_biases[user_id]
                user_factors = self._user_factors[user_id]
                item_biases = self._item_biases[item_id]
                item_factors = self._item_factors[item_id]
                loss = (
                    (
                        rating
                        - self._global_bias
                        - user_biases
                        - item_biases
                        - (item_factors * user_factors).sum(axis=1)
                    )
                    ** 2
                ).mean() + self._regularizer * (
                    norm(user_biases)
                    + norm(item_biases)
                    + norm(self._user_factors)
                    + norm(self._item_factors)
                )
                tot = loss + tot
                loss.backward()
                optimizer.step()
            if self._print_loss:
                print("Loss:", torch.sqrt(tot / (len(user_ids)) * self._batch_size))
                print("Time:", time.time() - start)

    @property
    def dense_predictions(self):
        if self._dense_predictions is None:
            self._dense_predictions = (
                self._global_bias
                + torch.unsqueeze(self._user_biases, 1)
                + torch.unsqueeze(self._item_biases, 0)
                + self._user_factors @ self._item_factors.T
            )
            self._dense_predictions = self._dense_predictions.detach().cpu().numpy()

        return self._dense_predictions

    def _predict(self, user_item):  # noqa: D102
        predictions = []
        for user_id, item_id, _ in user_item:
            predictions.append(
                (
                    (
                        self._global_bias
                        + self._user_biases[user_id]
                        + self._item_biases[item_id]
                    )
                    + (self._user_factors[user_id] * self._item_factors[item_id]).sum()
                )
                .detach()
                .cpu()
                .numpy()
            )
        return np.array(predictions)

    def _reset_parameters(self):
        self._user_factors = torch.normal(
            0,
            self._init_stddev,
            size=(len(self._users), self._latent_dim),
            device=device,
            requires_grad=True,
        )
        self._user_biases = torch.normal(
            0,
            self._init_stddev,
            size=(len(self._users),),
            device=device,
            requires_grad=True,
        )
        self._item_factors = torch.normal(
            0,
            self._init_stddev,
            size=(len(self._items), self._latent_dim),
            device=device,
            requires_grad=True,
        )
        self._item_biases = torch.normal(
            0,
            self._init_stddev,
            size=(len(self._items),),
            device=device,
            requires_grad=True,
        )
        self._global_bias = torch.normal(
            0, self._init_stddev, size=[], device=device, requires_grad=True
        )
