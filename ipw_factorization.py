import time

import numpy as np
import reclab.recommenders
import scipy.sparse
import torch


class IPWFactorization(reclab.recommenders.PredictRecommender):
    def __init__(
        self,
        latent_dim,
        num_epochs=200,
        batch_size=100,
        regularizer=0.07,
        learning_rate=0.01,
        init_stddev=0.1,
        explore_ratio=100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._regularizer = regularizer
        self._learning_rate = learning_rate
        self._latent_dim = latent_dim
        self._init_stddev = init_stddev
        self._explore_ratio = explore_ratio
        self._user_factors = None
        self._user_biases = None
        self._item_factors = None
        self._item_biases = None
        self._rec_probabilities = None
        self._global_bias = 0.0

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self._rec_probabilities = None
        self._reset_parameters()
        super().reset(users, items, ratings)

    @property
    def name(self):
        return "ipw_factorization"

    def update(self, users=None, items=None, ratings=None, retrain=True):  # noqa: D102
        super().update(users, items, ratings)
        if not retrain:
            return
        if self._rec_probabilities is None:
            self._rec_probabilities = self._ratings.copy()
            self._rec_probabilities[self._rec_probabilities.nonzero()] = 1.0
            self._rec_probabilities *= (
                self._rec_probabilities.nnz / len(self._users) / len(self._items)
            )
        self._reset_parameters()
        dataset = rating_dataset.RatingDataset(self._ratings, self._rec_probabilities)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=10,
            prefetch_factor=self._batch_size,
        )
        optimizer = torch.optim.SGD(
            [
                self._user_factors,
                self._user_biases,
                self._item_factors,
                self._item_biases,
                self._global_bias,
            ],
            lr=self._learning_rate,
            momentum=0.9,
        )
        for i in range(self._num_epochs):
            tot = 0.0
            start = time.time()
            for batch in dataloader:
                optimizer.zero_grad()
                user_biases = self._user_biases[batch["user_id"]]
                user_factors = self._user_factors[batch["user_id"]]
                item_biases = self._item_biases[batch["item_id"]]
                item_factors = self._item_factors[batch["item_id"]]
                prob = np.clip(batch["probability"], 1e-1, 1 - 1e-1)
                N = len(self._users) * len(self._items)

                weights = 1
                loss = (
                    weights
                    * (
                        batch["rating"]
                        - self._global_bias
                        - user_biases
                        - item_biases
                        - (user_factors * item_factors).sum(axis=1)
                    )
                    ** 2
                    + self._regularizer
                    * (
                        user_biases ** 2
                        + item_biases ** 2
                        + torch.norm(user_factors, dim=1) ** 2
                        + torch.norm(item_factors, dim=1) ** 2
                    )
                ).mean()
                tot += loss
                loss.backward()
                optimizer.step()
            print("Loss:", torch.sqrt(tot / len(dataset) * self._batch_size))
            print("Time:", time.time() - start)

    def _predict(self, user_item):  # noqa: D102
        scores = scipy.sparse.dok_matrix(self._ratings.shape)
        user_ids = set()
        for user_id, item_id, _ in user_item:
            scores[user_id, item_id] = (
                (
                    self._global_bias
                    + self._user_biases[user_id]
                    + self._item_biases[item_id]
                    + self._user_factors[user_id] @ self._item_factors[item_id]
                )
                .detach()
                .numpy()
            )
            user_ids.add(user_id)

        recs = {}
        probs = {}
        for user_id in user_ids:
            user_scores = scores[user_id].toarray().flatten()
            base = np.exp(
                np.log(self._explore_ratio) / (user_scores.max() - user_scores.min())
            )
            user_scores = base ** user_scores
            prob = user_scores / user_scores.sum()
            item_id = np.random.choice(len(prob), p=prob)
            recs[user_id] = item_id
            probs[user_id, item_id] = prob[item_id]
        """
        for user_id in user_ids:
            user_means = scores[user_id].toarray().flatten()
            user_scores = user_means + np.random.randn(len(user_means)) * self._exploration_stddev
            item_id = np.argmax(user_scores)
            recs[user_id] = item_id
            # TODO: This is just a placeholder.
            probs[user_id, item_id] = 1.0
        """

        predictions = []
        for user_id, item_id, _ in user_item:
            if recs[user_id] == item_id:
                predictions.append(1)
                self._rec_probabilities[user_id, item_id] = probs[user_id, item_id]
            else:
                predictions.append(0)

        return np.array(predictions)

    def _reset_parameters(self):
        self._user_factors = torch.tensor(
            self._init_stddev * torch.randn(len(self._users), self._latent_dim),
            requires_grad=True,
        )
        self._user_biases = torch.tensor(
            self._init_stddev * torch.randn(len(self._users)), requires_grad=True
        )
        self._item_factors = torch.tensor(
            self._init_stddev * torch.randn(len(self._items), self._latent_dim),
            requires_grad=True,
        )
        self._item_biases = torch.tensor(
            self._init_stddev * torch.randn(len(self._items)), requires_grad=True
        )
        self._global_bias = torch.tensor(
            self._init_stddev * torch.randn([]), requires_grad=True
        )
