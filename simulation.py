import collections

import click
import numpy as np
import reclab
import reclab.recommenders

import ipw_factorization
import matrix_factorization


@click.command()
@click.option(
    "--latent-dim", default=20, help="The dimension of the user/item vectors."
)
@click.option(
    "--learning-rate", default=0.01, help="The learning rate to train the model with."
)
@click.option(
    "--regularizer", default=0.07, help="The regularization value to use for the model."
)
@click.option(
    "--num-epochs", default=200, help="The number of epochs to train the model for."
)
@click.option(
    "--batch-size", default=100, help="The batch size to train the model with."
)
@click.option(
    "--init-stddev",
    default=0.1,
    help="The standard deviation of the initial latent vectors.",
)
@click.option(
    "--model-type",
    default="ipw",
    type=click.Choice(["ipw", "vanilla"]),
    help="Which factorization model to use.",
)
@click.option(
    "--env-name", default="beta-rank-v1", help="The name of the environment to test on."
)
def main(
    latent_dim,
    learning_rate,
    regularizer,
    num_epochs,
    batch_size,
    init_stddev,
    model_type,
    env_name,
):
    env = reclab.make(env_name)
    env.seed(0)
    baseline_env = reclab.make(env_name)
    baseline_env.seed(0)
    users, items, ratings = env.reset()
    baseline_users, baseline_items, baseline_ratings = baseline_env.reset()
    if model_type == "ipw":
        model = ipw_factorization.IPWFactorization
    elif model_type == "vanilla":
        model = matrix_factorization.MatrixFactorization

    rec = model(
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        regularizer=regularizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        init_stddev=init_stddev,
    )
    baseline_rec = reclab.recommenders.PerfectRec(lambda: baseline_env.dense_ratings)

    rec.reset(users, items, ratings)
    baseline_rec.reset(baseline_users, baseline_items, baseline_ratings)

    recommended_items = collections.defaultdict(set)
    baseline_recommended_items = collections.defaultdict(set)
    for i in range(1000):
        online_users = env.online_users
        recommendations, _ = rec.recommend(online_users, 1)
        baseline_recommendations, _ = baseline_rec.recommend(
            baseline_env.online_users, 1
        )
        users, items, ratings, info = env.step(recommendations)
        baseline_users, baseline_items, baseline_ratings, info = baseline_env.step(
            baseline_recommendations
        )
        tot = 0.0
        n = 0
        for (user, item), (rating, context) in ratings.items():
            recommended_items[user].add(item)
            tot += rating
            n += 1
        baseline_tot = 0.0
        baseline_n = 0
        for (user, item), (rating, context) in baseline_ratings.items():
            baseline_recommended_items[user].add(item)
            baseline_tot += rating
            baseline_n += 1

        print("Average rating at round {}: {}".format(i, tot / n))
        print(
            "Baseline average rating at round {}: {}".format(
                i, baseline_tot / baseline_n
            )
        )
        print(
            "Jaccard {}".format(
                jaccard(get_top_k(rec)) - jaccard(get_top_k(baseline_rec))
            )
        )
        rec.update(users, items, ratings, retrain=i % 10 == 0)
        baseline_rec.update(baseline_users, baseline_items, baseline_ratings)


def get_top_k(model, k=10):
    best_items = collections.defaultdict(set)
    dense_preds = model.dense_predictions
    for user_id in range(dense_preds.shape[0]):
        items = np.argsort(dense_preds[user_id])[:-k]
        for item_id in items:
            best_items[user_id].add(item_id)
    return best_items


def compute_rec_prob(item_id, predictions, recommended, std_dev):
    def func(x):
        prod = 1.0
        for i, prediction in enumerate(predictions):
            if i == item_id or i in recommended:
                continue
            prod *= scipy.stats.norm.cdf(x, loc=prediction, scale=std_dev)
        return prod * scipy.stats.norm.pdf(x, predictions[item_id], std_dev)

    return scipy.integrate.quad(func, -np.inf, np.inf)


def jaccard(recommended_items):
    values = []
    for user1 in recommended_items:
        for user2 in recommended_items:
            if user1 != user2:
                intersection = recommended_items[user1].intersection(
                    recommended_items[user2]
                )
                union = recommended_items[user1].union(recommended_items[user2])
                values.append(len(intersection) / len(union))
    return np.mean(values)


if __name__ == "__main__":
    main()
