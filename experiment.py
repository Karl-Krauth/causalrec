import random

import numpy as np
import reclab
import reclab.recommenders

import als
import matrix_factorization
import utils


model_name_to_class = {
    "als": als.ALS,
    "ipw": lambda **kwargs: als.ALS(ipw=True, **kwargs),
    "libfm": reclab.recommenders.LibFM,
    "oracle": reclab.recommenders.PerfectRec,
    "vanilla": matrix_factorization.MatrixFactorization,
}


def run_experiments(
    recommenders, env_name, num_timesteps, step_size, num_test, overwrite=False
):
    test_set = sample_pairs(env_name, num_test, seed=1)
    utils.save_experiment(test_set, None, "test_{}".format(env_name))

    for neg_rating in [False, True]:
        get_true_ratings(env_name, neg_rating)
        for model_name, args in recommenders.items():
            run_feedback_experiment(
                model_name,
                args,
                env_name,
                test_set,
                num_timesteps,
                step_size,
                neg_rating,
                overwrite,
            )
            if model_name != "ipw":
                run_uniform_experiment(
                    model_name,
                    args,
                    env_name,
                    test_set,
                    num_timesteps,
                    step_size,
                    neg_rating,
                    overwrite,
                )


def get_true_ratings(env_name, neg_rating):
    experiment_name = "true_{}_{}".format(env_name, "neg" if neg_rating else "pos")
    env = reclab.make(env_name)
    env.seed(0)
    env.reset()
    ratings = 6 - env.dense_ratings if neg_rating else env.dense_ratings
    utils.save_experiment(None, ratings, experiment_name)


def run_feedback_experiment(
    model_name,
    model_args,
    env_name,
    test_set,
    num_timesteps=10,
    step_size=100,
    neg_rating=False,
    overwrite=False,
):
    experiment_name = "feedback_{}_{}_{}_{}".format(
        model_name, env_name, step_size, "neg" if neg_rating else "pos"
    )
    if not overwrite:
        recommended, predictions = utils.load_experiment(experiment_name)
        if predictions is not None:
            return recommended, predictions

    env = reclab.make(env_name)
    env.seed(0)
    users, items, ratings = env.reset()
    for uid, iid in test_set:
        env._rated_items[uid].add(iid)
    if neg_rating:
        ratings = {key: (6 - val[0], val[1]) for key, val in ratings.items()}
    model = model_name_to_class[model_name]
    if model_name == "oracle":
        if neg_rating:
            model_args["dense_rating_function"] = lambda: (6 - env.dense_ratings)
        else:
            model_args["dense_rating_function"] = lambda: env.dense_ratings

    rec = model(exclude=test_set, **model_args)
    rec.reset(users, items, ratings)

    recommended = []
    predictions = []
    if step_size % len(env.online_users) != 0:
        raise ValueError(
            "Step size is not divisible by the number of online users:",
            len(env.online_users),
        )

    retrain_timesteps = step_size // len(env.online_users)
    # TODO: Make this less confusingly named.
    num_timesteps *= retrain_timesteps
    tot = 0.0
    n = 0
    for i in range(num_timesteps):
        online_users = env.online_users
        recommendations, _ = rec.recommend(online_users, 1)
        users, items, ratings, info = env.step(recommendations)
        if neg_rating:
            ratings = {key: (6 - val[0], val[1]) for key, val in ratings.items()}

        for (user, item), (rating, context) in ratings.items():
            recommended.append((i, user, item, rating))
            tot += rating
            n += 1

        if i % retrain_timesteps == 0:
            predictions.append(rec.dense_predictions)
            print("Average rating at round {}: {}".format(i, tot / n))
            tot = 0.0
            n = 0
        rec.update(users, items, ratings, retrain=(i % retrain_timesteps) == 0)

    utils.save_experiment(recommended, predictions, experiment_name)

    return recommended, predictions


def run_uniform_experiment(
    model_name,
    model_args,
    env_name,
    test_set,
    num_timesteps,
    step_size,
    neg_rating=False,
    overwrite=False,
):
    experiment_name = "uniform_{}_{}_{}_{}".format(
        model_name, env_name, step_size, "neg" if neg_rating else "pos"
    )
    if not overwrite:
        _, predictions = utils.load_experiment(experiment_name)
        if predictions is not None:
            return predictions

    model = model_name_to_class[model_name]
    rec = model(**model_args)

    recommended = []
    predictions = []
    users, items, ratings, _ = get_dataset(
        env_name,
        train_size=1.0,
        num_ratings=num_timesteps * step_size,
        exclude=test_set,
    )
    if neg_rating:
        ratings = {key: (6 - val[0], val[1]) for key, val in ratings.items()}
    ratings = list(ratings.items())
    print(
        f"Num timesteps: {num_timesteps} Step size: {step_size} num ratings: {len(ratings)}"
    )
    for i in range(num_timesteps):
        r = dict(ratings[: i * step_size])
        rec.reset(users, items, r)
        predictions.append(rec.dense_predictions)

    utils.save_experiment(None, predictions, experiment_name)

    return predictions


def get_dataset(env_name, train_size=0.8, shuffle=True, num_ratings=None, exclude=None):
    env = reclab.make(env_name)
    if num_ratings is not None:
        env._num_init_ratings += num_ratings

    if exclude is not None:
        selected = sample_pairs(env_name, num_ratings + len(exclude), seed=0)
        selected = set(selected)
        selected = list(selected - set(exclude))
        selected = selected[:num_ratings]
        env._initial_sampling = selected

    env.seed(0)
    users, items, ratings = env.reset()
    ratings = list(ratings.items())
    if shuffle:
        random.shuffle(ratings)

    train_ratings = {}
    test_ratings = {}
    for i, ((user_id, item_id), (rating, context)) in enumerate(ratings):
        if i < len(ratings) * train_size:
            train_ratings[user_id, item_id] = (rating, context)
        else:
            test_ratings[user_id, item_id] = (rating, context)

    return users, items, train_ratings, test_ratings


def sample_pairs(env_name, num_pairs, seed=0):
    env = reclab.make(env_name)
    num_users, num_items = env._num_users, env._num_items
    rand = np.random.RandomState(seed=seed)
    idx_1d = rand.choice(num_users * num_items, num_pairs, replace=False)
    user_ids = idx_1d // num_items
    item_ids = idx_1d % num_items

    return list(zip(user_ids, item_ids))
