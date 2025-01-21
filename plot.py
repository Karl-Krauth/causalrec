import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import metrics
import utils


font = {"family": "normal", "weight": "bold", "size": 22}

matplotlib.rc("font", size=25)
linewidth = 3
figsize = (15, 10)


def plot_ndcg(recommender_names, env_name, step_size, overwrite=False):
    stat_name = "ndcg_{}_{}".format(env_name, step_size)
    stat = utils.load_statistic(stat_name)
    if stat is None or overwrite:
        stat = compute_metric(recommender_names, env_name, step_size, metrics.ndcg)
    utils.save_statistic(stat_name, stat)

    fig = plt.figure(figsize=figsize)
    left = fig.add_subplot(211)
    right = fig.add_subplot(212)
    for key, value in stat.items():
        x = step_size * (np.arange(len(value)))
        if key[-3:] == "pos":
            left.plot(x, value, label=key, linewidth=linewidth)
        else:
            right.plot(x, value, label=key, linewidth=linewidth)

    plt.xlabel("Number of Datapoints")
    plt.ylabel("NDCG")
    # plt.yscale('log')
    left.legend()
    right.legend()


def plot_jaccard(
    recommender_names, env_name, step_size, k=10, num_samples=10000, overwrite=False
):
    stat_name = "jaccard_{}_{}_{}_{}".format(env_name, step_size, k, num_samples)
    stat = utils.load_statistic(stat_name)
    if stat is None or overwrite:
        stat = compute_metric(
            recommender_names,
            env_name,
            step_size,
            lambda pred, true, test_idxs: metrics.jaccard(
                pred, test_idxs, k=k, num_samples=num_samples
            ),
        )
    utils.save_statistic(stat_name, stat)

    fig = plt.figure(figsize=figsize)
    left = fig.add_subplot(211)
    right = fig.add_subplot(212)
    for key, value in stat.items():
        x = step_size * (np.arange(len(value)))
        if key[-3:] == "pos":
            left.plot(x, value, label=key, linewidth=linewidth)
        else:
            right.plot(x, value, label=key, linewidth=linewidth)

    plt.xlabel("Number of Datapoints")
    plt.ylabel("Jaccard")
    left.legend()
    right.legend()


def plot_cosine(recommender_names, env_name, step_size, overwrite=False):
    stat_name = "cosine_{}_{}".format(env_name, step_size)
    stat = utils.load_statistic(stat_name)
    if stat is None or overwrite:
        stat = compute_metric(
            recommender_names,
            env_name,
            step_size,
            lambda pred, true, test_idxs: metrics.cosine(pred),
        )
    utils.save_statistic(stat_name, stat)

    fig = plt.figure(figsize=figsize)
    left = fig.add_subplot(211)
    right = fig.add_subplot(212)
    for key, value in stat.items():
        x = step_size * (np.arange(len(value)))
        if key[-3:] == "pos":
            left.plot(x, value, label=key, linewidth=linewidth)
        else:
            right.plot(x, value, label=key, linewidth=linewidth)

    plt.xlabel("Number of Datapoints")
    plt.ylabel("Mean Cosine Similarity")
    left.legend()
    right.legend()


def plot_predict(recommender_names, env_name, step_size, overwrite=False):
    stat_name = "predict_{}_{}".format(env_name, step_size)
    stat = utils.load_statistic(stat_name)
    if stat is None or overwrite:
        stat = compute_metric(
            recommender_names,
            env_name,
            step_size,
            lambda pred, true, test_idxs: (pred, true),
        )
    utils.save_statistic(stat_name, stat)

    fig = plt.figure(figsize=figsize)
    left = fig.add_subplot(211)
    right = fig.add_subplot(212)
    for key, value in stat.items():
        pred, true = value[-1]
        if key[-3:] == "pos":
            for i in range(len(pred[0])):
                left.scatter(true[:, i], pred[:, i], label=key, linewidth=linewidth)
        else:
            for i in range(len(pred[0])):
                right.scatter(true[:, i], pred[:, i], label=key, linewidth=linewidth)

    plt.xlabel("True Rating")
    plt.ylabel("Predicted Rating")
    # left.legend()
    # right.legend()


def compute_metric(recommender_names, env_name, step_size, metric):
    test_idxs, _ = utils.load_experiment("test_{}".format(env_name))
    test_idxs = test_idxs
    stat = {}
    for flip in ["pos", "neg"]:
        _, true = utils.load_experiment("true_{}_{}".format(env_name, flip))
        if true is None:
            raise FileNotFoundError(
                "Cound not find true ratings for {} ratings.".format(flip)
            )
        for mode in ["uniform", "feedback"]:
            for recommender_name in recommender_names:
                if recommender_name == "ipw" and mode == "uniform":
                    continue
                experiment_name = "{}_{}_{}_{}_{}".format(
                    mode, recommender_name, env_name, step_size, flip
                )
                _, predictions = utils.load_experiment(experiment_name)
                if predictions is None:
                    raise FileNotFoundError(
                        "Cound not find predictions for {}.".format(experiment_name)
                    )

                stat[recommender_name + "_" + mode + "_" + flip] = np.array(
                    [metric(prediction, true, test_idxs) for prediction in predictions]
                )
    return stat
