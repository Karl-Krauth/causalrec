import collections

import numpy as np


def ndcg(dense_preds, true_preds, test_set):
    return np.mean(ndcg_per_user(dense_preds, true_preds, test_set))


def ndcg_per_user(dense_preds, true_preds, test_set):
    test_items = collections.defaultdict(list)
    for user_id, item_id in test_set:
        test_items[user_id].append(item_id)

    def dcg(preds, true):
        ids = np.argsort(-preds)
        return (true[ids] / np.log2(np.arange(preds.shape[0]) + 2)).sum()

    ndcgs = []
    for user_id in range(dense_preds.shape[0]):
        item_ids = test_items[user_id]
        preds = dense_preds[user_id][item_ids]
        true = true_preds[user_id][item_ids]
        ndcgs.append(dcg(preds, true) / dcg(true, true))

    return ndcgs


def jaccard(dense_preds, test_set, k=10, num_samples=1000):
    return _jaccard(get_top_k(dense_preds, test_set, k=k), num_samples=num_samples)


def cosine(dense_preds):
    offdiag = ~np.eye(dense_preds.shape[0], dtype=bool)
    n = np.linalg.norm(dense_preds, axis=1)[:, np.newaxis]
    return (dense_preds @ dense_preds.T / n / n.T)[offdiag].mean()


def get_top_k(dense_preds, test_set, k=10):
    test_items = collections.defaultdict(list)
    for user_id, item_id in test_set:
        test_items[user_id].append(item_id)

    best_items = collections.defaultdict(set)
    for user_id in range(dense_preds.shape[0]):
        item_ids = np.array(test_items[user_id])
        items = item_ids[np.argsort(dense_preds[user_id][item_ids])[-k:]]
        for item_id in items:
            best_items[user_id].add(item_id)
    return best_items


def _jaccard(recommended_items, num_samples=1000):
    values = []
    user_ids = list(recommended_items)
    pairs = zip(
        np.random.choice(user_ids, num_samples), np.random.choice(user_ids, num_samples)
    )
    for user1, user2 in pairs:
        if user1 != user2:
            intersection = recommended_items[user1].intersection(
                recommended_items[user2]
            )
            union = recommended_items[user1].union(recommended_items[user2])
            values.append(len(intersection) / len(union))
    print(2 * np.std(values) / np.sqrt(num_samples), np.mean(values))
    return np.mean(values)
