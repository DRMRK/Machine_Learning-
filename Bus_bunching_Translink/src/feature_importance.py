# We use permutation importance for this
from sklearn.inspection import permutation_importance


def feature_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    sorted_idx = result.importances_mean.argsort()
    return result, sorted_idx
