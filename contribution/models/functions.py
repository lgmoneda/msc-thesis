import pandas as pd
import numpy as np
import math
import pdb

from sklearn import metrics

def std_agg(cnt, s1, s2):
    try:
        return math.sqrt((s2/cnt) - (s1/cnt)**2)
    except:
        ### When negative
        return 0

# def check_min_sample_periods_dict(count_dict, min_sample_periods):
#     df = pd.DataFrame.from_dict(count_dict, orient="index")
#     return (df > min_sample_periods).sum().values[0]

def check_min_sample_periods_dict(count_dict, min_sample_periods):
    for key in count_dict.keys():
        if count_dict[key] < min_sample_periods:
            return False
    return True

def check_min_sample_periods(X, time_column, min_sample_periods):
    return (X[time_column].value_counts() >= min_sample_periods).sum()

def initialize_period_dict(periods):
    period_dict = {}
    for period in periods:
        period_dict[period] = {}
        period_dict[period]["count"] = 0
        period_dict[period]["sum"] = 0
        period_dict[period]["squared_sum"] = 0

    return period_dict


def fill_right_dict(periods, target, weights, right_dict):
    for period in np.unique(periods):

        right_dict[period]["count"] = (weights[periods == period]).sum()
        right_dict[period]["sum"] = (target[periods == period] * weights[periods == period]).sum()
        right_dict[period]["squared_sum"] = ((target[periods == period] ** 2) * weights[periods == period]).sum()

    return right_dict

def std_score_by_period(right_dict, left_dict):
    current_score = []
    for key in right_dict.keys():

        left_std = std_agg(left_dict[key]["count"],
                           left_dict[key]["sum"],
                           left_dict[key]["squared_sum"])

        right_std = std_agg(right_dict[key]["count"],
                           right_dict[key]["sum"],
                           right_dict[key]["squared_sum"])

        total_count = left_dict[key]["count"] + right_dict[key]["count"]
        current_score.append(left_std * (left_dict[key]["count"] / total_count) + right_std * (right_dict[key]["count"] / total_count))

    return current_score


def gini_impurity_score_by_period(right_dict, left_dict):
    current_score = []
    for key in right_dict.keys():

        left_proba = left_dict[key]["sum"] / float(left_dict[key]["count"])
        left_gini = 1 - ((1 - left_proba) ** 2 + (left_proba) ** 2)

        right_proba = right_dict[key]["sum"] / float(right_dict[key]["count"])
        right_gini = 1 - ((1 - right_proba) ** 2 + (right_proba) ** 2)

        total_count = left_dict[key]["count"] + right_dict[key]["count"]
        current_score.append(left_gini * (left_dict[key]["count"] / total_count) + right_gini * (right_dict[key]["count"] / total_count))

    return current_score


def gini_impurity_score(left_dict, right_dict):
    left_proba = left_dict["sum"] / float(left_dict["count"])
    left_gini = 1 - ((1 - left_proba) ** 2 + (left_proba) ** 2)

    right_proba = right_dict["sum"] / float(right_dict["count"])
    right_gini = 1 - ((1 - right_proba) ** 2 + (right_proba) ** 2)

    total_count = left_dict["count"] + right_dict["count"]
    return left_gini * (left_dict["count"] / total_count)\
        + right_gini * (right_dict["count"] / total_count)


def gini_impurity_score_by_period(right_dict, left_dict, include_aggregated=False):
    """
    Calculate the gini impurity score by period given two dictionaries that
    charactize the left and right leaf after the potential split.
    """
    current_score = []
    agg_dict = {"left": {"sum": 0, "count": 0},
                "right":{"sum": 0, "count": 0}}

    for key in right_dict.keys():

        left_proba = left_dict[key]["sum"] / float(left_dict[key]["count"])
        left_gini = 1 - ((1 - left_proba) ** 2 + (left_proba) ** 2)

        right_proba = right_dict[key]["sum"] / float(right_dict[key]["count"])
        right_gini = 1 - ((1 - right_proba) ** 2 + (right_proba) ** 2)

        total_count = left_dict[key]["count"] + right_dict[key]["count"]
        current_score.append(
            left_gini * (left_dict[key]["count"] / total_count)
            + right_gini * (right_dict[key]["count"] / total_count)
        )

        if include_aggregated:
            agg_dict["left"]["sum"] += left_dict[key]["sum"]
            agg_dict["left"]["count"] += left_dict[key]["count"]
            agg_dict["right"]["sum"] += right_dict[key]["sum"]
            agg_dict["right"]["count"] += right_dict[key]["count"]

    if include_aggregated:
        current_score.append(gini_impurity_score(agg_dict["left"], agg_dict["right"]))
    return current_score

def score_by_period(right_dict, left_dict, criterion="std",
                    period_criterion="avg", verbose=False):
    if criterion == "gini":
        current_score = gini_impurity_score_by_period(right_dict, left_dict)

    if criterion == "std":
        current_score = std_score_by_period(right_dict, left_dict)

    if verbose: print("Score {} by period: {}".format(criterion, current_score))

    if period_criterion == "avg":
        return np.mean(current_score)
    else:
        return np.max(current_score)

### Functions related to the support check
def check_categoricals_match(data, categorical_features, environment_column, verbose=True):
    aggregated_metric = []
    for categorical_feature in categorical_features:
        all_categories = data[categorical_feature].unique()
        result = data.groupby(environment_column)[categorical_feature].unique().apply(lambda x: np.sum([1 for i in all_categories if i in x]) / len(all_categories))
        if verbose: print(result)
        aggregated_metric.append(np.mean(result))
    return np.mean(aggregated_metric)

def check_numerical_match(data, numerical_features, environment_column, verbose=True, n_q=10):
    aggregated_metric = []
    for numerical_feature in numerical_features:
        data[numerical_feature + "_quant"] = pd.cut(data[numerical_feature], bins=n_q, labels=[i for i in range(1, n_q+1)])
        all_categories = data[numerical_feature + "_quant"].unique()
        result = data.groupby(environment_column)[numerical_feature + "_quant"].unique().apply(lambda x: np.sum([1 for i in all_categories if i in x]) / len(all_categories))
        if verbose: print(result)
        aggregated_metric.append(np.mean(result))
    return np.mean(aggregated_metric)
