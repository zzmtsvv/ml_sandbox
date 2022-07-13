import numpy as np


def calculate_confusion_stats(y_true, preds):
    TP, FN, FP, TN = 0, 0, 0, 0

    for ground_truth, pred in zip(y_true, preds):
        if ground_truth == 1:
            if pred == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred == 1:
                FP += 1
            else:
                TN += 1
    
    return TP, FN, FP, TN


def precision_recall_fbeta(y_true, preds, beta=1):
    TP, FN, FP, TN = calculate_confusion_stats(y_true, preds)

    # the proportion of relevant objects actually found in the list of all objects returned by the model
    precision = TP / (TP + FP)
    
    # the ratio of relevant objects returned by the algorithm, compared to the total number of relevant objects should have been returned
    recall = TP / (TP + FN)

    beta2 = beta * beta

    fbeta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)

    return precision, recall, fbeta


def cohen_kappa_score(y_true, preds):
    '''
    performance metric that applies to both multiclass and imbalanced learning problems. The advantage of it over accuracy
    is that Cohen's kappa tells you how much better your classification problem is performed, compared to a classifier 
    that randomly guesses a class accroding to the frequency of each class
    (temporally realized only for binary classification)
    '''

    TP, FN, FP, TN = calculate_confusion_stats(y_true, preds)
    
    denom = TP + FN + FP + TN

    # the observed agreement
    p0 = (TP + TN) / denom

    p_class1 = (TP + FN) * (TP + FP) / (denom * denom)
    p_class2 = (FP + TN) * (FN + TN) / (denom * denom)

    # the expected agreement
    pe = p_class1 + p_class2

    kappa = (p0 - pe) / (1 - pe)

    return kappa


'''
    When reporting the model performance, sometimes, besides the value
    of the metric, it is required to also provide the statistical bounds,
    also known as the statistical interval
'''


def calculate_delta_accuracy(acc, n, confidence_level):
    '''
    calc delta = z_N * sqrt(err * (1 - err) / N), where err = 1 - accuracy and N - size of the set
    so that with prob=confidence_level "err" lies in the interval [err - delta; err + delta]

    :param acc: accuracy of the model on the valid/test size
    :param n: the size of the set
    :param confidence_level: probability (0; 100) of the error lying in the specific interval
    :return: boundaries of the interval - tuple(err - delta, err + delta)
    '''
    err = 1 - acc

    def get_z_N(conf_level):
        from scipy.stats import norm
        zn = norm.ppf(1 - 0.5 * (1 - conf_level / 100.0))
        return zn

    z_n = get_z_N(confidence_level)
    delta = z_n * np.sqrt(acc * (1 - acc) / n)
    return err - delta, err + delta


# bootstrapping statistical interval


def get_interval(values, confidence_level):
    '''
    obtaining a "confidence_level" percent statistical interval for the metric according to the values list
    '''
    lower = np.percentile(values, (100.0 - confidence_level) / 2.0)
    upper = np.percentile(values, confidence_level + ((100.0 - confidence_level) / 2.0))
    return lower, upper
