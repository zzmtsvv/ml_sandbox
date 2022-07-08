import numpy as np


def mean_squared_error(y_true, preds, sample_weights=None):
    '''
    :param y_true: list of real numbers, true values
    :param preds: list of real numbers, predicted values
    :sample_weights: array-like of shape (n_samples,), default=None
    :return: mean squared error (gives a higher penalty to large errors and vice versa) [0; +inf)
    '''
    return np.average((np.array(y_true) - np.array(preds)) ** 2, weights=sample_weights, axis=0)


def mean_absolute_error(y_true, preds, sample_weight=None):
    '''
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :sample_weights: array-like of shape (n_samples,), default=None
    :returns: mean absolute error [0; +inf)
    '''
    return np.average(np.abs(np.array(y_true) - np.array(preds)), weights=sample_weight, axis=0)


def median_absolute_error(y_true, preds):
    '''
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :returns: median absolute error (robust for outliers) [0; +inf)
    '''
    return np.median(np.abs(np.array(y_true) - np.array(preds)), axis=0)


def rmse(y_true, preds, sample_weights=None):
    '''
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :sample_weights: array-like of shape (n_samples,), default=None
    :returns: root mean squared error (for scaling up to appropriate unit) [0; +inf)
    '''
    return np.sqrt(mean_squared_error(y_true, preds, sample_weights))


def mean_absolute_percentage_error(y_true, preds, eps=1e-5):
    '''
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :returns: mean absolute percentage error [0; +inf)
    '''
    return 100 * np.mean(np.abs((np.array(y_true) - np.array(preds)) / (np.array(y_true) + eps)))


def mean_squared_log_error(y_true, preds, sample_weights=None):
    '''
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :sample_weights: array-like of shape (n_samples,), default=None
    :returns: mean squared logarithmic error (give more weight to small mistakes as well) [0; +inf)
    '''
    return mean_squared_error(np.log1p(y_true), np.log1p(preds), sample_weights)


def acper(y_true, preds, threshold=0.2):
    '''
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :param threshold: parameter that define arbitrary range in which values fall in. the parameter may take a value in [0; 1)
    :returns: Almost Correct Predictions Error Rate (acper score) [0; 1]
    '''
    counter = 0

    for ground_truth, pred in zip(y_true, preds):
        lower_bound = ground_truth * (1 - threshold)
        upper_bound = ground_truth * (1 + threshold)
        if lower_bound <= pred <= upper_bound:
            counter += 1
    return counter / len(y_true)


def r2_score(y_true, preds, sample_weights=None):
    '''
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :param sample_weights: array-like of shape (n_samples,), default=None
    :returns: R squared (coefficient of determination) (-inf; 1]
    (in econometrics, this can be interpreted as the percentage of variance explained by the model)
    '''

    y_true, preds = np.array(y_true), np.array(preds)

    if sample_weights is not None:
        # residual sum of squares
        sse = (sample_weights * (y_true - preds) ** 2).sum(axis=0)
        # total sum of squares (proportional to the variance of the data)
        tse = (sample_weights * (y_true - np.average(y_true, axis=0, weights=sample_weights)) ** 2).sum(axis=0)

    else:
        sse = sum((y_true - preds) ** 2)
        tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    
    r2 = 1 - (sse / tse)
    return r2
