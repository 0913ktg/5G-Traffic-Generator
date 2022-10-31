# Cell
from math import sqrt
from typing import Optional, Union

import numpy as np

# Cell
def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div

# Cell
def metric_protections(y: np.ndarray, y_hat: np.ndarray, weights: np.ndarray):
    assert (weights is None) or (np.sum(weights) > 0), 'Sum of weights cannot be 0'
    assert (weights is None) or (weights.shape == y_hat.shape), 'Wrong weight dimension'

# Cell
def mape(y: np.ndarray, y_hat: np.ndarray,
         weights: Optional[np.ndarray] = None,
         axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Mean Absolute Percentage Error.

    MAPE measures the relative prediction accuracy of a
    forecasting method by calculating the percentual deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array, optional
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    mape: numpy array or double
        Return the mape along the specified axis.
    """
    metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    mape = divide_no_nan(delta_y, scale)
    mape = np.average(mape, weights=weights, axis=axis)
    mape = 100 * mape

    return mape

# Cell
def mse(y: np.ndarray, y_hat: np.ndarray,
        weights: Optional[np.ndarray] = None,
        axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Mean Squared Error.

    MSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    mse: numpy array or double
        Return the mse along the specified axis.
    """
    metric_protections(y, y_hat, weights)

    delta_y = np.square(y - y_hat)
    if weights is not None:
        mse = np.average(delta_y[~np.isnan(delta_y)],
                         weights=weights[~np.isnan(delta_y)],
                         axis=axis)
    else:
        mse = np.nanmean(delta_y, axis=axis)

    return mse

# Cell
def rmse(y: np.ndarray, y_hat: np.ndarray,
         weights: Optional[np.ndarray] = None,
         axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Root Mean Squared Error.

    RMSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    rmse: numpy array or double
        Return the rmse along the specified axis.
    """

    return np.sqrt(mse(y, y_hat, weights, axis))

# Cell
def smape(y: np.ndarray, y_hat: np.ndarray,
          weights: Optional[np.ndarray] = None,
          axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Symmetric Mean Absolute Percentage Error.

    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    smape: numpy array or double
        Return the smape along the specified axis.
    """
    metric_protections(y, y_hat, weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.average(smape, weights=weights, axis=axis)

    if isinstance(smape, float):
        assert smape <= 200, 'SMAPE should be lower than 200'
    else:
        assert all(smape <= 200), 'SMAPE should be lower than 200'

    return smape

# Cell
def mase(y: np.ndarray, y_hat: np.ndarray,
         y_train: np.ndarray,
         seasonality: int,
         weights: Optional[np.ndarray] = None,
         axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates the Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    y_train: numpy array
        Actual insample values for Seasonal Naive predictions.
    seasonality: int
        Main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    mase: numpy array or double
        Return the mase along the specified axis.

    References
    ----------
    [1] https://robjhyndman.com/papers/mase.pdf
    """
    delta_y = np.abs(y - y_hat)
    delta_y = np.average(delta_y, weights=weights, axis=axis)

    scale = np.abs(y_train[:-seasonality] - y_train[seasonality:])
    scale = np.average(scale, axis=axis)

    mase = delta_y / scale

    return mase

# Cell
def mae(y: np.ndarray, y_hat: np.ndarray,
        weights: Optional[np.ndarray] = None,
        axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Mean Absolute Error.

    The mean absolute error

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    mae: numpy array or double
        Return the mae along the specified axis.
    """
    metric_protections(y, y_hat, weights)
    y_hat[np.isnan(y_hat)] = 1e-6
    
    delta_y = np.abs(y - y_hat)
    if weights is not None:
        mae = np.average(delta_y[~np.isnan(delta_y)],
                         weights=weights[~np.isnan(delta_y)],
                         axis=axis)
    else:
        mae = np.nanmean(delta_y, axis=axis)

    return mae

# Cell
def pinball_loss(y: np.ndarray, y_hat: np.ndarray, tau: float = 0.5,
                 weights: Optional[np.ndarray] = None,
                 axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates the Pinball Loss.

    The Pinball loss measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for tau is 0.5 for the deviation from the median.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array
        Predicted values.
    weights: numpy array
        Weights for weighted average.
    tau: float
        Fixes the quantile against which the predictions are compared.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    pinball loss: numpy array or double
        Return the pinball loss along the specified axis.
    """
    metric_protections(y, y_hat, weights)

    delta_y = y - y_hat
    pinball = np.maximum(tau * delta_y, (tau - 1) * delta_y)

    if weights is not None:
        pinball = np.average(pinball[~np.isnan(pinball)],
                             weights=weights[~np.isnan(pinball)],
                             axis=axis)
    else:
        pinball = np.nanmean(pinball, axis=axis)

    return pinball

# Cell
def rmae(y: np.ndarray,
         y_hat1: np.ndarray, y_hat2: np.ndarray,
         weights: Optional[np.ndarray] = None,
         axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates Relative Mean Absolute Error.

    The relative mean absolute error of two forecasts.
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat1: numpy array
        Predicted values of first model.
    y_hat2: numpy array
        Predicted values of second model.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    rmae: numpy array or double
        Return the rmae along the specified axis.
    """
    numerator = mae(y=y, y_hat=y_hat1, weights=weights, axis=axis)
    denominator = mae(y=y, y_hat=y_hat2, weights=weights, axis=axis)
    rmae = numerator / denominator

    return rmae

# Cell
def mqloss(y: np.ndarray, y_hat: np.ndarray,
           quantiles: np.ndarray,
           weights: Optional[np.ndarray] = None,
           axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculates the MultiQuantile loss.

    Calculates Average Multi-quantile Loss function, for
    a given set of quantiles, based on the absolute
    difference between predicted and true values.

    Parameters
    ----------
    y: numpy array
        Actual test values.
    y_hat: numpy array (-1, n_quantiles)
        Predicted values.
    quantiles: numpy array (n_quantiles)
        Quantiles to estimate from the distribution of y.
    weights: numpy array
        Weights for weighted average.
    axis: None or int, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over all of the
        elements of the input array.
        If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    mqloss: numpy array or double
        Return the mqloss along the specified axis.
    """
    metric_protections(y, y_hat, weights)
    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    loss = (quantiles * sq + (1 - quantiles) * s1_q)
    loss = np.average(loss, weights=weights, axis=axis)

    return loss