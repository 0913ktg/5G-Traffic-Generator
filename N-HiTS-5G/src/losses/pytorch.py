# Cell
import torch as t
import torch.nn as nn

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
def MAPELoss(y, y_hat, mask=None):
    """MAPE Loss

    Calculates Mean Absolute Percentage Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.
    As defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mape:
    Mean absolute percentage error.
    """
    if mask is None: mask = t.ones_like(y_hat)

    mask = divide_no_nan(mask, t.abs(y))
    mape = t.abs(y - y_hat) * mask
    mape = t.mean(mape)
    return mape

# Cell
def MSELoss(y, y_hat, mask=None):
    """MSE Loss

    Calculates Mean Squared Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mse:
    Mean Squared Error.
    """
    if mask is None: mask = t.ones_like(y_hat)

    mse = (y - y_hat)**2
    mse = mask * mse
    mse = t.mean(mse)
    return mse

# Cell
def RMSELoss(y, y_hat, mask=None):
    """RMSE Loss

    Calculates Mean Squared Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    rmse:
    Root Mean Squared Error.
    """
    if mask is None: mask = t.ones_like(y_hat)

    rmse = (y - y_hat)**2
    rmse = mask * rmse
    rmse = t.sqrt(t.mean(rmse))
    return rmse

# Cell
def SMAPELoss(y, y_hat, mask=None):
    """SMAPE2 Loss

    Calculates Symmetric Mean Absolute Percentage Error.
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
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.

    Returns
    -------
    smape:
        symmetric mean absolute percentage error

    References
    ----------
    [1] https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
    """
    if mask is None: mask = t.ones_like(y_hat)

    delta_y = t.abs((y - y_hat))
    scale = t.abs(y) + t.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = smape * mask
    smape = 2 * t.mean(smape)
    return smape

# Cell
def MASELoss(y, y_hat, y_insample, seasonality, mask=None) :
    """ Calculates the M4 Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    seasonality: int
        main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1
    y: tensor (batch_size, output_size)
        actual test values
    y_hat: tensor (batch_size, output_size)
        predicted values
    y_train: tensor (batch_size, input_size)
        actual insample values for Seasonal Naive predictions

    Returns
    -------
    mase:
        mean absolute scaled error

    References
    ----------
    [1] https://robjhyndman.com/papers/mase.pdf
    """
    if mask is None: mask = t.ones_like(y_hat)

    delta_y = t.abs(y - y_hat)
    scale = t.mean(t.abs(y_insample[:, seasonality:] - \
                            y_insample[:, :-seasonality]), axis=1)
    mase = divide_no_nan(delta_y, scale[:, None])
    mase = mase * mask
    mase = t.mean(mase)
    return mase

# Cell
def MAELoss(y, y_hat, mask=None):
    """MAE Loss

    Calculates Mean Absolute Error between
    y and y_hat. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mae:
    Mean absolute error.
    """
    if mask is None: mask = t.ones_like(y_hat)

    mae = t.abs(y - y_hat) * mask
    mae = t.mean(mae)
    return mae

# Cell
def PinballLoss(y, y_hat, mask=None, tau=0.5):
    """Pinball Loss
    Computes the pinball loss between y and y_hat.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    tau: float, between 0 and 1
        the slope of the pinball loss, in the context of
        quantile regression, the value of tau determines the
        conditional quantile level.

    Returns
    -------
    pinball:
        average accuracy for the predicted quantile
    """
    if mask is None: mask = t.ones_like(y_hat)

    delta_y = t.sub(y, y_hat)
    pinball = t.max(t.mul(tau, delta_y), t.mul((tau - 1), delta_y))
    pinball = pinball * mask
    pinball = t.mean(pinball)
    return pinball

# Cell
def LevelVariabilityLoss(levels, level_variability_penalty):
    """ Level Variability Loss
    Computes the variability penalty for the level.

    Parameters
    ----------
    levels: tensor with shape (batch, n_time)
        levels obtained from exponential smoothing component of ESRNN
    level_variability_penalty: float
        this parameter controls the strength of the penalization
        to the wigglines of the level vector, induces smoothness
        in the output

    Returns
    ----------
    level_var_loss:
        wiggliness loss for the level vector
    """
    assert levels.shape[1] > 2
    level_prev = t.log(levels[:, :-1])
    level_next = t.log(levels[:, 1:])
    log_diff_of_levels = t.sub(level_prev, level_next)

    log_diff_prev = log_diff_of_levels[:, :-1]
    log_diff_next = log_diff_of_levels[:, 1:]
    diff = t.sub(log_diff_prev, log_diff_next)
    level_var_loss = diff**2
    level_var_loss = level_var_loss.mean() * level_variability_penalty

    return level_var_loss

# Cell
def SmylLoss(y, y_hat, levels, mask, tau, level_variability_penalty=0.0):
    """Computes the Smyl Loss that combines level variability with
    with Pinball loss.
    windows_y: tensor of actual values,
                            shape (n_windows, batch_size, window_size).
    windows_y_hat: tensor of predicted values,
                                    shape (n_windows, batch_size, window_size).
    levels: levels obtained from exponential smoothing component of ESRNN.
                    tensor with shape (batch, n_time).
    return: smyl_loss.
    """

    if mask is None: mask = t.ones_like(y_hat)

    smyl_loss = PinballLoss(y, y_hat, mask, tau)

    if level_variability_penalty > 0:
        log_diff_of_levels = LevelVariabilityLoss(levels, level_variability_penalty)
        smyl_loss += log_diff_of_levels

    return smyl_loss

# Cell
def MQLoss(y, y_hat, quantiles, mask=None):
    """MQLoss

    Calculates Average Multi-quantile Loss function, for
    a given set of quantiles, based on the absolute
    difference between predicted and true values.

    Parameters
    ----------
    y: tensor (batch_size, output_size) actual values in torch tensor.
    y_hat: tensor (batch_size, output_size, n_quantiles) predicted values in torch tensor.
    mask: tensor (batch_size, output_size, n_quantiles) specifies date stamps per serie
          to consider in loss
    quantiles: tensor(n_quantiles) quantiles to estimate from the distribution of y.

    Returns
    -------
    lq: tensor(n_quantiles) average multi-quantile loss.
    """
    assert len(quantiles) > 1, f'your quantiles are of len: {len(quantiles)}'

    if mask is None: mask = t.ones_like(y_hat)

    n_q = len(quantiles)

    error = y_hat - y.unsqueeze(-1)
    sq = t.maximum(-error, t.zeros_like(error))
    s1_q = t.maximum(error, t.zeros_like(error))
    loss = (quantiles * sq + (1 - quantiles) * s1_q)

    return t.mean(t.mean(loss, axis=1))

# Cell
def wMQLoss(y, y_hat, quantiles, mask=None):
    """wMQLoss

    Calculates Average Multi-quantile Loss function, for
    a given set of quantiles, based on the absolute
    difference between predicted and true values.

    Parameters
    ----------
    y: tensor (batch_size, output_size) actual values in torch tensor.
    y_hat: tensor (batch_size, output_size, n_quantiles) predicted values in torch tensor.
    mask: tensor (batch_size, output_size, n_quantiles) specifies date stamps per serie
          to consider in loss
    quantiles: tensor(n_quantiles) quantiles to estimate from the distribution of y.

    Returns
    -------
    lq: tensor(n_quantiles) average multi-quantile loss.
    """
    assert len(quantiles) > 1, f'your quantiles are of len: {len(quantiles)}'

    if mask is None: mask = t.ones_like(y_hat)

    n_q = len(quantiles)

    error = y_hat - y.unsqueeze(-1)

    sq = t.maximum(-error, t.zeros_like(error))
    s1_q = t.maximum(error, t.zeros_like(error))
    loss = (quantiles * sq + (1 - quantiles) * s1_q)

    loss = divide_no_nan(t.sum(loss * mask, axis=-2),
                         t.sum(t.abs(y.unsqueeze(-1)) * mask, axis=-2))

    return t.mean(loss)