# Cell
from typing import Union, List, Optional

import torch as t
from fastcore.foundation import patch

from .pytorch import (
    MAPELoss, MASELoss, SMAPELoss,
    MSELoss, MAELoss, SmylLoss,
    PinballLoss, MQLoss, wMQLoss
)

# Cell
class LossFunction:
    def __init__(self, loss_name: str, seasonality: Optional[int] = None,
                 percentile: Optional[Union[List[int], int]] = None,
                 level_variability_penalty: Optional[int] = None) -> 'LossFunction':
        """Instantiates a callable class of the `loss_name` loss.

        Parameters
        ----------
        loss_name: str
            Name of the loss.
        seasonality: int
            main frequency of the time series
            Hourly 24,  Daily 7, Weekly 52,
            Monthly 12, Quarterly 4, Yearly.
            Default `None`.
            Mandatory for MASE loss.
        percentile: Union[List[int], int]
            Target percentile.
            For SMYL and PINBALL losses an int
            is expected.
            For MQ and wMQ losses a list of ints
            is expected.
            Default `None`.
        level_variability_penalty: int
            Only used for SMYL loss.
        """
        if loss_name in ['SMYL', 'PINBALL'] and not isinstance(percentile, int):
            raise Exception(f'Percentile should be integer for {loss_name} loss.')
        elif loss_name in ['MQ', 'wMQ'] and not isinstance(percentile, list):
            raise Exception(f'Percentile should be list for {loss_name} loss')
        elif loss_name == 'MASE' and seasonality is None:
            raise Exception(f'Seasonality should be a list of integers for {loss_name} loss')


        self.loss_name = loss_name
        self.seasonality = seasonality
        self.percentile = percentile
        self.level_variability_penalty = level_variability_penalty

        self.tau = self.percentile / 100 if isinstance(percentile, int) else None
        self.quantiles = [tau / 100 for tau in percentile] if isinstance(percentile, list) else None

# Cell
@patch
def __call__(self: LossFunction,
             y: t.Tensor,
             y_hat: t.Tensor,
             mask: Optional[t.Tensor] = None,
             y_insample: Optional[t.Tensor] = None,
             levels: Optional[t.Tensor] = None) -> t.Tensor:
    """Returns loss according to `loss_name`."""
    if self.loss_name == 'SMYL':
        return SmylLoss(y=y, y_hat=y_hat, levels=levels, mask=mask,
                        tau=self.tau,
                        level_variability_penalty=self.level_variability_penalty)

    elif self.loss_name == 'PINBALL':
        return PinballLoss(y=y, y_hat=y_hat, mask=mask,
                           tau=self.tau)

    elif self.loss_name == 'MQ':
        quantiles = t.Tensor(self.quantiles, device=y.device)
        return MQLoss(y=y, y_hat=y_hat, quantiles=quantiles, mask=mask)

    elif self.loss_name == 'wMQ':
        quantiles = t.Tensor(self.quantiles, device=y.device)
        return wMQLoss(y=y, y_hat=y_hat, quantiles=quantiles, mask=mask)

    elif self.loss_name == 'MAPE':
        return MAPELoss(y=y, y_hat=y_hat, mask=mask)

    elif self.loss_name == 'MASE':
        return MASELoss(y=y, y_hat=y_hat, y_insample=y_insample,
                        seasonality=self.seasonality, mask=mask)

    elif self.loss_name == 'SMAPE':
        return SMAPELoss(y=y, y_hat=y_hat, mask=mask)

    elif self.loss_name == 'MSE':
        return MSELoss(y=y, y_hat=y_hat, mask=mask)

    elif self.loss_name == 'MAE':
        return MAELoss(y=y, y_hat=y_hat, mask=mask)

    raise Exception(f'Unknown loss function: {loss_name}')