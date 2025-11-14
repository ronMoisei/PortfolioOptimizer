from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Stock:
    """
    Rappresenta un titolo azionario con:
    - ticker
    - settore (se disponibile)
    - rating_score (None se il rating non è disponibile)
    - rating_date (data dell'ultimo rating, se disponibile)
    - prices: serie dei prezzi 'close'
    - returns: rendimenti log
    """
    ticker: str
    sector: Optional[str]
    rating_score: Optional[float]
    rating_date: Optional[pd.Timestamp] = None

    prices: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None

    def set_prices(self, prices: pd.Series) -> None:
        """
        Imposta la serie dei prezzi (close) e calcola i rendimenti log.
        """
        prices = prices.sort_index().astype(float)
        self.prices = prices

        # log-return: log(p_t / p_{t-1}) = log(p_t) - log(p_{t-1})
        log_prices = prices.apply(np.log)          # Series
        self.returns = log_prices.diff()
