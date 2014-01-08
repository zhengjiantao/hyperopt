import numpy as np
import numpy.random as mtrand
from scipy.stats import rv_continuous


class loguniform_gen(rv_continuous):
    def __init__(self, low=0, high=1):
        rv_continuous.__init__(self,
                a=np.exp(low),
                b=np.exp(high))
        self._low = low
        self._high = high

    def _rvs(self):
        rval = np.exp(mtrand.uniform(
            self._low,
            self._high,
            self._size))
        return rval

    def _pdf(self, x):
        return 1.0 / (x * (self._high - self._low))

    def _cdf(self, x):
        log = np.log
        return (log(x) - self._low) / (self._high - self._low)


