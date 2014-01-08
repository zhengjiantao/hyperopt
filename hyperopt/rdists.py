import numpy as np
import numpy.random as mtrand
from scipy.stats import rv_continuous, rv_discrete


class loguniform_gen(rv_continuous):
    """ Stats for Y = e^X where X ~ U(low, high).

    """
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

    def _logpdf(self, x):
        return - np.log(x) - np.log(self._high - self._low)

    def _cdf(self, x):
        return (np.log(x) - self._low) / (self._high - self._low)


class quniform_gen(rv_discrete):
    """ Stats for Y = q * round(X / q) where X ~ U(low, high).

    """
    def __init__(self, low, high, q):
        qlow = np.round(low / q) * q
        qhigh = np.round(high / q) * q
        xs = np.arange(qlow, qhigh, q)
        ps = [1.0 / len(xs) for xk in xs]
        rv_discrete.__init__(self, name='custm', values=(xs, ps))

    def _rvs(self):
        rval = mtrand.uniform(low=self.a, high=self.b)
        return np.round(rval / self.inc) * self.inc

