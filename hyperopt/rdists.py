import numpy as np
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


import unittest
from scipy.stats.tests.test_continuous_basic import (
        check_cdf_logcdf,
        check_pdf_logpdf,
        check_pdf,
        check_cdf_ppf,
        )

class TestLogUniform(unittest.TestCase):
    def test_cdf_logcdf(self):
        check_cdf_logcdf(loguniform_gen(), (0, 1), 'loguniform')
        check_cdf_logcdf(loguniform_gen(), (-5, 5), 'loguniform')

    def test_cdf_ppf(self):
        check_cdf_ppf(loguniform_gen(), (0, 1), 'loguniform')
        check_cdf_ppf(loguniform_gen(-2, 1), (-5, 5), 'loguniform')

    def test_pdf_logpdf(self):
        check_pdf_logpdf(loguniform_gen(), (0, 1), 'loguniform')
        check_pdf_logpdf(loguniform_gen(low=-4, high=-0.5), (-2, 1), 'loguniform')

    def test_pdf(self):
        check_pdf(loguniform_gen(), (0, 1), 'loguniform')
        check_pdf(loguniform_gen(low=-4, high=-2), (-3, 2), 'loguniform')

