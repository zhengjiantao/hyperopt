import unittest
import numpy.testing as npt
from hyperopt.rdists import loguniform_gen
from scipy import stats
from scipy.stats.tests.test_continuous_basic import (
    check_cdf_logcdf,
    check_pdf_logpdf,
    check_pdf,
    check_cdf_ppf,
    )


class TestLogUniform(unittest.TestCase):
    def test_cdf_logcdf(self):
        check_cdf_logcdf(loguniform_gen(0, 1), (0, 1), '')
        check_cdf_logcdf(loguniform_gen(0, 1), (-5, 5), '')

    def test_cdf_ppf(self):
        check_cdf_ppf(loguniform_gen(0, 1), (0, 1), '')
        check_cdf_ppf(loguniform_gen(-2, 1), (-5, 5), '')

    def test_pdf_logpdf(self):
        check_pdf_logpdf(loguniform_gen(0, 1), (0, 1), '')
        check_pdf_logpdf(loguniform_gen(low=-4, high=-0.5), (-2, 1), '')

    def test_pdf(self):
        check_pdf(loguniform_gen(0, 1), (0, 1), '')
        check_pdf(loguniform_gen(low=-4, high=-2), (-3, 2), '')

    def test_distribution_rvs(self):
        alpha = 0.01
        loc = 0
        scale = 1
        arg = (loc, scale)
        distfn = loguniform_gen(0, 1)
        D,pval = stats.kstest(distfn.rvs, distfn.cdf, args=arg, N=1000)
        if (pval < alpha):
            npt.assert_(pval > alpha,
                        "D = %f; pval = %f; alpha = %f; args=%s" % (
                            D, pval, alpha, arg))

