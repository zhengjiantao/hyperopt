from functools import partial
import unittest
import numpy as np
from hyperopt import tree
from hyperopt import rand
from hyperopt import Trials, fmin

from test_bandits import CasePerBandit

def passthrough(x):
    return x

class TestItJustRuns(unittest.TestCase, CasePerBandit):
    def work(self):
        trials = Trials()
        space = self.bandit.expr
        fmin(
            fn=passthrough,
            space=space,
            trials=trials,
            algo=tree.suggest,
            max_evals=10)


def test_distractor():
    # -- Even with the 1/4 of random trials thrown in by current tree
    #    implementation, this test still fails for several random seeds.
    #    This is sort of Gaussian-Process stomping grounds... interesting
    #    to think about how other models can get a natural sort of
    #    uncertainty estimate.
    from test_domains import distractor
    trials = Trials()
    fmin(fn=lambda x: x,
        space=distractor().expr,
        trials=trials,
        algo=tree.suggest,
        rstate=np.random.RandomState(125),
        max_evals=75)
    import matplotlib.pyplot as plt
    Xs = [t['misc']['vals']['x'][0] for t in trials.trials]
    Ys = [t['result']['loss'] for t in trials.trials]
    plt.scatter(Xs, Ys, c='b')

    ta = tree.TreeAlgo(distractor(), trials, 123)
    rtrials = ta.test_foo(50)
    Xs = [t['misc']['vals']['x'][0] for t in rtrials.trials]
    Ys = [t['result']['loss'] for t in rtrials.trials]
    Vs = [np.sqrt(t['result']['var']) for t in rtrials.trials]
    print 'n Xs', len(Xs)
    print 'X range', min(Xs), max(Xs)
    print 'Y range', min(Ys), max(Ys)
    print 'vars', Vs
    XYV = sorted(zip(Xs, Ys, Vs))
    Xs, Ys, Vs = zip(*XYV)
    plt.errorbar(Xs, Ys, yerr=Vs)
    plt.show()



class TestItAtLeastSortOfWorks(unittest.TestCase, CasePerBandit):
    thresholds = dict(
            quadratic1=1e-5,
            q1_lognormal=0.01,
            distractor=-1.96,
            gauss_wave=-2.0,
            gauss_wave2=-2.0,
            n_arms=-2.5,
            many_dists=.0005,
            )

    LEN = dict(
            # -- running a long way out tests overflow/underflow
            #    to some extent
            #quadratic1=100,
            #many_dists=200,
            #q1_lognormal=100,
            )

    def setUp(self):
        self.olderr = np.seterr('raise')
        np.seterr(under='ignore')

    def tearDown(self, *args):
        np.seterr(**self.olderr)

    def work(self):
        bandit = self.bandit
        assert bandit.name is not None
        algo = partial(
            tree.suggest,
                )
        LEN = self.LEN.get(bandit.name, 50)

        trials = Trials()
        fmin(fn=passthrough,
            space=self.bandit.expr,
            trials=trials,
            algo=algo,
            max_evals=LEN)
        assert len(trials) == LEN

        if 1:
            rtrials = Trials()
            fmin(fn=passthrough,
                space=self.bandit.expr,
                trials=rtrials,
                algo=rand.suggest,
                max_evals=LEN)
            print 'RANDOM BEST 6:', list(sorted(rtrials.losses()))[:6]

        if 0:
            plt.subplot(2, 2, 1)
            plt.scatter(range(LEN), trials.losses())
            plt.title('TPE losses')
            plt.subplot(2, 2, 2)
            plt.scatter(range(LEN), ([s['x'] for s in trials.specs]))
            plt.title('TPE x')
            plt.subplot(2, 2, 3)
            plt.title('RND losses')
            plt.scatter(range(LEN), rtrials.losses())
            plt.subplot(2, 2, 4)
            plt.title('RND x')
            plt.scatter(range(LEN), ([s['x'] for s in rtrials.specs]))
            plt.show()
        if 0:
            plt.hist(
                    [t['x'] for t in self.experiment.trials],
                    bins=20)

        #print trials.losses()
        print 'ANNEAL BEST 6:', list(sorted(trials.losses()))[:6]
        #logx = np.log([s['x'] for s in trials.specs])
        #print 'TPE MEAN', np.mean(logx)
        #print 'TPE STD ', np.std(logx)
        thresh = self.thresholds[bandit.name]
        print 'Thresh', thresh
        assert min(trials.losses()) < thresh


