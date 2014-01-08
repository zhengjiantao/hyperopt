"""
Tree-based algorithm for hyperopt

"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging
import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor


from pyll.stochastic import (
    # -- integer
    categorical,
    # randint, -- unneeded
    # -- normal
    normal,
    lognormal,
    qnormal,
    qlognormal,
    # -- uniform
    uniform,
    loguniform,
    quniform,
    qloguniform,
    )
from .base import miscs_to_idxs_vals, Trials
from .algobase import (
    SuggestAlgo,
    ExprEvaluator,
    make_suggest_many_from_suggest_one,
    )
from .pyll_utils import expr_to_config, Cond
import rand
from fmin import fmin

logger = logging.getLogger(__name__)

rng = np.random.RandomState()

def logEI(mean, var, thresh):
    # -- TODO: math for analytic form
    samples = rng.randn(50) * np.sqrt(var) + mean
    samples -= thresh
    return samples[samples < 0].sum()


def UCB(mean, var, zscore):
    return mean - np.sqrt(var) * zscore

from tpe import normal_cdf, lognormal_cdf


def uniform_lpdf(x, low, high):
    return -math.log(high - low)


def loguniform_lpdf(x, low, high):
    assert math.exp(low) <= x <= math.exp(high)
    return -math.log(high - low) - math.log(x)


def uniform_cdf(x, low, high):
    return (x - low) / (high - low)


def loguniform_cdf(x, low, high):
    assert math.exp(low) <= x <= math.exp(high)
    return (math.log(x) - low) / (high - low)


def quniform_lpdf(x, low, high, q):
    lbound = max(low, x - q / 2.0)
    ubound = min(high, x - q / 2.0)
    return np.log(uniform_cdf(ubound, low, high)
        - uniform_cdf(lbound, low, high))


def qloguniform_lpdf(x, low, high, q):
    assert math.exp(low) <= x <= math.exp(high)
    lbound = max(low, x - q / 2.0)
    ubound = min(high, x - q / 2.0)
    return np.log(loguniform_cdf(ubound, low, high)
        - loguniform_cdf(lbound, low, high))


def logprior(config, memo):
    # -- shallow copy
    memo_cpy = dict(memo)
    # -- for each e.g. hyperopt_param('x', uniform(0, 1)) -> 0.33
    #    create another memo entry for uniform(0, 1) -> 0.33
    #    This is useful because the config doesn't have the hyperopt_param nodes,
    #    it only has the e.g. uniform(0, 1).
    for node in memo:
        if node.name == 'hyperopt_param':
            memo_cpy[node.inputs()[1]] = memo[node]

    def logp(apply_node):
        val = memo_cpy[node]
        if 'uniform' in apply_node.name:
            low = apply_node.arg['low'].obj
            high = apply_node.arg['high'].obj
            if 'q' in apply_node.name:
                q = apply_node.arg['q'].obj
            if apply_node.name == 'uniform':
                return uniform_lpdf(val, low, high)
            elif apply_node.name == 'quniform':
                return quniform_lpdf(val, low, high, q)
            elif apply_node.name == 'loguniform':
                return loguniform_lpdf(val, low, high)
            elif apply_node.name == 'qloguniform':
                return qloguniform_lpdf(val, low, high, q)
            else:
                raise NotImplementedError(name) 
        elif apply_node.name == 'randint':
            return -math.log(apply_node.arg['upper'].obj)
        elif apply_node.name == 'lognormal':
            return tpe.lognormal_lpdf(val, 
                mu=apply_node.arg['mu'].obj,
                sigma=apply_node.arg['sigma'].obj)
        elif apply_node.name == 'qlognormal':
            return tpe.qlognormal_lpdf(val, 
                mu=apply_node.arg['mu'].obj,
                sigma=apply_node.arg['sigma'].obj,
                q=apply_node.arg['q'].obj)
        else:
            raise NotImplementedError(apply_node.name)
    logs = [logp(hpvar['node']) for hpvar in config.values()]
    return sum(logs)


class TreeAlgo(SuggestAlgo):

    def __init__(self, domain, trials, seed,
                 sub_suggest=rand.suggest):
        SuggestAlgo.__init__(self, domain, trials, seed=seed)

        self.random_draw_fraction = 0.25  # I think this is what SMAC does
        self.EI_thresh_improvement = 0.1 # ??
        self.n_EI_evals = 200  # misnomer

        doc_by_tid = {}
        for doc in trials.trials:
            tid = doc['tid']
            loss = domain.loss(doc['result'], doc['spec'])
            if loss is None:
                # -- associate infinite loss to new/running/failed jobs
                loss = float('inf')
            else:
                loss = float(loss)
            doc_by_tid[tid] = (doc, loss)
        self.tid_docs_losses = sorted(doc_by_tid.items())
        self.tids = np.asarray([t for (t, (d, l)) in self.tid_docs_losses])
        self.losses = np.asarray([l for (t, (d, l)) in self.tid_docs_losses])
        self.tid_losses_dct = dict(zip(self.tids, self.losses))
        self.node_tids, self.node_vals = miscs_to_idxs_vals(
            [d['misc'] for (tid, (d, l)) in self.tid_docs_losses],
            keys=domain.params.keys())
        self.best_tids = []
        self.sub_suggest = sub_suggest

        config = {}
        expr_to_config(domain.expr, None, config)
        # -- conditions that activate each hyperparameter
        self.config = config
        self.conditions = dict([(k, config[k]['conditions'])
                            for k in config])
        self.tree_ = self.recursive_split(self.tids, {})
        self.best_pt = self.optimize_in_model()

    def recursive_split(self, tids, conditions):
        """

        Paramater
        ---------
        tids
        conditions : dictionary of tuples
            hyperparam name -> Conditions that are True
        """
        def satisfied(activation_conjunction):
            """True iff all conditions are satisfied"""
            for want in activation_conjunction:
                assert want.op == '='
                for have in conditions.get(want.name, ()):
                    assert have.op == '='
                    if want.val == have.val:
                        break
                else:
                    return False
            return True
        if len(tids) < 2:
            return {
                'node': 'leaf',
                'mean': 0,
                'var': 1,
                'n': len(tids)}

        Y = np.empty((len(tids),))
        for ii, tid in enumerate(tids):
            Y[ii] = self.losses[list(self.tids).index(tid)]

        leaf_rval = {
            'node': 'leaf',
            'mean': Y.mean(),
            'var': Y.var(),
            'n': len(Y)}

        #print 'CONDITIONS:', conditions
        #print 'CRITERIA:', self.conditions
        hps = [k for k, criteria in sorted(self.conditions.items())
               if any(map(satisfied, criteria))]
        #print 'SPLITTABLE:', hps
        if not hps:
            return leaf_rval

        X = np.empty((len(tids), len(hps)))
        node_vals = self.node_vals
        node_tids = self.node_tids
        # TODO: better data structures to make this faster
        for ii, tid in enumerate(tids):
            for jj, hp in enumerate(hps):
                X[ii, jj] = node_vals[hp][node_tids[hp].index(tid)]

        #print X
        dtr = DecisionTreeRegressor(
            max_depth=1,
            min_samples_leaf=2,
            )
        dtr.fit(X, Y)
        #print dir(dtr.tree_)
        if dtr.tree_.node_count == 1:
            # -- no split was made
            return leaf_rval

        #print dtr.tree_.node_count
        feature = dtr.tree_.feature[0]
        threshold = dtr.tree_.threshold[0]
        vals = X[:, feature]
        tids_below = [tid for tid, val in zip(tids, vals)
                      if val < threshold]
        tids_above = [tid for tid, val in zip(tids, vals)
                      if val >= threshold]
        vals_below = set(vals[vals < threshold])
        vals_above = set(vals[vals >= threshold])

        fname = hps[feature]
        if len(vals_below) == 1:
            val_below, = list(vals_below)
            cond_below = dict(conditions)
            cond_below[fname] = conditions.get(fname, ()) + (
                Cond(op='=', name=fname, val=val_below),)
        else:
            cond_below = conditions

        if len(vals_above) == 1:
            val_above, = list(vals_above)
            cond_above = dict(conditions)
            cond_above[fname] = conditions.get(fname, ()) + (
                Cond(op='=', name=fname, val=val_above),)
        else:
            cond_above = conditions

        below = self.recursive_split(tids_below, cond_below)
        above = self.recursive_split(tids_above, cond_above)

        return {
            'node': 'split',
            'hp': fname,
            'thresh': threshold,
            'below': below,
            'above': above,
        }

    def optimize_in_model(self):
        # TODO: multiply EI by prior
        # TODO: consider anneal.suggest instead of rand.suggest
        def tree_eval(expr, memo, ctrl):
            assert expr is self.domain.expr
            # -- expr is the search space expression
            # -- memo is a hyperparameter assignment

            def descend_branch(node):
                # -- node is a node in the regression tree
                if node['node'] == 'split':
                    for k, v in memo.items():
                        if k.arg['label'].obj == node['hp']:
                            if v < node['thresh']:
                                return descend_branch(node['below'])
                            else:
                                return descend_branch(node['above'])
                    else:
                        raise Exception('did not find node')
                else:
                    assert node['node'] == 'leaf'
                    mean = node['mean']
                    var = node['var']
                    # zscore is search param
                    # return UCB(mean, var, zscore = 0.7)
                    return mean, var
            mean, var = descend_branch(self.tree_)
            logloss = logEI(mean, var, thresh=0.7)
            logp = logprior(self.config, memo)
            loss = logloss + logp
            return {
                'loss': loss,
                'status': 'ok',
            }
        if len(self.losses) > 0:
            if self.rng.rand() < self.random_draw_fraction:
                # -- some of the time (e.g. 1/4) ignore our model
                #    TODO: mark the points drawn from the prior, because they
                #    are more useful for online [tree] model evaluation.
                max_evals = 1
                EI_thresh = 0 # -- irrelevant with max_evals == 1
            else:
                max_evals = self.n_EI_evals
                EI_thresh = min(self.losses) - self.EI_thresh_improvement
        else:
            max_evals = 1
            EI_thresh = 0 # -- irrelevant with max_evals == 1

        # -- This algorithm (fmin) is dumb for optimizing a single tree, but
        # reasonable for optimizing an ensemble or a tree of non-constant
        # predictors.
        best = fmin(
            tree_eval,
            space=self.domain.expr,
            algo=self.sub_suggest,
            max_evals=max_evals,
            pass_expr_memo_ctrl=True,
            rstate=self.rng,
            )
        return best

    def on_node_hyperparameter(self, memo, node, label):
        if label in self.best_pt:
            return [self.best_pt[label]]
        else:
            return []

    def test_foo(self, max_evals):
        def tree_eval(expr, memo, ctrl):
            def foo(node):
                if node['node'] == 'split':
                    for k, v in memo.items():
                        if k.arg['label'].obj == node['hp']:
                            if v < node['thresh']:
                                return foo(node['below'])
                            else:
                                return foo(node['above'])
                    else:
                        raise Exception('did not find node')
                else:
                    assert node['node'] == 'leaf'
                    mean = node['mean']
                    var = node['var']
                    # zscore is search param
                    return mean, var
            mean, var = foo(self.tree_)
            return {
                'loss': mean,
                'var': var,
                'status': 'ok',
            }
        # -- This algorithm (fmin) is dumb for optimizing a single tree, but
        # reasonable for optimizing an ensemble or a tree of non-constant
        # predictors.
        trials = Trials()
        fmin(
            tree_eval,
            space=self.domain.expr,
            algo=self.sub_suggest,
            trials=trials,
            max_evals=max_evals,
            pass_expr_memo_ctrl=True,
            rstate = self.rng,
            )
        return trials



@make_suggest_many_from_suggest_one
def suggest(new_ids, domain, trials, seed, *args, **kwargs):
    new_id, = new_ids
    return TreeAlgo(domain, trials, seed, *args, **kwargs)(new_id)

# -- flake-8 abhors blank line EOF
