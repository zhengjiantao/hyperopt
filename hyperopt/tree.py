"""
Tree-based algorithm for hyperopt

"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging

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

def EI(mean, var, thresh):
    raise NotImplementedError()


def UCB(mean, var, zscore):
    return mean + np.sqrt(var) * zscore


class TreeAlgo(SuggestAlgo):

    def __init__(self, domain, trials, seed,
                 sub_suggest=rand.suggest):
        SuggestAlgo.__init__(self, domain, trials, seed=seed)

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
                    return UCB(mean, var, zscore = 0.7)
            loss = foo(self.tree_)
            return {
                'loss': loss,
                'status': 'ok',
            }
        if len(self.losses) > 0:
            max_evals = 200
        else:
            max_evals = 1
        # -- This algorithm (fmin) is dumb for optimizing a single tree, but
        # reasonable for optimizing an ensemble or a tree of non-constant
        # predictors.
        best = fmin(
            tree_eval,
            space=self.domain.expr,
            algo=self.sub_suggest,
            max_evals=max_evals,
            pass_expr_memo_ctrl=True,
            )
        return best

    def on_node_hyperparameter(self, memo, node, label):
        if label in self.best_pt:
            return [self.best_pt[label]]
        else:
            return []


@make_suggest_many_from_suggest_one
def suggest(new_ids, domain, trials, seed, *args, **kwargs):
    new_id, = new_ids
    return TreeAlgo(domain, trials, seed, *args, **kwargs)(new_id)

# -- flake-8 abhors blank line EOF
