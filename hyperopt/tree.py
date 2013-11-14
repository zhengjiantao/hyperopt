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
from .base import miscs_to_idxs_vals
from .algobase import (
    SuggestAlgo,
    ExprEvaluator,
    make_suggest_many_from_suggest_one,
    )
from .pyll_utils import expr_to_config, Cond

logger = logging.getLogger(__name__)

class TreeAlgo(SuggestAlgo):

    def __init__(self, domain, trials, seed):
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

        config = {}
        expr_to_config(domain.expr, None, config)
        # -- conditions that activate each hyperparameter
        self.conditions = dict([(k, config[k]['conditions'])
                            for k in config])
        self.tree_ = self.recursive_split(self.tids, {})

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

        #print 'CONDITIONS:', conditions
        #print 'CRITERIA:', self.conditions
        hps = [k for k, criteria in sorted(self.conditions.items())
               if any(map(satisfied, criteria))]
        #print 'SPLITTABLE:', hps
        if not hps:
            return None

        X = np.empty((len(tids), len(hps)))
        Y = np.empty((len(tids),))
        node_vals = self.node_vals
        node_tids = self.node_tids
        # TODO: better data structures to make this faster
        for ii, tid in enumerate(tids):
            for jj, hp in enumerate(hps):
                X[ii, jj] = node_vals[hp][node_tids[hp].index(tid)]
            Y[ii] = self.losses[list(self.tids).index(tid)]

        #print X
        dtr = DecisionTreeRegressor(max_depth=1)
        dtr.fit(X, Y)
        #print dir(dtr.tree_)
        if dtr.tree_.node_count == 1:
            # -- no split was made
            return None

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
            'hp': hps[feature],
            'thresh': threshold,
            'below': below,
            'above': above,
        }


    def shrinking(self, label):
        T = len(self.node_vals[label])
        return 1.0 / (1.0 + T * self.shrink_coef)

    def choose_ltv(self, label):
        """Returns (loss, tid, val) of best/runner-up trial
        """
        tids = self.node_tids[label]
        vals = self.node_vals[label]
        losses = [self.tid_losses_dct[tid] for tid in tids]

        # -- try to return the value corresponding to one of the
        #    trials that was previously chosen
        tid_set = set(tids)
        for tid in self.best_tids:
            if tid in tid_set:
                idx = tids.index(tid)
                rval = losses[idx], tid, vals[idx]
                break
        else:
            # -- choose a new best idx
            ltvs = sorted(zip(losses, tids, vals))
            best_idx = int(self.rng.geometric(1.0 / self.avg_best_idx)) - 1
            best_idx = min(best_idx, len(ltvs) - 1)
            assert best_idx >= 0
            best_loss, best_tid, best_val = ltvs[best_idx]
            self.best_tids.append(best_tid)
            rval = best_loss, best_tid, best_val
        return rval

    def on_node_hyperparameter(self, memo, node, label):
        """
        Return a new value for one hyperparameter.

        Parameters:
        -----------

        memo - a partially-filled dictionary of node -> list-of-values
               for the nodes in a vectorized representation of the
               original search space.

        node - an Apply instance in the vectorized search space,
               which corresponds to a hyperparameter

        label - a string, the name of the hyperparameter


        Returns: a list with one value in it: the suggested value for this
        hyperparameter


        Notes
        -----

        This function works by delegating to self.hp_HPTYPE functions to
        handle each of the kinds of hyperparameters in hyperopt.pyll_utils.

        Other search algorithms can implement this function without
        delegating based on the hyperparameter type, but it's a pattern
        I've used a few times so I show it here.

        """
        vals = self.node_vals[label]
        if len(vals) == 0:
            return ExprEvaluator.on_node(self, memo, node)
        else:
            loss, tid, val = self.choose_ltv(label)
            try:
                handler = getattr(self, 'hp_%s' % node.name)
            except AttributeError:
                raise NotImplementedError('Annealing', node.name)
            return handler(memo, node, label, tid, val)

    def hp_uniform(self, memo, node, label, tid, val,
                   log_scale=False,
                   pass_q=False,
                   uniform_like=uniform):
        """
        Return a new value for a uniform hyperparameter.

        Parameters:
        -----------

        memo - (see on_node_hyperparameter)

        node - (see on_node_hyperparameter)

        label - (see on_node_hyperparameter)

        tid - trial-identifier of the model trial on which to base a new sample

        val - the value of this hyperparameter on the model trial

        Returns: a list with one value in it: the suggested value for this
        hyperparameter
        """
        if log_scale:
            val = np.log(val)
        high = memo[node.arg['high']]
        low = memo[node.arg['low']]
        assert low <= val <= high
        width = (high - low) * self.shrinking(label)
        new_high = min(high, val + width / 2)
        if new_high == high:
            new_low = new_high - width
        else:
            new_low = max(low, val - width / 2)
            if new_low == low:
                new_high = new_low + width
        assert low <= new_low <= new_high <= high
        if pass_q:
            return uniform_like(
                low=new_low,
                high=new_high,
                rng=self.rng,
                q=memo[node.arg['q']],
                size=memo[node.arg['size']])
        else:
            return uniform_like(
                low=new_low,
                high=new_high,
                rng=self.rng,
                size=memo[node.arg['size']])

    def hp_quniform(self, *args, **kwargs):
        return self.hp_uniform(
            pass_q=True,
            uniform_like=quniform,
            *args,
            **kwargs)

    def hp_loguniform(self, *args, **kwargs):
        return self.hp_uniform(
            log_scale=True,
            pass_q=False,
            uniform_like=loguniform,
            *args,
            **kwargs)

    def hp_qloguniform(self, *args, **kwargs):
        return self.hp_uniform(
            log_scale=True,
            pass_q=True,
            uniform_like=qloguniform,
            *args,
            **kwargs)

    def hp_randint(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        upper = memo[node.arg['upper']]
        counts = np.zeros(upper)
        counts[val] += 1
        prior = self.shrinking(label)
        p = (1 - prior) * counts + prior * (1.0 / upper)
        rval = categorical(p=p, upper=upper, rng=self.rng,
                           size=memo[node.arg['size']])
        return rval

    def hp_categorical(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        p = p_orig = np.asarray(memo[node.arg['p']])
        if p.ndim == 2:
            assert len(p) == 1
            p = p[0]
        counts = np.zeros_like(p)
        counts[val] += 1
        prior = self.shrinking(label)
        new_p = (1 - prior) * counts + prior * p
        if p_orig.ndim == 2:
            rval = categorical(p=[new_p], rng=self.rng,
                               size=memo[node.arg['size']])
        else:
            rval = categorical(p=new_p, rng=self.rng,
                               size=memo[node.arg['size']])
        return rval

    def hp_normal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return normal(
            mu=val,
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            rng=self.rng,
            size=memo[node.arg['size']])

    def hp_lognormal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return lognormal(
            mu=np.log(val),
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            rng=self.rng,
            size=memo[node.arg['size']])

    def hp_qlognormal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return qlognormal(
            # -- prevent log(0) without messing up algo
            mu=np.log(1e-16 + val),
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            q=memo[node.arg['q']],
            rng=self.rng,
            size=memo[node.arg['size']])

    def hp_qnormal(self, memo, node, label, tid, val):
        """
        Parameters: See `hp_uniform`
        """
        return qnormal(
            mu=val,
            sigma=memo[node.arg['sigma']] * self.shrinking(label),
            q=memo[node.arg['q']],
            rng=self.rng,
            size=memo[node.arg['size']])


@make_suggest_many_from_suggest_one
def suggest(new_ids, domain, trials, seed, *args, **kwargs):
    new_id, = new_ids
    return TreeAlgo(domain, trials, seed, *args, **kwargs)(new_id)

# -- flake-8 abhors blank line EOF
