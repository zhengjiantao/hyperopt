"""
Tree-based algorithm for hyperopt

"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import math
import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor


import pyll
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
    )
from .pyll_utils import expr_to_config, Cond
import rand
import anneal
from fmin import fmin
import scipy.stats
import rdists
import criteria


def sample_w_replacement(lst, n, rng):
    if len(lst):
        return np.take(lst, rng.randint(0, len(lst), size=n))
    else:
        return list(lst)


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
        val = memo_cpy[apply_node]
        if val is pyll.base.GarbageCollected:
            # -- XXX: confirm this happens because the hyperparam is unused.
            return 0
        if 'uniform' in apply_node.name:
            low = apply_node.arg['low'].obj
            high = apply_node.arg['high'].obj
            if 'q' in apply_node.name:
                q = apply_node.arg['q'].obj
            if apply_node.name == 'uniform':
                return rdists.uniform_gen(a=low, b=high).logpdf(
                    val, loc=low, scale=(high - low))
            elif apply_node.name == 'quniform':
                return rdists.quniform_gen(
                    low=low, high=high, q=q).logpmf(val)
            elif apply_node.name == 'loguniform':
                return rdists.loguniform_gen(
                    low=low, high=high).logpdf(val)
            elif apply_node.name == 'qloguniform':
                return rdists.qloguniform_gen(
                    low=low, high=high, q=q).logpmf(val)
            else:
                raise NotImplementedError(apply_node.name) 
        elif 'normal' in apply_node.name:
            mu = apply_node.arg['mu'].obj
            sigma = apply_node.arg['sigma'].obj
            if 'q' in apply_node.name:
                q = apply_node.arg['q'].obj
            if apply_node.name == 'normal':
                return scipy.stats.norm(
                    loc=mu, scale=sigma).logpdf(val)
            elif apply_node.name == 'qnormal':
                return rdists.qnormal_gen(
                    mu=mu, sigma=sigma, q=q).logpmf(val)
            elif apply_node.name == 'lognormal':
                return rdists.lognorm_gen(
                    mu=mu, sigma=sigma).logpdf(val)
            elif apply_node.name == 'qlognormal':
                return rdists.qlognormal_gen(
                    mu=mu, sigma=sigma, q=q).logpmf(val)
            else:
                raise NotImplementedError(apply_node.name) 
        elif apply_node.name == 'randint':
            return -math.log(apply_node.arg['upper'].obj)
        elif apply_node.name == 'categorical':
            assert val == int(val), val
            p = pyll.rec_eval(apply_node.arg['p'])
            return math.log(p[int(val)])
        else:
            raise NotImplementedError(apply_node.name)
    logs = [logp(hpvar['node']) for hpvar in config.values()]
    return sum(logs)


class ReplayBest(SuggestAlgo):
    """Suggest previously-suggested points
    """
    def __init__(self, domain, trials, seed, tid_docs_losses):
        self.n_trials_so_far = len(trials.trials)
        self.ltd = sorted([(loss, tid, doc)
                      for (tid, (doc, loss)) in tid_docs_losses])
        SuggestAlgo.__init__(self, domain, trials, seed)

    def on_node_hyperparameter(self, memo, node, label):
        #print 'inserting from', self.n_trials_so_far
        loss, tid, doc = self.ltd[self.n_trials_so_far]
        return doc['misc']['vals'][label]

    def __call__(self, *args, **kwargs):
        rval = SuggestAlgo.__call__(self, *args, **kwargs)
        #print rval[0]['misc']
        # -- mark points from this algorithm.
        #    We must filter them out in TreeAlgo.optimize_in_model.
        for rv in rval:
            rv['misc']['ReplayBest'] = 1
        return rval


class TreeAlgo(SuggestAlgo):

    def __init__(self, domain, trials, seed,
                 n_trees,
                 min_samples_leaf):
        SuggestAlgo.__init__(self, domain, trials, seed=seed)


        # -- extract the information we need from the trials object
        self.trials = trials
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
        # -- sorted by tid ??
        self.tid_docs_losses = sorted(doc_by_tid.items())
        self.tids = np.asarray([t for (t, (d, l)) in self.tid_docs_losses])
        self.losses = np.asarray([l for (t, (d, l)) in self.tid_docs_losses])
        self.tid_losses_dct = dict(zip(self.tids, self.losses))
        self.node_tids, self.node_vals = miscs_to_idxs_vals(
            [d['misc'] for (tid, (d, l)) in self.tid_docs_losses],
            keys=domain.params.keys())
        self.best_tids = []

        # -- use expr_to_config to build dependency graph that tells
        #    us which hyperparams depend on which others
        config = {}
        expr_to_config(domain.expr, None, config)
        self.config = config
        self.conditions = dict([(k, config[k]['conditions'])
                            for k in config])

        # -- build some predictive models of the response surface
        # XXX: init conditions w all discrete hyperparams that happen
        #      to all equal the same value.
        self.min_samples_leaf = min_samples_leaf
        t0 = time.time()
        if n_trees > 1:
            self.trees = [self.recursive_split(
                              sample_w_replacement(self.tids,
                                                   max(len(self.tids), 10),
                                                   self.rng),
                                               conditions={})
                          for ii in range(n_trees)]
        else:
            self.trees = [self.recursive_split(self.tids, conditions={})]
        t1 = time.time()
        print 'building trees took %f seconds' % (t1 - t0)

    def on_node_hyperparameter(self, memo, node, label):
        if label in self.best_pt:
            return self.best_pt[label]
        else:
            return []

    def leaf_node(self, hps, tids, conditions, Y, X):
        if len(tids) == 0:
            print 'warning: estimating mean from 0 samples'
            rval = {
                'node': 'leaf',
                'mean': 0.0, # -- pure guess
                'var': 1.0, # -- pure guess
                'n': len(tids)}
        elif len(tids) == 1:
            rval = {
                'node': 'leaf',
                'mean': Y.mean(),
                'var': 1.0, # -- pure guess
                'n': len(tids)}
        else:
            # -- approx student-T business XXX do it correctly
            tratio = len(tids) / max(0.1, float(len(tids) - 2))
            rval = {
                'node': 'leaf',
                'mean': Y.mean(),
                'var': tratio * Y.var() + 1e-6,
                'n': len(Y)}
        print 'leaf_node', rval
        return rval

    def leaf_node_logEI(self, leaf, memo, thresh):
        # -- negate because EI is improvement *over* thresh, but
        #    fmin is set up for minimization
        return criteria.logEI(-leaf['mean'], leaf['var'], -thresh)

    def leaf_node_meanvar(self, leaf, memo, thresh):
        # -- negate because EI is improvement *over* thresh, but
        #    fmin is set up for minimization
        return leaf['mean'], leaf['var']

    def recursive_split(self, tids, conditions):
        """Return subtree or leaf node to handle `tids`

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

        Y = np.empty((len(tids),))
        for ii, tid in enumerate(tids):
            Y[ii] = self.losses[list(self.tids).index(tid)]

        # -- prepare a model for use in case we don't need to split
        #    tids with another node.

        hps = [k for k, criteria in sorted(self.conditions.items())
               if any(map(satisfied, criteria))]
        if not hps or len(Y) < 2:
            return self.leaf_node(hps, tids, conditions, Y, X=None)

        # TODO: consider limiting the number of features available
        #       in order to be more like random forests.
        #
        #       One reason not to do it though, is that random forests
        #       that use this technique are meant to be much bigger
        #       than the e.g. 10 trees typically fit by TreeAlgo.
        #
        #       We are already randomizing the dataset by bootstrap.
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
            min_samples_leaf=self.min_samples_leaf,
            )
        dtr.fit(X, Y)
        #print dir(dtr.tree_)
        if dtr.tree_.node_count == 1:
            # -- no split was made
            return self.leaf_node(hps, tids, conditions, Y, X)

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


    def optimize_in_model(self, max_evals,
                          sub_suggest,
                          thresh_epsilon,
                          logprior_strength,
                          random_draw_fraction,
                          n_seed_pts,
                          n_random_start_pts,
                         ):
        """
        Parameters
        ----------
        sub_suggest : algo for fmin call to optimize in surrogate
        max_evals : max_evals for fmin call to optimize surrogate
        thresh_epsilon : optimize EI better than (min(losses) - epsilon)
        """
        def descend_branch(node, memo, fn):
            # -- node is a node in the regression tree
            if node['node'] == 'split':
                for k, v in memo.items():
                    if k.arg['label'].obj == node['hp']:
                        if v < node['thresh']:
                            return descend_branch(node['below'], memo, fn)
                        else:
                            return descend_branch(node['above'], memo, fn)
                else:
                    raise Exception('did not find node')
            else:
                assert node['node'] == 'leaf'
                return fn(node, memo, EI_thresh)

        def trees_logEI(expr, memo, ctrl):
            assert expr is self.domain.expr
            # -- expr is the search space expression
            # -- memo is a hyperparameter assignment

            logEIs = [descend_branch(tree, memo, self.leaf_node_logEI)
                      for tree in self.trees]
            # XXX is sign on this right?
            logp = logprior(self.config, memo)
            weighted_logp = logprior_strength * logp
            weighted_logEImean = len(self.tids) * np.mean(logEIs)

            #loss = weighted_logEImean + weighted_logp
            loss = np.mean(logEIs)
            #print np.mean(logEIs)

            return {
                'loss': -loss, # -- improvements are (+) and we're minimizing
                'status': 'ok',
            }

        if len(self.losses) > n_random_start_pts:
            ignore_surrogate = self.rng.rand() < random_draw_fraction
            if ignore_surrogate:
                #    TODO: mark the points drawn from the prior, because they
                #    are more useful for online [tree] model evaluation.
                #    and they provide unbiased estimates of Y mean and var
                #    over search space.
                max_evals = 1
                EI_thresh = 0 # -- irrelevant with max_evals == 1
            else:
                EI_thresh = min(self.losses) - thresh_epsilon
        else:
            max_evals = 1
            EI_thresh = 0 # -- irrelevant with max_evals == 1


        # -- initialize a Trials object with the most promising points from the
        #    original space:

        def _suggest_first_from_best_trials(new_ids, _domain, _trials, _seed):
            # -- return 
            if max_evals == 1:
                return sub_suggest(new_ids, _domain, _trials, _seed)

            if len(_trials.trials) < min(n_seed_pts, len(self.tid_docs_losses)):
                rb = ReplayBest(_domain, _trials, _seed, self.tid_docs_losses)
                return rb(new_ids[0])
            else:
                return sub_suggest(new_ids, _domain, _trials, _seed)

        t0 = time.time()
        rnd_trials = Trials()
        fmin(
            trees_logEI,
            space=self.domain.expr,
            algo=rand.suggest,
            max_evals=max_evals,
            pass_expr_memo_ctrl=True,
            rstate=self.rng,
            trials=rnd_trials,
            )
        tmp_trials = Trials()
        fmin(
            trees_logEI,
            space=self.domain.expr,
            algo=_suggest_first_from_best_trials,
            max_evals=max_evals,
            pass_expr_memo_ctrl=True,
            rstate=self.rng,
            trials=tmp_trials,
            )

        tmp_losses, tmp_docs = (tmp_trials.losses(), tmp_trials.trials)
        assert np.all(np.isfinite(tmp_losses))
        for loss, doc in sorted(zip(tmp_losses, tmp_docs)):
            if doc['misc'].get('ReplayBest', 0):
                continue
            self.best_pt = doc['misc']['vals']
            break


        t1 = time.time()
        if max_evals > 1:
            print 'optimizing surrogate took %f' % (t1 - t0)
            print 'RND', sorted(rnd_trials.losses())[:5]
            print 'ANN', sorted(tmp_trials.losses())[:5]

        if max_evals > 1:
            # -- PLOT CONTOURS
            xs = np.arange(-15, 15, 0.1)
            apply_node = self.config['x']['node']
            ys = []
            vus = []
            vls = []
            for x in xs:
                memo = {pyll.scope.hyperopt_param('x', apply_node): x}
                meanvars = [descend_branch(tree, memo, self.leaf_node_meanvar)
                            for tree in self.trees]
                treemeans, treevars = map(np.asarray,zip(*meanvars))
                ys.append(treemeans.mean())
                vus.append(max(treemeans + np.sqrt(treevars)))
                vls.append(min(treemeans - np.sqrt(treevars)))

            print 'vus', vus[:10]
            print 'vls', vls[:10]

            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.cla()
            # -- show error bars
            plt.plot(xs, ys)
            plt.axhline(EI_thresh, c='g')
            plt.vlines(xs, vls, vus)
            Xs = [t['misc']['vals']['x'][0] for t in self.trials.trials]
            Ys = [t['result']['loss'] for t in self.trials.trials]
            plt.scatter(Xs, Ys, c='b')
            plt.axvline(self.best_pt['x'][0], linestyle='dashed')
            plt.show()

        return self.best_pt


def suggest(new_ids, domain, trials, seed,
        n_optimize_in_model_calls=200,
        thresh_epsilon=0.1, # XXX really need better default :/
        n_trees=10,
        sub_suggest=anneal.suggest,
        logprior_strength=5.0,
        ):
    new_id, = new_ids
    tree_algo = TreeAlgo(domain, trials, seed,
            n_trees=n_trees,
            min_samples_leaf=2,
            )
    tree_algo.optimize_in_model(
        max_evals=n_optimize_in_model_calls,
        sub_suggest=sub_suggest,
        thresh_epsilon=thresh_epsilon,
        random_draw_fraction=0.25,  # SMAC sets this at 0.25
        logprior_strength=logprior_strength,
        n_seed_pts=3, # -- seed surrogate search with known good points
        n_random_start_pts=5, # -- ignore SMBO for these first points
        )
    return tree_algo(new_id)

# -- flake-8 abhors blank line EOF
