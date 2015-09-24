import sys
import time
from copy import copy
import random
import sklearn.cross_validation as cross_validation
from sklearn.base import clone
import numpy as np
from sklearn.metrics.scorer import get_scorer
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import _fit_and_score

class SimulatedAnneal(object):
    def __init__(self, estimator, param_grid, scoring='f1_macro',
                 T=10, T_min=0.0001, alpha=0.75, n_trans=10,
                 max_iter=300, max_runtime=300, cv=3,
                 verbose=False, refit=True, n_jobs=1):

        assert alpha <= 1.0
        assert T > T_min
        assert isinstance(param_grid, dict) or isinstance(param_grid, list)
        # If param_grid is a list of dicts, convert to a single dict
        if isinstance(param_grid, list):
            try:
                param_grid_dict = {}
                for each in param_grid:
                    k,v = each.items()[0]
                    param_grid_dict[k] = v
                param_grid = param_grid_dict
            except:
                sys.stderr.write(str(sys.exc_info()[0]))
                sys.exit()

        # The total number of iterations that can be performed
        n_possible_iters = n_trans*((np.log(T_min)-np.log(T))/np.log(alpha))
        # If fractional max_iter provided, convert to a number
        if 0 < max_iter <= 1:
            max_iter = int(max_iter*n_possible_iters)

        assert hasattr(estimator, 'fit'), "The provided classifer has no fit method."
        assert hasattr(estimator, 'predict'), "The provided estimator has no predict method"
        assert max_iter is not None and max_iter > 0
        if max_iter > n_possible_iters and verbose:
            print("\nWARNING: The value for max_iter=%s does not constrain the number of "
                  "iterations for the specified cooling schedule (%s).  Setting"
                  " max_iter=%s"
                  % (str(max_iter), str(int(n_possible_iters)), str(int(n_possible_iters))))
            max_iter = n_possible_iters

        if verbose:
            print("\nINFO: Number of possible iterations given cooling schedule: %s\n"
                  % str(int(n_possible_iters)))

        # Hidden attributes
        self.__T = T
        self.__T_min = T_min
        self.__alpha = alpha
        self.__max_iter = max_iter
        self.__grid = param_grid
        self.__est = estimator
        self.__verbose = verbose
        self.__n_trans = n_trans
        self.__max_runtime = max_runtime
        self.__cv = cv
        self.__refit = refit
        self.__n_jobs = n_jobs

        # Exposed attributes
        self._scorer = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.grid_scores_ = None
        self.runtime_ = None

    def fit(self, X, y):
        # Set up  the initial params
        T = self.__T
        T_min = self.__T_min
        alpha = self.__alpha
        score_func = self._scorer
        max_iter = self.__max_iter
        n_trans = self.__n_trans
        grid = self.__grid
        max_runtime = self.__max_runtime
        cv = self.__cv

        # Computes the acceptance probability as a function of T; maximization
        accept_prob = lambda old, new, T: np.exp((new-old)/T)

        # Select random values for each parameter and convert to dict
        old_params = dict((k, val.rvs() if hasattr(val, 'rvs') else \
                                   np.random.choice(val)) for k, val in grid.iteritems())

        # Compute the initial score based off randomly selected params
        old_est = clone(self.__est)
        old_est.set_params(**old_params)

        if self.__n_jobs > 1:
            old_score, old_std = MultiProcCvFolds(old_est, score_func, cv, self.__n_jobs,
                                                  self.__verbose).fit_score(X, y)
        else:
            old_score, old_std = CVFolds(old_est, scorer=score_func, cv=cv).fit_score(X, y)

        # Variables to hold the best params
        best_score = old_score
        best_params = old_params

        # Hash table to store states checked and the score for that model
        states_checked = {}
        states_checked[tuple(sorted(old_params.items()))] = (old_score, old_std)
        total_iter = 1
        grid_scores = [(1, T, old_score, old_std, old_params)]

        # If max runtime is not None, the set up the time tracking
        if max_runtime is None:
            max_runtime = 1
            time_at_start = None
        else:
            time_at_start = time.time()
        dt = lambda t0,t1: t1-t0 if t0 is not None else 0
        t_elapsed = dt(time_at_start, time.time())

        while T > T_min and total_iter < max_iter and t_elapsed < max_runtime:
            iter_ = 0
            while iter_ < n_trans:
                total_iter += 1
                # Move to a random neighboring point in param space
                new_params = copy(old_params)
                rand_key = np.random.choice(grid.keys())
                val = grid[rand_key]
                new_rand_key_val = val.rvs() if hasattr(val, 'rvs') else \
                                   np.random.choice(v for v in grid[rand_key] if v != old_params[rand_key])
                new_params[rand_key] = new_rand_key_val
                try:
                    # Look to see if the score has been computed for the given params
                    new_score, new_std = states_checked[tuple(sorted(new_params.items()))]
                except:
                    # If unseen train estimator on new params and store score
                    new_est = clone(self.__est)
                    new_est.set_params(**new_params)
                    new_score, new_std = MultiProcCvFolds(new_est, score_func, cv, self.__n_jobs,
                                                          self.__verbose).fit_score(X, y)
                    states_checked[tuple(sorted(new_params.items()))] = (new_score, new_std)
                grid_scores.append((total_iter, T, new_score, new_std, new_params))

                # Keep track of the best score and best params
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params

                if self.__verbose:
                    print("%s T: %s, score: %s, std: %s, params: %s"
                          % (str(total_iter), '{:.5f}'.format(T),
                             '{:.3f}'.format(new_score), '{:.3f}'.format(new_std),
                             str(new_params)))

                # Decide whether to keep old params or move to new params
                a = accept_prob(old_score, new_score, T)
                a_rand = random.random()
                if a > a_rand:
                    old_params = new_params
                    old_score = new_score

                t_elapsed = dt(time_at_start, time.time())
                iter_ += 1
            T *= alpha

        if self.__refit:
            # Refit a estimator with the best params
            self.__est.set_params(**best_params)
            self.__est.fit(X, y)
            self.best_estimator_ = self.__est

        self.runtime_ = t_elapsed
        self.grid_scores_ = grid_scores
        self.best_score_ = best_score
        self.best_params_ = best_params

class MultiProcCvFolds(object):
    def __init__(self, clf, metric, cv, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        try:
            cv = int(cv)
        except:
            cv = cv

        self.clf = clf
        self.metric = get_scorer(metric)
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

    def fit_score(self, X, Y):
        if isinstance(self.cv, int):
            n_folds = self.cv
            self.cv = cross_validation.KFold(len(Y), n_folds=n_folds)

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )(
            delayed(_fit_and_score)(clone(self.clf), X, Y, self.metric,
                                    train, test, self.verbose, {},
                                    {}, return_parameters=False,
                                    error_score='raise')
                for train, test in self.cv)

        # Out is a list of triplet: score, estimator, n_test_samples
        scores = zip(*out)[0]
        return np.mean(scores), np.std(scores)


class CVFolds(object):
    def __init__(self, estimator, scorer, cv=3):
        try:
            cv = int(cv)
        except:
            cv = cv
        self.__est = estimator
        self.__cv = cv
        self.__scorer = get_scorer(scorer)

    def fit_score(self, X, y):
        if isinstance(self.__cv, int):
            cross_valid = cross_validation.KFold(len(y), n_folds=self.__cv)
        else:
            cross_valid = self.__cv
        scorer = self.__scorer
        scores = []
        for train_index, test_index in cross_valid:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            est = clone(self.__est)
            est.fit(X_train, y_train)
            k_score = scorer(est, X_test, y_test)
            scores.append(k_score)
        return (np.mean(scores), np.std(scores))
