from itertools import product
import random
from sklearn.metrics import (make_scorer, r2_score, mean_squared_error,
                             mean_absolute_error, median_absolute_error,
                             accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, precision_score,
                             recall_score, log_loss, adjusted_rand_score,
                             silhouette_score)
from sklearn import cross_validation
from sklearn.base import clone
from copy import copy
import numpy as np
import json
import sys
import time


# Standard regression scores
r2_scorer = make_scorer(r2_score)
mean_squared_error_scorer = make_scorer(mean_squared_error,
                                        greater_is_better=False)
mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                         greater_is_better=False)
median_absolute_error_scorer = make_scorer(median_absolute_error,
                                           greater_is_better=False)

# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)

# Score function for probabilistic classification
log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                              needs_proba=True)

# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
silhouette_scorer = make_scorer(silhouette_score)
SCORERS = dict(r2=r2_scorer,
               median_absolute_error=median_absolute_error_scorer,
               mean_absolute_error=mean_absolute_error_scorer,
               mean_squared_error=mean_squared_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               average_precision=average_precision_scorer,
               log_loss=log_loss_scorer,
               adjusted_rand_score=adjusted_rand_scorer,
               silhouette_scorer=silhouette_scorer)

for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score)]:
    SCORERS[name] = make_scorer(metric)
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = make_scorer(metric, average=average)


class SimulatedAnneal(object):
    def __init__(self, classifier, param_grid, scoring='f1_macro',
                 T=10, T_min=0.0001, alpha=0.75, n_trans=10,
                 max_iter=300, max_runtime=300, cv=3,
                 verbose=False, refit=True):

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

        # If scoring a string, get scorer from SCORER
        if isinstance(scoring, basestring):
            try:
                score_function = SCORERS[scoring]._score_func
            except:
                sys.exit("\nERROR: %s not a valid score function."
                "\nUse one of the following: %s"
                % (scoring, ', '.join(SCORERS.keys())))
        # If scoring not a string, assume the user has provided a scoring function
        else:
            score_function = scoring

        # The total number of iterations that can be performed
        n_possible_iters = n_trans*((np.log(T_min)-np.log(T))/np.log(alpha))
        # If fractional max_iter provided, convert to a number
        if max_iter <= 1:
            max_iter = int(max_iter*n_possible_iters)

        assert hasattr(classifier, 'fit'), "The provided classifer has no fit method."
        assert hasattr(classifier, 'predict'), "The provided classifier has no predict method"
        assert max_iter <= n_possible_iters,  "WARNING: The value for max_iter=%s is greater than the number of possible iterations for the specified cooling schedule: %s\n" % (str(max_iter), str(int(n_possible_iters)))

        # Hidden attributes
        self.__T = T
        self.__T_min = T_min
        self.__alpha = alpha
        self.__max_iter = max_iter
        self.__grid = param_grid
        self.__clf = classifier
        self.__verbose = verbose
        self.__n_trans = n_trans
        self.__max_runtime = max_runtime
        self.__cv = cv
        self.__refit = refit

        # Exposed attributes
        self._scorer = score_function
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

        # List of all possible parameter combinations
        possible_params = list(product(*grid.values()))

        # Computes the acceptance probability as a function of T
        accept_prob = lambda old, new, T: np.exp((new-old)/T)

        # Compute the initial score based off randomly selected param
        old_clf = clone(self.__clf)
        old_params = dict(zip(grid.keys(), random.choice(possible_params)))
        old_clf.set_params(**old_params)
        old_score, old_std = CVFolds(old_clf, scorer=score_func, cv=cv).fit_score(X, y)

        # Variables to hold the best params
        best_score = old_score
        best_params = old_params

        # Hash table to store states checked and the score for that model
        states_checked = {}
        states_checked[json.dumps(old_params)] = (old_score, old_std)
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
                new_params = copy(old_params)
                # Select random parameter to change
                rand_key = random.choice(grid.keys())
                new_rand_key_val = random.choice([v for v in grid[rand_key]
                                                  if v != old_params[rand_key]])
                # Set randomly selected parameter to new randomly selected value
                new_params[rand_key] = new_rand_key_val
                # Look to see if the score has been computed for the given params
                try:
                    new_score, new_std = states_checked[json.dumps(new_params)]
                # If unseen train classifier on new params and store score
                except:
                    new_clf = self.__clf.set_params(**new_params)
                    new_score, new_std = CVFolds(new_clf, scorer=score_func, cv=cv).fit_score(X, y)
                    # Add param combo to hash table of states checked
                    states_checked[json.dumps(new_params)] = (new_score, new_std)

                grid_scores.append((total_iter, T, new_score, new_std, new_params))
                # Keep track of the best score and best params
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params

                # If verbose print Temp and params
                if self.__verbose:
                    print("%s T: %s, score: %s, std: %s, params: %s"
                          % (str(total_iter), '{:.5f}'.format(T),
                             '{:.3f}'.format(new_score), '{:.3f}'.format(new_std),
                             json.dumps(new_params)))

                # Decide whether to keep old params or move to new params
                a = accept_prob(old_score, new_score, T)
                a_rand = random.random()
                if a > a_rand:
                    old_params = new_params
                    old_score = new_score
                t_elapsed = dt(time_at_start, time.time())
                iter_ += 1
            # Decrease the temperature
            T = T*alpha

        if self.__refit:
            # Refit a classifier with the best params
            self.__clf.set_params(**best_params)
            self.__clf.fit(X, y)
            self.best_estimator_ = self.__clf

        self.runtime_ = t_elapsed
        self.grid_scores_ = grid_scores
        self.best_score_ = best_score
        self.best_params_ = best_params
        return None


class CVFolds(object):
    def __init__(self, classifier, scorer, cv=3):
        self.__clf = classifier
        self.__cv = cv
        self.__scorer = scorer

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
            clf = clone(self.__clf)
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            k_score = scorer(y_test, y_test_pred)
            scores.append(k_score)
        return (np.mean(scores), np.std(scores))
