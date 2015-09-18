from itertools import product
import random
from sklearn.metrics import (make_scorer, r2_score, mean_squared_error,
                             mean_absolute_error, median_absolute_error,
                             accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, precision_score,
                             recall_score, log_loss, adjusted_rand_score,
                             silhouette_score)
from sklearn.cross_validation import train_test_split
from sklearn.base import clone
from copy import copy
import numpy as np
import json
import sys


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
                 T=10, T_min=0.0001, alpha=0.9, max_iter=100, n_trans=10,
                 verbose=False, refit=True):
        try:
            score_function = SCORERS[scoring]._score_func
        except:
            sys.exit("\nERROR: %s not a valid score function."
            "\nUse one of the following: %s"
            % (scoring, ', '.join(SCORERS.keys())))

        # The total number of iterations that can be performed
        n_possible_iters = n_trans*((np.log(T_min)-np.log(T))/np.log(alpha))
        # If fractional max_iter provided, convert to a number
        if max_iter <= 1:
            max_iter = int(max_iter*n_possible_iters)

        assert hasattr(classifier, 'fit'), "The provided classifer has no fit method."
        assert hasattr(classifier, 'predict'), "The provided classifier has no predict method"
        assert max_iter <= n_possible_iters,  "WARNING: The value for max_iter=%s is greater than the number of possible iterations for the specified cooling schedule: %s\n" % (str(max_iter), str(int(n_possible_iters)))

        self._T = T
        self._T_min = T_min
        self._alpha = alpha
        self._max_iter = max_iter
        self._score = score_function
        self._grid = param_grid
        self._clf = classifier
        self._verbose = verbose
        self._n_trans = n_trans

        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.grid_scores_ = None

    def fit(self, X, y):
        # Set up  the initial params
        T = self._T
        T_min = self._T_min
        alpha = self._alpha
        score_func = self._score
        max_iter = self._max_iter
        n_trans = self._n_trans
        grid = self._grid

        # Split the data into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # List of all possible parameter combinations
        possible_params = list(product(*grid.values()))

        # Computes the acceptance probability as a function of T
        accept_prob = lambda old, new, T: np.exp((new-old)/T)

        # Compute the initial score based off randomyl selected param
        old_clf = clone(self._clf)
        old_params = dict(zip(grid.keys(), random.choice(possible_params)))
        old_clf.set_params(**old_params)
        old_clf.fit(X_train, y_train)
        y_pred_old = old_clf.predict(X_test)
        old_score = score_func(y_test, y_pred_old)

        # Variables to hold the best params
        best_score = old_score
        best_params = old_params

        # Hash table to store states checked and the score for that model
        states_checked = {}
        states_checked[json.dumps(old_params)] = old_score
        total_iterations = 1
        grid_scores = [(1, T, old_score, old_params)]

        while T > T_min and total_iterations < max_iter:
            iter_ = 0
            while iter_ < n_trans:
                total_iterations += 1
                new_params = copy(old_params)
                # Select random parameter to change
                rand_key = random.choice(grid.keys())
                new_rand_key_val = random.choice([v for v in grid[rand_key]
                                                  if v != old_params[rand_key]])
                # Set randomly selected parameter to new randomly selected value
                new_params[rand_key] = new_rand_key_val
                # Look to see if the score has been computed for the given params


                try:
                    new_score = states_checked[json.dumps(new_params)]
                # If unseen train classifier on new params and store score
                except:
                    new_clf = self._clf.set_params(**new_params)
                    new_clf.fit(X_train, y_train)
                    y_pred_new = new_clf.predict(X_test)
                    new_score = score_func(y_test, y_pred_new)
                    # Add param combo to hash table of states checked
                    states_checked[json.dumps(new_params)] = new_score
                    # Update grid score list

                grid_scores.append((total_iterations, T, new_score, new_params))
                # Keep track of the best score and best params
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params

                # If verbose print Temp and params
                if self._verbose:
                    print("%s T: %s, score: %s, params: %s"
                          % (str(total_iterations), '{:.5f}'.format(T),
                             '{:.3f}'.format(new_score),
                             json.dumps(new_params)))

                # Decide whether to keep old params or move to new params
                a = accept_prob(old_score, new_score, T)
                a_rand = random.random()
                if a > a_rand:
                    old_params = new_params
                    old_score = new_score
                iter_ += 1
            # Decrease the temperature
            T = T*alpha

        # Refit a classifier with the best params
        self._clf.set_params(**best_params)
        self._clf.fit(X_train, y_train)

        self.best_estimator_ = self._clf
        self.grid_scores_ = grid_scores
        self.best_score_ = best_score
        self.best_params_ = best_params

        return None
