Simulated Annealing
===
This module provides a hyperparameter optimization using simulated annealing.  It has a SciKit-Learn-style API and uses multiprocessing for the fitting and scoring of the cross validation folds.  The benefit of using Simulated Annealing over an exhaustive grid search is that Simulated Annealing is a heuristic search algorithm that is immune to getting stuck in local minima or maxima.  

Installation
===
Installation can be performed using pip:

```bash
pip install simulated_annealing
```

Description of Simulated Annealing Algorithm
===
- Start with some initial `T` and `alpha`
- Generate and score a random solution (`score_old`)
- Generate and score a solution with "neighboring" hyperparameters (`score_new`)
- Compare `score_old` and `score_new`:
    - If `score_new` > `score_old`: move to neighboring solution
    - If `score_new` < `score_old`: maybe move to neighboring solution
- Decrease T: `T*=alpha`
- Repeat the above steps until one of the stopping conditions met:
    - `T < T_min`
    - `n_iterations > max_iterations`
    - `total_runtime > max_runtime`
- Return the score and hyperparameters of the best solution

The decision to move to a new solution from an old solution is probabilistic and temperature dependent.  Specifically, the comparison between the solutions is performed by computing `a = exp((score_new - score_old)/T)`.  The value of `a` is then compared to a randomly generated number in [0,1].  If `a` is greater than the randomly generated number, the algorithm moves to the hyperparameters of the neighboring solution.  This means that while `T` is large, *almost all new solutions are preferred regardless of their score*.  As `T` decreases, the likelihood of moving to hyperparameters resulting in a poor solution decreases.  

Dependencies
===
Simulated Annealing was written on a OS X 10.10.5 and using Python 2.7.10.  External library dependencies include:
- NumPy 1.6.1+
- SciKit-Learn 0.16+

Important Info Regarding Scoring
===
This implementation of Simulated Annealing can use any of the built-in SciKit Learn scoring metrics or any other scoring function/object with the signature `score(estimator, X, y)`.  It is important to note that during the annealing process, the algorithm will always be ***maximizing*** the score.  So if you intend on finding optimal hyperparameters for a regression algorithm, it is important to multiply your scoring metric by -1.  

Example
===
```python
from sklearn.cross_validation import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from simulated_annealing.optimize import SimulatedAnneal

iris = datasets.load_iris()
X = iris.data
y = iris.target
svc_params = {'C':np.logspace(-8, 10, 19, base=2),
              'fit_intercept':[True, False]
             }
clf = svm.LinearSVC()
sa = SimulatedAnneal(clf, svc_params, T=10.0, T_min=0.001, alpha=0.75,
                         verbose=True, max_iter=0.25, n_trans=5, max_runtime=300,
                         cv=3, scoring='f1_macro', refit=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
sa.fit(X_train, y_train)
print sa.best_score_, sa.best_params_
optimized_clf = sa.best_estimator_
y_test_pred = optimized_clf.predict(X_test)
print classification_report(y_test, y_test_pred)
```
