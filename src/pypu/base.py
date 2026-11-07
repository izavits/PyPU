"""Base class shared by every PU classifier in PyPU."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted, validate_data

from pypu.utils.validation import check_pu_labels

# Probabilities are clipped to [_EPS, 1 - _EPS] before taking log-odds.
_EPS = 1e-12


class BasePUClassifier(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for positive-unlabeled (PU) classifiers.

    Every PU classifier in PyPU follows the scikit-learn estimator API, so it works with
    :func:`~sklearn.base.clone`, :class:`~sklearn.pipeline.Pipeline` and the model selection
    tools:

    - ``fit(X, s)`` learns from features ``X`` and PU labels ``s``, where ``s[i] == 1``
      marks a labeled positive and ``s[i] == 0`` an unlabeled sample (positive or negative).
    - ``predict_proba(X)`` returns an array of shape ``(n_samples, 2)`` whose second column
      is the estimated probability of the positive class.
    - ``decision_function(X)`` returns one score per sample: higher means more likely
      positive. By default it is the log-odds of the positive-class probability, so ``0``
      corresponds to a probability of ``0.5``.
    - ``predict(X)`` returns hard labels in ``{0, 1}``: a sample is predicted positive when
      its positive-class probability is above :attr:`threshold_`. A probability equal to the
      threshold counts as negative, as with scikit-learn's ``argmax`` of ``predict_proba``.

    ``score(X, y)`` (from :class:`~sklearn.base.ClassifierMixin`) computes the accuracy
    against *true* labels ``y``, not against PU labels.

    Subclasses implement :meth:`_fit` and :meth:`_predict_proba`, and may override
    :meth:`_decision_function` (to return raw scores) and :meth:`_compute_threshold` (to
    change the decision threshold). Input validation and the fitted attributes listed below
    are handled by this class.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when ``X`` has feature
        names that are all strings, e.g. a pandas DataFrame.
    threshold_ : float
        Positive-class probability threshold used by :meth:`predict`.
    """

    def fit(self, X, s):
        """Fit the model to PU data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        s : array-like of shape (n_samples,)
            PU labels: ``1`` for a labeled positive, ``0`` for an unlabeled sample.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        X = self._validate_X(X, reset=True)
        s = check_pu_labels(s)
        check_consistent_length(X, s)
        self._fit(X, s)
        self.classes_ = np.array([0, 1])
        self.threshold_ = float(self._compute_threshold(X, s))
        return self

    def predict_proba(self, X):
        """Estimate class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probability of the negative class (column 0) and of the positive class (column 1).
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        proba = np.asarray(self._predict_proba(X), dtype=np.float64).reshape(-1)
        return np.column_stack([1.0 - proba, proba])

    def decision_function(self, X):
        """Score samples; higher means more likely positive.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            One score per sample.
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return np.asarray(self._decision_function(X), dtype=np.float64).reshape(-1)

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            ``1`` for samples whose positive-class probability is above
            :attr:`threshold_`, ``0`` otherwise.
        """
        proba = self.predict_proba(X)[:, 1]
        return self.classes_[(proba > self.threshold_).astype(np.int64)]

    # ------------------------------------------------------------------ hooks for subclasses

    @abstractmethod
    def _fit(self, X, s):
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Validated training samples (float64 or float32).
        s : ndarray of shape (n_samples,)
            Validated PU labels (int64, 1 = labeled positive, 0 = unlabeled).
        """

    @abstractmethod
    def _predict_proba(self, X):
        """Return the positive-class probability of each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Validated samples.

        Returns
        -------
        proba : ndarray of shape (n_samples,)
            Positive-class probabilities, in ``[0, 1]``.
        """

    def _decision_function(self, X):
        """Return a score per sample; by default the log-odds of the positive class."""
        proba = np.clip(self._predict_proba(X), _EPS, 1.0 - _EPS)
        return logit(proba)

    def _compute_threshold(self, X, s):
        """Return the probability threshold used by :meth:`predict`; ``0.5`` by default.

        Called at the end of :meth:`fit`, after :meth:`_fit`, with the validated training data.
        """
        return 0.5

    def _validate_X(self, X, *, reset):
        """Validate ``X``; ``reset=True`` in ``fit``, ``False`` when predicting."""
        return validate_data(self, X, reset=reset, dtype=[np.float64, np.float32])

    # ------------------------------------------------------------------ scikit-learn protocol

    def __sklearn_is_fitted__(self):
        return hasattr(self, "threshold_")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
