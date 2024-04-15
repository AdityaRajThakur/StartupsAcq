#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin
class BinaryClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

class MulticlassClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.predict(X).reshape(-1, 1)

    def predict(self, X):
        return self.model.predict(X)

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'StartupAacquisition.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
