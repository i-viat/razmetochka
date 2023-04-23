import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import numpy as np


class Advisor:
    def __init__(
            self,
            random_state: int = 12345,
            model: object = False,
            vectorizer: object = None
    ):
        self.random_state = random_state
        self.model = False
        self.vectorizer = False

        # todo: добавить кустомный скорер
        self.scorer = False

        if model:
            self.model = CalibratedClassifierCV(model)
        if vectorizer:
            self.vectorizer = vectorizer

    def save(self, path_to_model: str = './model.bin'):
        try:
            with open(path_to_model, 'wb') as f:
                pickle.dump(self, f)
        except IOError:
            print("Can't save model to file")

    def load(self, path_to_model: str = './model.bin'):
        try:
            with open(path_to_model, 'rb') as f:
                self = pickle.load(f)
        except IOError:
            print("File with model not accessible")

    def fit_vectorizer(self, corpus):
        assert self.vectorizer, 'Add vectorizer to advisor'
        self.vectorizer.fit(corpus)

    def transform_vectorizer(self, corpus):
        assert self.vectorizer, 'Add vectorizer to advisor'
        return self.vectorizer.transform(corpus)

    def fit_model(self, X, y):
        assert self.model, 'Add ML model to advisor'
        X_train = self.vectorizer.transform(X)
        self.model.fit(X_train, y)

    def classification_report(self, X, y):

        return classification_report(y, self.predict(X), output_dict=True)

    def predict(self, X):
        assert self.model, 'Add ML model to advisor'

        return self.model.predict(self.transform_vectorizer(X))

    def predict_proba(self, X):
        assert self.model, 'Add ML model to advisor'
        X_vctrs = self.transform_vectorizer(X)
        X = pd.DataFrame(X)

        X['proba'] = np.max(self.model.predict_proba(X_vctrs), axis=1)

        return X
