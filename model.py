import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pickle


# Create a custom VotingModel class, API similar to sklearn
class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Average the predictions of all the estimators
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        # Average the predicted probabilities of all the estimators
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


class Model:
    def __init__(self, model_params, num_folds=5):
        self.model = None
        self.model_params = model_params
        self.num_folds = num_folds

        self.fitted_models = []
        self.cv_scores = []

        self.cv = StratifiedGroupKFold(n_splits=num_folds, shuffle=False)

    def fit(self, X=None, y=None, weeks=None):
        print("--- Training the model ...")
        # Cross-validation loop to train the model
        for idx_train, idx_valid in self.cv.split(X, y, groups=weeks):
            print("-" * 50)
            print("Fold {} ...".format(len(self.fitted_models) + 1))
            print("Train week range: ", (weeks.iloc[idx_train].min(), weeks.iloc[idx_train].max()))

            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
            print("Train shape: ", X_train.shape)
            print("Valid shape: ", X_valid.shape)
            print("Fitting the model ...")
            model = lgb.LGBMClassifier(**self.model_params)

            # Initialize an empty dictionary to store evaluation results
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)]
            )

            self.fitted_models.append(model)
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            auc_score = roc_auc_score(y_valid, y_pred_valid)
            self.cv_scores.append(auc_score)
            print("Fold AUC score: ", auc_score)
            lgb.plot_metric(model)

        print("-" * 50)
        print("Creating the VotingModel ...")
        print("Results:")
        self.model = VotingModel(self.fitted_models)
        print("CV AUC scores: ", self.cv_scores)
        print("Average CV AUC score: ", sum(self.cv_scores) / len(self.cv_scores))
        print("-" * 50)
        print("Training completed.")
        return self.model

    def predict(self, X):
        print("--- Making predictions ...")
        pred = pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
        print("Predictions completed.")
        return pred

    def save_model(self, path):
        print("--- Saving model ...")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print("Model saved. Check the file: ", path)

    @staticmethod
    def load_model(path):
        print("--- Loading model ...")
        print("Model path: ", path)
        with open(path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded.")
        return model
