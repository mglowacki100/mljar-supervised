import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML

from supervised.algorithms.random_forest import additional

additional["max_steps"] = 1
additional["trees_in_step"] = 1

from supervised.algorithms.xgboost import additional

additional["max_rounds"] = 1


class AutoMLExplainLevelsTest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_explain_default(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Random Forest"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
            start_random_models=1,
        )

        X, y = datasets.make_classification(
            n_samples=100,
            n_features=5,
            n_informative=4,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(
            os.path.join(self.automl_dir, "1_Default_RandomForest")
        )

        produced = any("importance.csv" in f and "shap" not in f for f in result_files)
        self.assertTrue(produced)
        produced = any("importance.csv" in f and "shap" in f for f in result_files)
        self.assertTrue(produced)
        produced = any("dependence.png" in f for f in result_files)
        self.assertTrue(produced)
        produced = any("decisions.png" in f for f in result_files)
        self.assertTrue(produced)

    def test_no_explain_linear(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Linear"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
            explain_level=0,
            start_random_models=1,
        )

        X, y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(os.path.join(self.automl_dir, "1_Linear"))

        produced = any("importance.csv" in f and "shap" not in f for f in result_files)
        self.assertFalse(produced)
        produced = any("importance.csv" in f and "shap" in f for f in result_files)
        self.assertFalse(produced)
        produced = any("dependence.png" in f for f in result_files)
        self.assertFalse(produced)
        produced = any("decisions.png" in f for f in result_files)
        self.assertFalse(produced)
        produced = any("coefs.csv" in f for f in result_files)
        self.assertFalse(produced)

    def test_explain_just_permutation_importance(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
            explain_level=1,
            start_random_models=1,
        )

        X, y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(os.path.join(self.automl_dir, "1_Default_Xgboost"))

        produced = any("importance.csv" in f and "shap" not in f for f in result_files)
        self.assertTrue(produced)
        produced = any("importance.csv" in f and "shap" in f for f in result_files)
        self.assertFalse(produced)
        produced = any("dependence.png" in f for f in result_files)
        self.assertFalse(produced)
        produced = any("decisions.png" in f for f in result_files)
        self.assertFalse(produced)

    def test_build_decision_tree(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Decision Tree"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
            explain_level=2,
            start_random_models=1,
        )

        X, y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(os.path.join(self.automl_dir, "1_DecisionTree"))

        produced = any("tree.svg" in f for f in result_files)
        self.assertTrue(produced)
        produced = any("importance.csv" in f and "shap" not in f for f in result_files)
        self.assertTrue(produced)
        produced = any("importance.csv" in f and "shap" in f for f in result_files)
        self.assertTrue(produced)
        produced = any("dependence.png" in f for f in result_files)
        self.assertTrue(produced)
        produced = any("decisions.png" in f for f in result_files)
        self.assertTrue(produced)
