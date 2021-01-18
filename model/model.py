"""
Binary classification model.
"""

from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from model.constants import (
    DEPARTMENT_NAME,
    GENDER_F,
    GENDER_M,
    GENDER_SHORT,
    PRODUCE_DEP,
    STATUS,
    STORE_MANAGEMENT_DEP,
    TERMINATED_DUMMY,
)

Dummy_columns = namedtuple("Dummy_columns", ["column", "value", "new_column"])

# department_name values
STORE_MANAGEMENT = "Store Management"
PRODUCE = "Produce"
# gender_short values
FEMALE = "F"
MALE = "M"
# STATUS - label - values
STATUS_TERMINATED = "TERMINATED"


# create named tuples for dummy columns
store_management_dummy = Dummy_columns(
    DEPARTMENT_NAME, STORE_MANAGEMENT, STORE_MANAGEMENT_DEP
)
produce_dummy = Dummy_columns(DEPARTMENT_NAME, PRODUCE, PRODUCE_DEP)
female_dummy = Dummy_columns(GENDER_SHORT, FEMALE, GENDER_F)
male_dummy = Dummy_columns(GENDER_SHORT, MALE, GENDER_M)
label_dummy = Dummy_columns(STATUS, STATUS_TERMINATED, TERMINATED_DUMMY)

dummy_columns_list = [
    store_management_dummy,
    produce_dummy,
    female_dummy,
    male_dummy,
    label_dummy,
]


def prepare_data_columns(
    input_data: pd.DataFrame, dummy_columns_list: List = dummy_columns_list
) -> pd.DataFrame:
    """
    Create dummy columns for Store Management and Produce department,
    and for gender. Keep only numerical columns.

    :param input_data: input dataset
    :param dummy_columns: named tuple of columns to convert to dummy ones
    :return: transformed dataset
    """
    data = input_data.copy(deep=True)

    # create encoding columns
    for dummy_column in dummy_columns_list:
        print(
            f"Creating feature column {dummy_column.new_column}"
            f"for {dummy_column.column} with value {dummy_column.value}."
        )
        data[dummy_column.new_column] = (
            data[dummy_column.column] == dummy_column.value
        ).astype(int)

    # leave only numerical columns
    data_num = data.select_dtypes(["number"]).reset_index(drop=True)
    print(f"Dataset contains only numerical features: {data_num.columns}")

    return data_num


class RandomForestClassifierModel(object):
    """
    Random Forest Classifier class to split data, train model, make predictions
    and evaluate.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        label: str,
        n_estimators: int = 50,
        test_size: Optional[float] = 0.3,
        k_fold: Optional[int] = 10,
    ):
        """
        :param data: input data as pd.DataFrame
        :param label: label column
        :param n_estimators: number of trees in the forest
        :param test_size: share of data to form test set
        :param k_fold: number of folds in cross validation
        """
        self.data = data
        self.features = features
        self.label = label
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.k_fold = k_fold
        self.eval_metrics = None

    def split_train_valid_data(self) -> Tuple:
        """
        Split input data into train and valid sets.

        :return: tuple of train features and labels, and valid features and labels
        """
        #  fill the function
        pass

    def train_rfc_model(self, train_data: Tuple) -> RandomForestClassifier:
        """
        Train random forest classification (RFC) model.

        :param train_data: tuple of training features and labels, both np.array
        :return: trained RFC model
        """
        train_X, train_y = train_data
        train_y = train_y.ravel()
        # model definition
        forest_model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=1
        )

        return forest_model.fit(train_X, train_y)

    def predict_rfc_model(
        self, test_data: np.array, fit_model: RandomForestClassifier
    ) -> np.array:
        """
        Predict label based on test features.

        :param test_data: features to make predictions on
        :param fit_model: learned RFC model
        :return: predicted labels
        """

        return fit_model.predict(test_data)

    def evaluate_rfc_model(self, predictions: np.array, valid_labels: np.array) -> Dict:
        """
        Calculate accuracy, recall and precision as evaluation metrics.

        :param predictions: predicted labels by RFC model
        :param valid_labels: labels to validate
        :return: dict of evaluation metrics
        """
        accuracy = accuracy_score(valid_labels, predictions)
        recall = recall_score(valid_labels, predictions)
        precision = precision_score(valid_labels, predictions)

        return {"accuracy": accuracy, "recall": recall, "precision": precision}

    def cross_validate_rfc_model(
        self, scoring: List[str], k_folds: int
    ) -> Dict[str, Union[accuracy_score, precision_score, recall_score]]:

        """
        Get mean accuracy, recall and precision for K-fold cross validation.

        :param k_folds: number of subsamples
        :param scoring: list of metrics to evaluate
        :return: dict with means of given scoring metrics
        """
        #  fill the function
        pass

    def run(self):
        """
        Train RFC, predict labels and evaluate these predictions.
        """
        # split data
        print("Splitting data into train and validation datasets...")
        train_X, train_y, valid_X, valid_y = self.split_train_valid_data()
        # train RFC model
        print("Training RFC model...")
        rfr_model = self.train_rfc_model(train_data=(train_X, train_y))
        # predictions
        print("Making predictions with trained RFC model...")
        labels_pred = self.predict_rfc_model(test_data=valid_X, fit_model=rfr_model)
        # evaluation of given metrics
        print("Predictions evaluation...")
        self.eval_metrics = self.evaluate_rfc_model(labels_pred, valid_y)
        print(self.eval_metrics)

        # K-fold cross validation metrics
        print("Running cross validation...")
        cv_metrics = self.cross_validate_rfc_model(
            scoring=list(self.eval_metrics.keys()), k_folds=self.k_fold
        )
        print(cv_metrics)
        self.eval_metrics.update(cv_metrics)
