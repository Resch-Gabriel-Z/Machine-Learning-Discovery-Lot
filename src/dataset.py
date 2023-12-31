from sklearn.datasets import (
    make_blobs,
    make_moons,
    make_circles,
    make_classification,
    load_diabetes,
    load_iris,
    load_breast_cancer,
    make_regression,
)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(dataset_name, test_size, random_state=0):
    if dataset_name == "blobs":
        X, y = make_blobs(
            n_samples=1000, centers=2, n_features=2, random_state=random_state
        )
        dataframe = pd.DataFrame(X, columns=["x", "y"])
        dataframe["target"] = y
        # data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    elif dataset_name == "moons":
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=random_state)
        dataframe = pd.DataFrame(X, columns=["x", "y"])
        dataframe["target"] = y
        # data = dataframe
    elif dataset_name == "circles":
        X, y = make_circles(n_samples=1000, noise=0.1, random_state=random_state)
        dataframe = pd.DataFrame(X, columns=["x", "y"])
        dataframe["target"] = y
        # data = dataframe
    elif dataset_name == "regression":
        X, y = make_regression(
            n_samples=1000,
            n_features=1,
            noise=10,
            random_state=random_state,
            shuffle=True,
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        dataframe = pd.DataFrame(X, columns=["x"])
        dataframe["target"] = y
        # data = dataframe
    elif dataset_name == "classification":
        X, y = make_classification(
            n_samples=1000,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=random_state,
            n_clusters_per_class=1,
        )
        dataframe = pd.DataFrame(X, columns=["x", "y"])
        dataframe["target"] = y
        # data = dataframe
    elif dataset_name == "diabetes":
        data = load_diabetes()
        dataframe = pd.DataFrame(data.data, columns=data.feature_names)
        # attach target
        dataframe["target"] = data.target

        X_train, X_test, y_train, y_test = train_test_split(
            dataframe, data.target, test_size=test_size, random_state=0
        )
        return X_train, X_test, y_train, y_test, dataframe

    elif dataset_name == "iris":
        data = load_iris()
        dataframe = pd.DataFrame(data.data, columns=data.feature_names)
        # attach target
        dataframe["target"] = data.target

        X_train, X_test, y_train, y_test = train_test_split(
            dataframe, data.target, test_size=test_size, random_state=0
        )
        return X_train, X_test, y_train, y_test, dataframe

    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        dataframe = pd.DataFrame(data.data, columns=data.feature_names)
        # attach target
        dataframe["target"] = data.target

        X_train, X_test, y_train, y_test = train_test_split(
            dataframe, data.target, test_size=test_size, random_state=0
        )
        return X_train, X_test, y_train, y_test, dataframe

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    X_train, X_test, y_train, y_test = (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test),
    )
    return X_train, X_test, y_train, y_test, dataframe
