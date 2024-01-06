# sklearn models: classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# sklearn models: regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

# sklearn models: clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# sklearn models: dimensionality reduction
from sklearn.decomposition import PCA


def create_model(model_type, **kwargs):
    if model_type == "logistic_regression":
        # kwargs: C, penalty, solver, max_iter
        return LogisticRegression(**kwargs)
    elif model_type == "svm":
        # kwargs: C, kernel, degree, gamma
        return SVC(**kwargs)
    elif model_type == "random_forest":
        # kwargs: n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap
        return RandomForestClassifier(**kwargs)
    elif model_type == "knn":
        # kwargs: n_neighbors, weights, metric
        return KNeighborsClassifier(**kwargs)
    elif model_type == "decision_tree":
        # kwargs: max_depth, min_samples_split, min_samples_leaf
        return DecisionTreeClassifier(**kwargs)
    elif model_type == "linear_regression":
        return LinearRegression()
    elif model_type == "svr":
        # kwargs: C, kernel, degree, gamma
        return SVR(**kwargs)
    elif model_type == "random_forest_regression":
        # kwargs: n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap
        return RandomForestRegressor(**kwargs)
    elif model_type == "knn_regression":
        # kwargs: n_neighbors, weights, metric
        return KNeighborsRegressor(**kwargs)
    elif model_type == "decision_tree_regression":
        # kwargs: max_depth, min_samples_split, min_samples_leaf
        return DecisionTreeRegressor(**kwargs)
    elif model_type == "lasso":
        # kwargs: alpha
        return Lasso(**kwargs)
    elif model_type == "ridge":
        # kwargs: alpha
        return Ridge(**kwargs)
    elif model_type == "kernel_ridge":
        # kwargs: alpha, kernel
        return KernelRidge(**kwargs)
    elif model_type == "kmeans":
        # kwargs: n_clusters, init, max_iter
        return KMeans(**kwargs, n_init="auto")
    elif model_type == "dbscan":
        # kwargs: eps, min_samples, metric
        return DBSCAN(**kwargs)
    elif model_type == "agglomerative_clustering":
        # kwargs: n_clusters, linkage
        return AgglomerativeClustering(**kwargs)
    elif model_type == "pca":
        # kwargs: n_components
        return PCA(**kwargs)
    else:
        raise Exception("Invalid model type: {}".format(model_type))
