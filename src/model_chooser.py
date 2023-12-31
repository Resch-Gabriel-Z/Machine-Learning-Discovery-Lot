import streamlit as st
from algorithm import create_model


def model_chooser(algorithm):
    # ___________ Logistic Regression ___________
    if algorithm == "Logistic Regression":
        st.sidebar.markdown("Choose parameters")
        C = st.sidebar.slider("C", 0.01, 10.0)
        penalty = st.sidebar.selectbox("Penalty", ["l1", "l2", "none"])
        max_iter = st.sidebar.slider("Max iterations", 100, 10000)
        model = create_model(
            "logistic_regression",
            C=C,
            penalty=penalty,
            solver="saga",
            max_iter=max_iter,
        )

    # ___________ SVM ___________
    elif algorithm == "SVM":
        st.sidebar.markdown("Choose parameters")
        C = st.sidebar.slider("C", 0.01, 10.0)
        kernel = st.sidebar.selectbox(
            "Kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"]
        )
        degree = st.sidebar.slider("Degree", 1, 10)
        gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        model = create_model("svm", C=C, kernel=kernel, degree=degree, gamma=gamma)

    # ___________ Random Forest ___________
    elif algorithm == "Random Forest":
        st.sidebar.markdown("Choose parameters")
        n_estimators = st.sidebar.slider("Number of estimators", 100, 500)
        max_depth = st.sidebar.slider("Max depth", 1, 10)
        min_samples_split = st.sidebar.slider("Min samples split", 1, 10)
        min_samples_leaf = st.sidebar.slider("Min samples leaf", 1, 10)
        bootstrap = st.sidebar.selectbox("Bootstrap", ["True", "False"])
        bootstrap = 1 if bootstrap == "True" else 0
        model = create_model(
            "random_forest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
        )

    # ___________ KNN ___________
    elif algorithm == "KNN":
        st.sidebar.markdown("Choose parameters")
        n_neighbors = st.sidebar.slider("Number of neighbors", 1, 10)
        weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan", "minkowski"])
        model = create_model(
            "knn", n_neighbors=n_neighbors, weights=weights, metric=metric
        )

    # ___________ Decision Tree ___________
    elif algorithm == "Decision Tree":
        st.sidebar.markdown("Choose parameters")
        max_depth = st.sidebar.slider("Max depth", 1, 10)
        min_samples_split = st.sidebar.slider("Min samples split", 1, 10)
        min_samples_leaf = st.sidebar.slider("Min samples leaf", 1, 10)
        model = create_model(
            "decision_tree",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    # ___________ Linear Regression ___________
    elif algorithm == "Linear Regression":
        model = create_model("linear_regression")

    # ___________ SVR ___________
    elif algorithm == "SVR":
        st.sidebar.markdown("Choose parameters")
        C = st.sidebar.slider("C", 0.01, 10.0)
        kernel = st.sidebar.selectbox(
            "Kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"]
        )
        degree = st.sidebar.slider("Degree", 1, 10)
        gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        model = create_model("svr", C=C, kernel=kernel, degree=degree, gamma=gamma)

    # ___________ Random Forest Regression ___________
    elif algorithm == "Random Forest Regression":
        st.sidebar.markdown("Choose parameters")
        n_estimators = st.sidebar.slider("Number of estimators", 100, 500)
        max_depth = st.sidebar.slider("Max depth", 1, 10)
        min_samples_split = st.sidebar.slider("Min samples split", 1, 10)
        min_samples_leaf = st.sidebar.slider("Min samples leaf", 1, 10)
        bootstrap = st.sidebar.selectbox("Bootstrap", ["True", "False"])
        bootstrap = 1 if bootstrap == "True" else 0
        model = create_model(
            "random_forest_regression",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
        )

    # ___________ KNN Regression ___________
    elif algorithm == "KNN Regression":
        st.sidebar.markdown("Choose parameters")
        n_neighbors = st.sidebar.slider("Number of neighbors", 1, 10)
        weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan", "minkowski"])
        model = create_model(
            "knn_regression", n_neighbors=n_neighbors, weights=weights, metric=metric
        )

    # ___________ Decision Tree Regression ___________
    elif algorithm == "Decision Tree Regression":
        st.sidebar.markdown("Choose parameters")
        max_depth = st.sidebar.slider("Max depth", 1, 10)
        min_samples_split = st.sidebar.slider("Min samples split", 1, 10)
        min_samples_leaf = st.sidebar.slider("Min samples leaf", 1, 10)
        model = create_model(
            "decision_tree_regression",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    # ___________ Lasso ___________
    elif algorithm == "Lasso":
        st.sidebar.markdown("Choose parameters")
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0)
        model = create_model("lasso", alpha=alpha)

    # ___________ Ridge ___________
    elif algorithm == "Ridge":
        st.sidebar.markdown("Choose parameters")
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0)
        model = create_model("ridge", alpha=alpha)

    # ___________ Kernel Ridge ___________
    elif algorithm == "Kernel Ridge":
        st.sidebar.markdown("Choose parameters")
        alpha = st.sidebar.slider("Alpha", 0.01, 10.0)
        kernel = st.sidebar.selectbox(
            "Kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"]
        )
        degree = st.sidebar.slider("Degree", 1, 10)
        model = create_model("kernel_ridge", alpha=alpha, kernel=kernel, degree=degree)

    # ___________ KMeans ___________
    elif algorithm == "KMeans":
        st.sidebar.markdown("Choose parameters")
        n_clusters = st.sidebar.slider("Number of clusters", 1, 10)
        init = st.sidebar.selectbox("Init", ["k-means++", "random"])
        max_iter = st.sidebar.slider("Max iterations", 100, 1000)
        model = create_model(
            "kmeans", n_clusters=n_clusters, init=init, max_iter=max_iter
        )

    # ___________ DBSCAN ___________
    elif algorithm == "DBSCAN":
        st.sidebar.markdown("Choose parameters")
        eps = st.sidebar.slider("Epsilon", 0.01, 10.0)
        min_samples = st.sidebar.slider("Min samples", 1, 10)
        metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan", "minkowski"])
        model = create_model("dbscan", eps=eps, min_samples=min_samples, metric=metric)

    # ___________ Agglomerative Clustering ___________
    elif algorithm == "Agglomerative Clustering":
        st.sidebar.markdown("Choose parameters")
        n_clusters = st.sidebar.slider("Number of clusters", 1, 10)
        linkage = st.sidebar.selectbox(
            "Linkage", ["ward", "complete", "average", "single"]
        )
        model = create_model(
            "agglomerative_clustering", n_clusters=n_clusters, linkage=linkage
        )

    # ___________ PCA ___________
    elif algorithm == "PCA":
        st.sidebar.markdown("Choose parameters")
        n_components = st.sidebar.slider("Number of components", 1, 10)
        model = create_model("pca", n_components=n_components)

    return model
