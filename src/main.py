import streamlit as st
import numpy as np
import pandas as pd
from dataset import load_data
from data_visualizer import plot_data
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from model_chooser import model_chooser
from sklearn.model_selection import (
    learning_curve,
)

from texts import (
    dataset_string,
    algorithm_string,
    other_string,
)
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
)
from sklearn import tree

# TODO: visualizer of training steps (curves)
# TODO: dendogram for agglomerative clustering
# TODO: clean up the code
# TODO: comments
# TODO: upload on github

score = None
conf = None
st.set_page_config(
    page_title="Streamlit Template",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ML_choices = ["Classification", "Regression", "Clustering", "Dimensionality Reduction"]
Algorithms_classification = [
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "KNN",
    "Decision Tree",
]
Algorithms_regression = [
    "Linear Regression",
    "SVR",
    "Random Forest Regression",
    "KNN Regression",
    "Decision Tree Regression",
    "Lasso",
    "Ridge",
    "Kernel Ridge",
]
Algorithms_clustering = ["KMeans", "DBSCAN", "Agglomerative Clustering"]
Algorithms_dimensionality_reduction = ["PCA"]

Dataset_choices_classification_clustering = [
    "Blobs",
    "Moons",
    "Circles",
    "Classification",
]
Dataset_choices_regression = ["Regression"]
Dataset_choices_dim_reduction = ["Iris", "Breast Cancer", "Diabetes"]


def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# ------------------- Sidebar -------------------
st.sidebar.title("Navigation")
st.sidebar.markdown("Choose ML type")
ML_type = st.sidebar.selectbox("ML type", ML_choices)

if ML_type == "Classification":
    Dataset_choices = Dataset_choices_classification_clustering
    Algorithm_choices = Algorithms_classification
elif ML_type == "Regression":
    Dataset_choices = Dataset_choices_regression
    Algorithm_choices = Algorithms_regression
elif ML_type == "Clustering":
    Dataset_choices = Dataset_choices_classification_clustering
    Algorithm_choices = Algorithms_clustering
elif ML_type == "Dimensionality Reduction":
    Dataset_choices = Dataset_choices_dim_reduction
    Algorithm_choices = Algorithms_dimensionality_reduction
st.sidebar.markdown("Choose dataset")
dataset = st.sidebar.selectbox("Dataset", Dataset_choices)
st.sidebar.markdown("Choose test size")
test_size = st.sidebar.slider("Test size", 0.1, 0.8, 0.2)
st.sidebar.markdown("Choose random state")
random_state = st.sidebar.slider("Random state", 0, 100, 0)
st.sidebar.markdown("Choose algorithm")
algorithm = st.sidebar.selectbox("Algorithm", Algorithm_choices)


model = model_chooser(algorithm)

# ___________ Dataset ___________
X_train, X_test, y_train, y_test, df = load_data(
    dataset.lower().replace(" ", "_"), test_size, random_state
)
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
# for the plot i need to come up with a better solution, this is just a workaround
# classifcation is good, so scatter plot is fine, same goes for clustering and dimensionality reduction
# regression is like 10d, so i need to come up with a better solution
# probably just a heatmap with the correlation matrix


# ------------------- Sidebar -------------------


# ------------------- Main -------------------

st.markdown('<p class="Page_title">Streamlit Template</h1>', unsafe_allow_html=True)

Home, About = st.tabs(["Home", "About"])

with Home:
    Home_col1, Home_col2 = st.columns([2, 1])
    with Home_col1:
        header_container = st.container()
        description_container = st.container()
        main_content = st.container()

        with header_container:
            st.markdown(
                f'<p class="Section_title">{other_string("header")}</p>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

        with description_container:
            st.markdown(
                f'<p class="Section_title">{algorithm_string(algorithm)}</p>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

        with main_content:
            main_content_col1 = st.container()

            main_content_col1.markdown(
                '<p class="Section_title">Main content</p>', unsafe_allow_html=True
            )
            main_content_col1.markdown("---")

            if main_content_col1.button("Run"):
                # main_content_col1.markdown(f'<p class="Result">Score: {score}</p>', unsafe_allow_html=True)
                if ML_type == "Classification":
                    model.fit(X_train, y_train)
                    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
                    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
                    xx, yy = np.meshgrid(
                        np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
                    )
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig = go.Figure(
                        data=[
                            go.Contour(
                                x=xx[0],
                                y=yy[:, 0],
                                z=Z,
                                colorscale="Viridis",
                                opacity=0.2,
                                contours=dict(start=0, end=1, size=1),
                            )
                        ]
                    )

                    fig.add_scatter(
                        x=X_test[:, 0],
                        y=X_test[:, 1],
                        mode="markers",
                        marker=dict(color=y_test),
                    )
                    main_content_col1.plotly_chart(fig, use_container_width=True)
                    conf = px.imshow(
                        confusion_matrix(
                            y_test, model.predict(X_test), normalize="true"
                        )
                        * 100,
                        text_auto=True,
                    )
                    conf.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                    )
                elif ML_type == "Clustering":
                    model.fit(X)
                    fig = px.scatter(
                        X,
                        x=0,
                        y=1,
                        color=model.labels_,
                        title="Clustering",
                        color_continuous_scale="Viridis",
                    )
                    main_content_col1.plotly_chart(fig, use_container_width=True)

                elif ML_type == "Regression":
                    model.fit(X_train, y_train)
                    if (
                        algorithm != "Kernel Ridge"
                        and algorithm != "Lasso"
                        and algorithm != "Ridge"
                        and algorithm != "Linear Regression"
                    ):
                        common_params = {
                            "X": X_train,
                            "y": y_train,
                            "cv": 5,
                            "train_sizes": np.linspace(0.1, 1.0, 5),
                        }
                        train_sizes, train_scores, test_scores = learning_curve(
                            model,
                            X_train,
                            y_train,
                            cv=5,
                            train_sizes=np.linspace(0.1, 1.0, 5),
                            return_times=False,
                        )
                        train_scores_mean = np.mean(train_scores, axis=1)
                        test_scores_mean = np.mean(test_scores, axis=1)
                        y_axis_lower_limit = (
                            min(min(train_scores_mean), min(test_scores_mean)) - 0.01
                        )
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=train_sizes,
                                y=train_scores_mean,
                                mode="lines+markers",
                                name="Training score",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=train_sizes,
                                y=test_scores_mean,
                                mode="lines+markers",
                                name="Cross-validation score",
                            )
                        )
                        fig.update_layout(
                            title="Learning Curve",
                            xaxis_title="Training examples",
                            yaxis_title="Score",
                            yaxis=dict(range=[y_axis_lower_limit, 1]),
                        )
                        main_content_col1.plotly_chart(fig, use_container_width=True)
                        score = mean_squared_error(model.predict(X_test), y_test)
                    else:
                        model.fit(X_train, y_train)
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=X_test[:, 0],
                                y=y_test,
                                mode="markers",
                                name="Test data",
                            )
                        )
                        line_X = np.sort(X_test[:, 0])
                        fig.add_trace(
                            go.Scatter(
                                x=X_test[:, 0],
                                y=model.predict(X_test),
                                mode="lines",
                                name="Prediction",
                            )
                        )
                        fig.update_layout(
                            title="Regression", xaxis_title="X", yaxis_title="y"
                        )
                        main_content_col1.plotly_chart(fig, use_container_width=True)
                        score = mean_squared_error(model.predict(X_test), y_test)

                elif ML_type == "Dimensionality Reduction":
                    model.fit(X_train, y_train)
                    fig = px.scatter_matrix(
                        df,
                        dimensions=df.columns,
                        color="target",
                        title="Dimensionality Reduction",
                    )
                    fig.update_traces(diagonal_visible=False, showupperhalf=False)
                    main_content_col1.plotly_chart(fig, use_container_width=True)

                    components = model.transform(df)
                    samples, n_components = components.shape
                    labels = {
                        str(i): f"PC {i+1} ({var:.1f}%)"
                        for i, var in enumerate(model.explained_variance_ratio_ * 100)
                    }
                    fig2 = px.scatter_matrix(
                        components,
                        labels=labels,
                        dimensions=range(n_components),
                        title="Dimensionality Reduction",
                    )
                    if n_components == 1:
                        fig2.update_traces(diagonal_visible=True, showupperhalf=True)
                    elif n_components <= 3:
                        fig2.update_traces(diagonal_visible=False, showupperhalf=True)
                    else:
                        fig2.update_traces(diagonal_visible=False, showupperhalf=False)
                    fig2.update_layout(
                        height=800, width=800, title="Dimensionality Reduction"
                    )
                    main_content_col1.plotly_chart(fig2, use_container_width=True)
                    cummlative_variance = np.cumsum(model.explained_variance_ratio_)
                    st.markdown(
                        f'<p class="PCA_Result">Cumulative explained variance: {np.round(cummlative_variance[-1],2)}%</p>',
                        unsafe_allow_html=True,
                    )
                    score = model.score(X_test, y_test)

    with Home_col2:
        st.write(dataset_string(dataset))
        if (
            dataset.lower().replace(" ", "_") == "diabetes"
            or dataset.lower().replace(" ", "_") == "iris"
            or dataset.lower().replace(" ", "_") == "breast_cancer"
        ):
            dataset_corr = pd.DataFrame(df).corr()
            color_scale = st.selectbox(
                "Color scale",
                [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "cividis",
                    "blues",
                    "greens",
                    "greys",
                    "oranges",
                    "purples",
                    "reds",
                    "ylorbr",
                    "ylorrd",
                    "sunset",
                    "sunsetdark",
                    "aggrnyl",
                    "agsunset",
                    "algae",
                    "amp",
                    "armyrose",
                    "balance",
                    "blackbody",
                    "bluered",
                    "blugrn",
                    "bluyl",
                    "brbg",
                    "brwnyl",
                    "bugn",
                    "bupu",
                    "burg",
                    "burgyl",
                    "cividis",
                    "curl",
                    "darkmint",
                    "deep",
                    "delta",
                    "dense",
                    "earth",
                    "edge",
                    "electric",
                    "emrld",
                    "fall",
                    "geyser",
                    "gnbu",
                    "gray",
                    "greens",
                    "greys",
                    "haline",
                    "hot",
                    "hsv",
                    "ice",
                    "icefire",
                    "inferno",
                    "jet",
                    "magenta",
                    "magma",
                    "matter",
                    "mint",
                    "mrybm",
                    "mygbm",
                    "oranges",
                    "orrd",
                    "oryel",
                    "peach",
                    "phase",
                    "picnic",
                    "pinkyl",
                    "piyg",
                    "plasma",
                    "plotly3",
                    "portland",
                    "prgn",
                    "pubu",
                    "pubugn",
                    "puor",
                    "purd",
                    "purp",
                    "purples",
                    "purpor",
                    "rainbow",
                    "rdbu",
                    "rdgy",
                    "rdpu",
                    "rdylbu",
                    "rdylgn",
                    "redor",
                    "reds",
                    "solar",
                    "spectral",
                    "speed",
                    "sunset",
                    "sunsetdark",
                    "teal",
                    "tealgrn",
                    "tealrose",
                    "tempo",
                    "temps",
                    "thermal",
                    "tropic",
                    "turbid",
                    "twilight",
                    "viridis",
                    "ylgn",
                    "ylgnbu",
                    "ylorbr",
                    "ylorrd",
                ],
            )
            fig = plot_data(
                color_scale, dataset.lower().replace(" ", "_"), dataset_corr
            )
        else:
            fig = plot_data("target", dataset.lower().replace(" ", "_"), df)
        st.plotly_chart(fig, use_container_width=True)
        # write score here
        if score != None:
            st.markdown(
                f'<p class="Result">Score: {np.round(score,2)}</p>',
                unsafe_allow_html=True,
            )
        if conf != None:
            Home_col2.plotly_chart(conf, use_container_width=True)


with About:
    st.markdown(
        f'<p class="Section_title">{other_string("about")}</p>', unsafe_allow_html=True
    )


# ------------------- Main -------------------

# ------------------- Footer -------------------
st.markdown("---")

st.markdown(f'<p class="footer">{other_string("footer")}</p>', unsafe_allow_html=True)

# ------------------- Footer -------------------
