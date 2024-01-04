def dataset_string(dataset):
    if dataset == "Moon":
        return Moon_dataset
    elif dataset == "Bloobs":
        return Bloobs_dataset
    elif dataset == "Circles":
        return Circles_dataset
    elif dataset == "Classification":
        return Classification_datasets
    elif dataset == "Regression":
        return Regression_datasets
    elif dataset == "Moons":
        return Moon_dataset_clustering
    elif dataset == "Blobs":
        return Blobs_dataset_clustering
    elif dataset == "Circles_clustering":
        return Circles_dataset_clustering
    elif dataset == "Classification_clustering":
        return Classification_datasets_clustering
    elif dataset == "Iris":
        return Iris_dataset
    elif dataset == "Diabetes":
        return Diabetes_dataset
    elif dataset == "Breast Cancer":
        return Breast_cancer_dataset
    else:
        return "Error"


def algorithm_string(algorithm):
    if algorithm == "KNN":
        return Knn
    elif algorithm == "SVM":
        return Svm
    elif algorithm == "Decision Tree":
        return Decision_tree
    elif algorithm == "Random Forest":
        return Random_forest
    elif algorithm == "Linear Regression":
        return Linear_regression
    elif algorithm == "Lasso":
        return Lasso_regression
    elif algorithm == "Ridge":
        return Ridge_regression
    elif algorithm == "Kernel Ridge":
        return Kernel_regression
    elif algorithm == "SVR":
        return Svr
    elif algorithm == "KNN Regression":
        return KNN_regression
    elif algorithm == "Decision Tree Regression":
        return Decision_tree_regression
    elif algorithm == "Random Forest Regression":
        return Random_forest_regression
    elif algorithm == "KMeans":
        return K_means
    elif algorithm == "DBSCAN":
        return DBScan
    elif algorithm == "Agglomerative Clustering":
        return Hierarchical_clustering
    elif algorithm == "Pca":
        return Pca
    else:
        return "Error"

def other_string(text):
    if text == "header":
        return header
    elif text == "footer":
        return footer
    elif text == "about":
        return about
    else:
        return "Error"

# --------------------- Dataset Description --------------------- #

Moon_dataset = """
    The Moon dataset is great for classification tasks, especially for non-linear problems.
    It is a binary classification task, where the data points are shaped as two interleaving half circles.
"""

Bloobs_dataset = """
    The Bloobs dataset is made for binary classification tasks. It contains 2 Blobs that are strictly separated.
    This makes it easy even for linear classifiers to learn a decision boundary.
"""

Circles_dataset = """
    The Circles dataset is made for binary classification tasks. It contains two circles where one is inside the other.
    This makes it a non-linearly separable problem. Due to the circles being so close to each other, it most likely won't have a 100 percent accuracy.
"""

Classification_datasets = """
    The Classification datasets are made for binary classification tasks. They are mostly not linearly separable.
    Changing the random state will change the shape of the data. Sometimes they are interwoven, sometimes they are not.
    Sometimes one class is very spread out, while the other one is very compact.
    This makes it a versatile dataset for testing classifiers.
"""

""" Regression Datasets """

Regression_datasets = """
    This Dataset is simply a few points spreaded out. It is made for regression tasks.
    The goal is to fit a line through the data points. such that it looks like it finds a middle ground.
    Which is why this dataset is versatile for testing regressors.
    Depending on the random state, the data points will be spreaded out differently.
    Sometimes like a line, sometimes like a big blob
"""

""" Clustering Datasets """

Moon_dataset_clustering = """
    The Moon dataset consists of two interleaving half circles. 
    While also used for classification tasks, it can be used for clustering tasks.
"""

Blobs_dataset_clustering = """
    The Bloobs dataset consists of two Blobs that are strictly separated.
    This makes it easy to cluster them, since they are so clearly separated aka clustered.
    """

Circles_dataset_clustering = """
    The Circles dataset consists of two circles where one is inside the other.
    This makes it harder to cluster them, since they are not clearly separated.
    Giving a good result is harder, but not impossible.
    """

Classification_datasets_clustering = """
    The Classification datasets are made for binary classification tasks. But we can misuse them for clustering tasks.
    """

""" Dimensionality Reduction Datasets """

Iris_dataset = """
    The Iris dataset is a very popular dataset for classification tasks.
    it consists of 4 dimensions, which makes it a good dataset for dimensionality reduction.
    """

Diabetes_dataset = """
    The Diabetes dataset is a very popular dataset for regression tasks.
    it consists of 10 dimensions, which makes it a good dataset for dimensionality reduction.
    """

Breast_cancer_dataset = """
    The Breast Cancer dataset is a very popular dataset for classification tasks.
    it consists of 15 dimensions, which makes it a good dataset for dimensionality reduction.
    """

# --------------------- Algorithmen Description --------------------- #

""" Classification Algorithms """

Knn = """
    The KNN is the most classic Classification Algorithm.
    Going through all the data points and checking which class is the most common in the nearest neighbors.
    """

Svm = """
    The SVM is a very popular Classification Algorithm.
    It gave companies billions of dollars in revenue.
    It works by finding a hyperplane that separates the data points such that the margin is maximized. (That means the distance between the hyperplane and the data points is maximized).
    The data points that are closest to the hyperplane are called support vectors and are theoretically the only data points that matter.
    
    The SVM is very good for linearly separable data, but not so good for non-linearly separable data.
    Luckily, there are kernels that can transform the data into a higher dimension, where it is linearly separable.
    The SVM has many kernels, that make it very versatile even for many non-linearly separable problems.
    """

Decision_tree = """
    The Decision Tree works by splitting the data points into smaller and smaller groups.
    It does so by finding the feature that splits the data points the best. It hopes to find a feature and a split, such that one leaf contains only one class.
    The Decision Tree is very good for non-linearly separable data, but it is prone to overfitting.
    """

Random_forest = """
    A Random forest is a collection of Decision Trees. It works by creating many Decision Trees and then letting them vote on the class.
    Often, only a subset of the features is used for each Decision Tree. This is called bagging.
    The Random Forest is very good for non-linearly separable data, and less prone to overfitting than a single Decision Tree.
    """

""" Regression Algorithms """

Linear_regression = """
    The Linear Regression is the most classic Regression Algorithm.
    It works by finding a line that fits the data points the best. that means it minimizes the distance between the line and the data points.
    While there are sophisticated ways to gradually find the best line, the Linear Regression uses the normal equation to find the best line.
    That makes it unique, because it can learn the best line in one step.
    No matter how many dimensions the data has, the Linear Regression can find the best line in one step.
    
    However, the Linear Regression is only good for linearly separable data.
    """

Lasso_regression = """
    The Lasso Regression is a Regression Algorithm that uses regularization.
    So while it works like the Linear Regression, it also tries to minimize the weights of the features.
    This makes it less prone to overfitting.
    """

Ridge_regression = """
    The Ridge Regression is a Regression Algorithm that uses regularization.
    So while it works like the Linear Regression, it also tries to minimize the weights of the features.
    This makes it less prone to overfitting.
    """

Kernel_regression = """
    The Kernel Regression is a Regression Algorithm that uses kernels.
    So while it works like the Linear Regression, it also uses kernels to transform the data into a higher dimension.
    This makes it very good for non-linearly separable data.
    """

Svr = """
    The SVR is a very popular Regression Algorithm.
    It is similar to SVM, but tailored for regression tasks.
    """

KNN_regression = """
    KNN is another algorithm that can be used for regression tasks.
    It works by finding the nearest neighbors and then averaging their values.
    """

Decision_tree_regression = """
    The Decision Tree Regression works by splitting the data points into smaller and smaller groups.
    It does so by finding the feature that splits the data points the best. It hopes to find a feature and a split, such that one leaf contains only one value.
    I.e. not a class, but a value.
    The Decision Tree Regression is very good for non-linearly separable data, but it is prone to overfitting.
    """

Random_forest_regression = """
    A Random forest is a collection of Decision Trees. It works by creating many Decision Trees and then averaging their values.
    Often, only a subset of the features is used for each Decision Tree. This is called bagging.
    The Random Forest Regression is very good for non-linearly separable data, and less prone to overfitting than a single Decision Tree.
    """

""" Clustering Algorithms """

K_means = """
    The K-Means algorithm is a very popular clustering algorithm.
    It initializes k centroids and then assigns each data point to the nearest centroid.
    then it recalculates the centroids and repeats the process until the centroids don't move anymore.
    """

DBScan = """
    The DBSCAN or Density-Based Spatial Clustering of Applications with Noise is a clustering algorithm, that doesn't need to know the number of clusters.
    It works by finding a core sample and then expanding the cluster by finding all core samples that are reachable. defined by a distance and a number of samples.
    Afterward, it includes all border samples that are reachable by the core samples but not core samples themselves.
    If no more samples can be added, it starts a new cluster.
    Every sample left is considered noise.
    """

Hierarchical_clustering = """
    The Hierarchical Clustering is a clustering algorithm that creates a hierarchy of clusters.
    It works by defining each data point as a cluster and then merging the two closest clusters.
    It repeats this process until there is only one cluster left.
    This creates a hierarchy of clusters, which can be visualized as a dendrogram.
    """

""" Dimensionality Reduction Algorithms """

Pca = """
    The PCA or Principal Component Analysis is a very popular dimensionality reduction algorithm.
    By finding the most variance in the data, it can reduce the dimensionality of the data.
    This makes it easier to visualize the data and to train models on it.
    But also makes it harder to interpret the data.
    
    PCA features are not the same as the original features. They are a combination of the original features. That are orthogonal to each other.
    """


# --------------------- Header, Footer, About, Description --------------------- #

header = """
    <h1>Machine Learning Playground</h1>
    <h2>Explore and visualize different algorithms from scikit-learn and how they work on different datasets</h2>
    <ul>
        <li>Choose a dataset</li>
        <li>Choose an algorithm</li>
        <li>Choose the hyperparameters</li>
        <li>click on "Run" to see the results</li>
    </ul>
    
    Machine Learning can be very confusing and overwhelming.
    But it has the advantage of being very intuitive when visualized.
    You can view Machine Learning as a way to find mathematical language to describe how humans would solve a problem.
    <br>
    So if you want to learn Machine Learning, you should start with the intuition.
    From the Intuition, you can learn the mathematics. You can see which part of the formula represents which part of the intuition.
    Especially those, non neural network algorithms, are very intuitive. As they are not a giant black box of neurons and numbers.
    They are simple algorithms that are easy to understand, if you know what they want to convey.
    While I don't can give you the mathematics (there are fantastic books for that), I can give you the intuition. The visualization.
"""

footer = """
        2023 - Sci-kit Learning Playground
        <br>
        Created by <a href="https://github.com/Resch-Gabriel-Z"<b>Gabe</b></a>
"""

about = """
    <h1>About</h1>
    <p>
        This was made to visualize how different algorithms work on different datasets.
        How some algorithms have poor results due to data being linearly inseparable.
        or how some algorithms are very sensitive to hyperparameters.
        I hope it helps you a bit :)
    </p>
    <p>
        The source code can be found on <a href="
        </p>
"""