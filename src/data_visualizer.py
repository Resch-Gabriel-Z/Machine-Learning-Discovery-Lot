import plotly.express as px
def plot_data(color,dataset_name,dataset):
    if dataset_name == "moons":
        fig = px.scatter(dataset, x="x", y="y", color=color, title="Moons")
        return fig
    elif dataset_name == "blobs":
        fig = px.scatter(dataset, x="x", y="y", color=color, title="Blobs")
        return fig
    elif dataset_name == "circles":
        fig = px.scatter(dataset, x="x", y="y", color=color, title="Circles")
        return fig
    elif dataset_name == "classification":
        fig = px.scatter(dataset, x="x", y="y", color=color, title="Classification")
        return fig
    elif dataset_name == 'diabetes' or dataset_name == 'iris' or dataset_name == 'breast_cancer':
        fig = px.imshow(dataset, color_continuous_scale=color,text_auto=True,range_color=[-1, 1])
        return fig
    elif dataset_name == "regression":
        fig = px.scatter(dataset, x="x", y="target", color=color, title="Regression")
        return fig
    
    