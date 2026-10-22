
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import preprocessing, clustering, evaluate


st.title("Clustering App")


file = st.file_uploader(
    "Upload your dataset",
    type=["csv"]
)

if file is None:
    st.info("Upload a CSV file to start clustering.")
    st.stop()

df = pd.read_csv(file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

features = st.multiselect(
    "Select the features for clustering",
    df.columns.to_list()
)

if len(features) < 2:
    st.warning("Select at least 2 features.")
    st.stop()

st.subheader("Model Selection")

algorithm = st.selectbox(
    "Select a clustering algorithm",
    ["DBSCAN", "Hierarchical Clustering"]
)


if algorithm == "Hierarchical Clustering":

    n_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    linkage = st.selectbox(
        "Select linkage",
        ["ward", "single", "complete"]
    )

    eps = 0.5
    min_samples = 5

else:

    eps = st.slider(
        "DBSCAN eps",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1
    )

    min_samples = st.slider(
        "DBSCAN min_samples",
        min_value=2,
        max_value=20,
        value=5
    )

    n_clusters = 3
    linkage = "ward"


try:
    X = preprocessing(df, features)

    if X.isnull().values.any():
        st.error(
            "The selected features contain missing values. "
            "Please clean the dataset before clustering."
        )
        st.stop()

    
    labels = clustering(
        X,
        algorithm=algorithm,
        n_clusters=n_clusters,
        linkage=linkage,
        eps=eps,
        min_samples=min_samples
    )

except Exception as error:
    st.error(f"Error during clustering: {error}")
    st.stop()


df_result = df.copy()
df_result["cluster"] = labels

st.subheader("Clustered Dataset")
st.dataframe(df_result)


st.subheader("Cluster Distribution")

cluster_counts = pd.Series(labels).value_counts().sort_index()

for cluster, count in cluster_counts.items():
    if cluster == -1:
        st.write(f"Noise points: {count}")
    else:
        st.write(f"Cluster {cluster}: {count} observations")


st.subheader("Model Evaluation")

silhouette, davies = evaluate(X, labels)

if silhouette is not None and davies is not None:
    st.write("Silhouette Score:", round(silhouette, 4))
    st.write("Davies-Bouldin Index:", round(davies, 4))
else:
    st.info(
        "The evaluation metrics cannot be calculated "
        "for the current clustering results."
    )


st.subheader("Cluster Visualization")

if len(features) == 2:

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X[features[0]],
        X[features[1]],
        c=labels,
        cmap="viridis"
    )

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f"Results: {algorithm}")

    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info(
        "Select exactly 2 features to visualize the clusters."
    )

csv = df_result.to_csv(index=False).encode("utf-8")


