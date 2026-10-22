
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


def preprocessing(df, features):
    X = df[features].copy()

    numerical = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    scaler = StandardScaler()

    if len(numerical) > 0:
        X[numerical] = scaler.fit_transform(X[numerical])

    for column in categorical:
        X[column] = LabelEncoder().fit_transform(
            X[column].astype(str)
        )

    return X


def clustering(
    X,
    algorithm,
    n_clusters=3,
    linkage="ward",
    eps=0.5,
    min_samples=5
):
    if algorithm == "Hierarchical Clustering":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )

    elif algorithm == "DBSCAN":
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

    else:
        raise ValueError("Clustering algorithm not supported")

    return model.fit_predict(X)


def evaluate(X, labels):
    unique_labels = set(labels)

    if -1 in unique_labels:
        unique_labels = unique_labels - {-1}
        mask = labels != -1
        X_eval = X.loc[mask] if hasattr(X, "loc") else X[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X
        labels_eval = labels

    n_clusters = len(set(labels_eval))

    if n_clusters < 2 or n_clusters >= len(labels_eval):
        return None, None

    silhouette = silhouette_score(X_eval, labels_eval)
    davies = davies_bouldin_score(X_eval, labels_eval)

    return silhouette, davies
