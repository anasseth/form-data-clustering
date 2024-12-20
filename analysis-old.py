from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

def cluster_analysis(data):
    # Step 1: Preprocess the Data
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Scale the numerical data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Step 2: PCA for Dimensionality Reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    # Step 3: Elbow Method for Optimal Clusters
    wcss = []
    cluster_range = range(8, 15)  # Focus on clusters between 8 and 15
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(reduced_data)
        wcss.append(kmeans.inertia_)

    # Find the initial optimal number of clusters
    elbow = KneeLocator(cluster_range, wcss, curve="convex", direction="decreasing")
    optimal_clusters = elbow.knee if elbow.knee else 8  # Default to 8 clusters if elbow fails

    # Step 4: Enforce Cluster Size Constraints
    def find_valid_clusters(data, optimal_k, min_size=8, max_size=10, max_retries=10):
        best_kmeans = None
        best_labels = None
        silhouette_best = -1
        
        # Adjust cluster count dynamically
        for k in range(optimal_k, 7, -1):  # Start from optimal_k and go down to 8
            for random_state in range(max_retries):  # Retry clustering with different seeds
                kmeans = KMeans(n_clusters=k, random_state=random_state)
                labels = kmeans.fit_predict(data)
                cluster_sizes = np.bincount(labels)

                # Check if all cluster sizes meet constraints
                if all(min_size <= size <= max_size for size in cluster_sizes):
                    silhouette_avg = silhouette_score(data, labels)
                    if silhouette_avg > silhouette_best:
                        silhouette_best = silhouette_avg
                        best_kmeans = kmeans
                        best_labels = labels
                        print(f"Valid clustering found: k={k}, silhouette={silhouette_avg}")
            
            # Stop if a valid solution is found
            if best_kmeans:
                break

        if not best_kmeans:
            print("Warning: No clustering met the size constraints. Returning fallback.")
            best_kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(data)
            best_labels = best_kmeans.labels_

        return best_labels, best_kmeans

    # Find clusters with constraints
    cluster_labels, final_kmeans = find_valid_clusters(reduced_data, optimal_clusters)

    # Step 5: Silhouette Score
    silhouette_avg = silhouette_score(reduced_data, cluster_labels)

    # Step 6: Correlation Matrix for Heatmap
    correlation_matrix = pd.DataFrame(scaled_data, columns=data.columns).corr()

    # Return results
    return (
        data, 
        silhouette_avg, 
        {"wcss": wcss, "range_values": list(cluster_range)},  # Elbow Data
        {"reduced_data": reduced_data, "cluster_labels": cluster_labels},  # Cluster Visualization Data
        correlation_matrix  # Correlation Heatmap Data
    )