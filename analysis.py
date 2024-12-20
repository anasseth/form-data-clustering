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

def setup_logger():
    # Create logger
    logger = logging.getLogger('clustering')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(c_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    return logger

def cluster_analysis(data, min_size, max_size):
    logger = setup_logger()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress_pct, status_msg):
        progress_bar.progress(progress_pct)
        status_text.text(status_msg)
        logger.info(status_msg)
    
    try:
        # Step 1: Preprocess the Data (10%)
        update_progress(0.1, "Starting data preprocessing...")
        start_time = time.time()
        
        label_encoders = {}
        for col in data.select_dtypes(include='object').columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        
        update_progress(0.2, f"Label encoding completed in {time.time() - start_time:.2f} seconds")
        
        # Scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        update_progress(0.3, "Data scaling completed")
        
        # Step 2: PCA (20%)
        update_progress(0.3, "Starting PCA dimensionality reduction...")
        pca_start = time.time()
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        update_progress(0.4, f"PCA completed in {time.time() - pca_start:.2f} seconds")
        
        def find_optimal_clusters(data, min_size=min_size, max_size=max_size):
            update_progress(0.5, "Finding optimal number of clusters...")
            n_samples = len(data)
            min_clusters = n_samples // max_size
            max_clusters = n_samples // min_size
            
            logger.info(f"Sample size: {n_samples}")
            logger.info(f"Possible cluster range: {min_clusters} to {max_clusters}")
            
            if min_clusters == max_clusters:
                return min_clusters
                
            best_score = -1
            optimal_k = min_clusters
            
            for k in range(min_clusters, max_clusters + 1):
                update_progress(0.5 + (0.1 * (k - min_clusters) / (max_clusters - min_clusters)),
                              f"Testing {k} clusters...")
                
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(data)
                cluster_sizes = np.bincount(labels)
                
                valid_sizes = all(min_size <= size <= max_size for size in cluster_sizes[:-1])
                last_cluster_valid = min_size <= cluster_sizes[-1] <= max_size + (n_samples % min_size)
                
                if valid_sizes and last_cluster_valid:
                    score = silhouette_score(data, labels)
                    logger.info(f"K={k}: Valid clustering found, silhouette score = {score:.4f}")
                    if score > best_score:
                        best_score = score
                        optimal_k = k
                else:
                    logger.info(f"K={k}: Invalid clustering, cluster sizes = {cluster_sizes}")
            
            return optimal_k
        
        def balanced_clustering(data, n_clusters, min_size=min_size, max_size=max_size, max_attempts=50):
            update_progress(0.7, f"Starting balanced clustering with {n_clusters} clusters...")
            n_samples = len(data)
            best_labels = None
            best_score = -1
            
            for attempt in range(max_attempts):
                update_progress(0.7 + (0.2 * attempt / max_attempts),
                              f"Clustering attempt {attempt + 1}/{max_attempts}")
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=attempt)
                labels = kmeans.fit_predict(data)
                cluster_sizes = np.bincount(labels)
                
                valid_sizes = all(min_size <= size <= max_size for size in cluster_sizes[:-1])
                last_cluster_valid = min_size <= cluster_sizes[-1] <= max_size + (n_samples % min_size)
                
                if valid_sizes and last_cluster_valid:
                    score = silhouette_score(data, labels)
                    logger.info(f"Attempt {attempt + 1}: Valid clustering found, score = {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                else:
                    logger.info(f"Attempt {attempt + 1}: Invalid clustering, sizes = {cluster_sizes}")
            
            if best_labels is None:
                logger.warning("No valid clustering found, using constrained K-means")
                best_labels = constrained_kmeans(data, n_clusters, min_size, max_size)
            
            return best_labels
        
        def constrained_kmeans(data, n_clusters, min_size, max_size):
            update_progress(0.9, "Starting constrained K-means...")
            n_samples = len(data)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data)
            
            iteration = 0
            while iteration < 100:  # Add maximum iteration limit
                cluster_sizes = np.bincount(labels)
                if all(min_size <= size <= max_size for size in cluster_sizes[:-1]) and \
                   min_size <= cluster_sizes[-1] <= max_size + (n_samples % min_size):
                    break
                
                logger.info(f"Iteration {iteration + 1}: Adjusting cluster sizes {cluster_sizes}")
                
                overflow_clusters = np.where(cluster_sizes > max_size)[0]
                underflow_clusters = np.where(cluster_sizes < min_size)[0]
                
                if len(overflow_clusters) > 0:
                    for cluster in overflow_clusters:
                        while cluster_sizes[cluster] > max_size:
                            points_in_cluster = np.where(labels == cluster)[0]
                            target_cluster = underflow_clusters[0] if len(underflow_clusters) > 0 else \
                                           np.argmin(cluster_sizes)
                            
                            distances = np.linalg.norm(
                                data[points_in_cluster] - kmeans.cluster_centers_[target_cluster], 
                                axis=1
                            )
                            point_to_move = points_in_cluster[np.argmin(distances)]
                            labels[point_to_move] = target_cluster
                            cluster_sizes = np.bincount(labels)
                
                iteration += 1
            
            return labels
        
        # Find optimal clusters and perform clustering
        optimal_k = find_optimal_clusters(reduced_data)
        logger.info(f"Optimal number of clusters determined: {optimal_k}")
        
        cluster_labels = balanced_clustering(reduced_data, optimal_k)
        
        # Calculate final metrics
        update_progress(0.95, "Calculating final metrics...")
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        
        # Calculate WCSS
        wcss = []
        cluster_range = range(max(2, optimal_k - 3), optimal_k + 4)
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(reduced_data)
            wcss.append(kmeans.inertia_)
        
        # Calculate correlation matrix
        correlation_matrix = pd.DataFrame(scaled_data, columns=data.columns).corr()
        
        update_progress(1.0, "Clustering analysis completed successfully!")
        
        return (
            data,
            silhouette_avg,
            {"wcss": wcss, "range_values": list(cluster_range)},
            {"reduced_data": reduced_data, "cluster_labels": cluster_labels},
            correlation_matrix
        )
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}")
        raise e
    finally:
        # Clean up progress bar
        progress_bar.empty()
        status_text.empty()
        
def plot_elbow(wcss, range_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=range_values, 
        y=wcss, 
        mode='lines+markers',
        marker=dict(size=10, color='royalblue'),
        line=dict(dash="dash")
    ))
    fig.update_layout(
        title="Elbow Method for Optimal Clusters",
        xaxis_title="Number of Clusters",
        yaxis_title="WCSS",
        template="plotly"
    )
    return fig

# Cluster Visualization Function
def plot_clusters(reduced_data, cluster_labels):
    fig = px.scatter(
        x=reduced_data[:, 0], 
        y=reduced_data[:, 1], 
        color=cluster_labels.astype(str),
        title="Clusters After PCA",
        labels={'x': "PCA Component 1", 'y': "PCA Component 2"},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    return fig

# Heatmap Function
def plot_heatmap(correlation_matrix):
    fig = px.imshow(
        correlation_matrix, 
        text_auto=True, 
        color_continuous_scale=[[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']],
        title="Feature Correlation Heatmap",
    )
    return fig
