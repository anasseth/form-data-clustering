import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import plotly.express as px
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClusteringApp")

def preprocess_data(data, meta_dict):
    """Preprocess the data based on meta_dict formulas and weights."""
    even_dist_cols = [col for col, meta in meta_dict.items() if meta["formula"] == "Even Distribution"]
    group_similarity_cols = [col for col, meta in meta_dict.items() if meta["formula"] == "Group Similarity"]

    # One-hot encode even distribution columns
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    even_dist_encoded = encoder.fit_transform(data[even_dist_cols])
    even_dist_encoded = pd.DataFrame(even_dist_encoded, columns=encoder.get_feature_names_out(even_dist_cols))
    even_dist_encoded *= [meta_dict[col.split("_")[0]]["weight"] for col in even_dist_encoded.columns]

    # Label encode group similarity columns
    group_sim_encoded = data[group_similarity_cols].copy()
    label_encoders = {}
    for col in group_similarity_cols:
        le = LabelEncoder()
        group_sim_encoded[col] = le.fit_transform(group_sim_encoded[col]) * meta_dict[col]["weight"]
        label_encoders[col] = le

    # Combine processed columns
    weighted_data = pd.concat([even_dist_encoded, group_sim_encoded.reset_index(drop=True)], axis=1)
    return weighted_data, label_encoders

def enforce_strict_cluster_size(data, cluster_sizes, min_size, max_size, max_iterations=200):
    """Ensure clusters strictly adhere to min and max size constraints with a maximum number of iterations."""
    def rebalance_clusters(data, cluster_sizes, min_size, max_size):
        large_clusters = cluster_sizes[cluster_sizes > max_size].index
        small_clusters = cluster_sizes[cluster_sizes < min_size].index

        for large_cluster in large_clusters:
            excess_members = data[data['Group'] == large_cluster].iloc[max_size:]
            for small_cluster in small_clusters:
                if len(excess_members) == 0:
                    break
                to_transfer = min(len(excess_members), min_size - cluster_sizes[small_cluster])
                indices_to_move = excess_members.index[:to_transfer]
                data.loc[indices_to_move, 'Group'] = small_cluster
                excess_members = excess_members.iloc[to_transfer:]
                cluster_sizes[small_cluster] += to_transfer
            cluster_sizes[large_cluster] = max_size

        return data

    iteration = 0
    while (any(cluster_sizes > max_size) or any(cluster_sizes < min_size)) and iteration < max_iterations:
        data = rebalance_clusters(data, cluster_sizes, min_size, max_size)
        cluster_sizes = data['Group'].value_counts()
        iteration += 1

    if iteration == max_iterations:
        logger.warning("Rebalancing stopped after reaching the maximum number of iterations.")

    return data


def cluster_data(data, meta_dict, min_group_size, max_group_size):
    """Group the data and rebalance to meet group size constraints."""
    weighted_data, _ = preprocess_data(data, meta_dict)

    # Determine the number of clusters
    total_members = weighted_data.shape[0]
    target_group_size = (min_group_size + max_group_size) // 2
    n_clusters = max(2, total_members // target_group_size)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(weighted_data)
    data['Group'] = cluster_labels

    # Rebalance clusters strictly
    cluster_sizes = data['Group'].value_counts()
    data = enforce_strict_cluster_size(data, cluster_sizes, min_group_size, max_group_size)

    return data, kmeans, weighted_data

def explain_silhouette_score(score):
    """Provide an explanation based on the silhouette score."""
    if score > 0.7:
        return """📈 **Excellent Grouping (> 0.7)** - Very well-defined clusters."""
    elif score > 0.5:
        return """✅ **Good Grouping (0.5 - 0.7)** - Clear group structure."""
    elif score > 0.25:
        return """📊 **Moderate Grouping (0.25 - 0.5)** - Reasonable group structure."""
    elif score > 0:
        return """⚠️ **Weak Grouping (0 - 0.25)** - Groups have some structure."""
    else:
        return """❌ **Poor Grouping (< 0)** - Groups may be incorrectly assigned."""

def explain_calinski_harabasz_score(score):
    """Provide an explanation based on the Calinski-Harabasz score."""
    if score > 500:
        return """📈 **Excellent Score (> 500)** - Very well-defined clusters."""
    elif score > 200:
        return """✅ **Good Score (200 - 500)** - Well-separated clusters."""
    elif score > 50:
        return """📊 **Moderate Score (50 - 200)** - Decent cluster separation."""
    else:
        return """⚠️ **Low Score (< 100)** - Clusters may overlap significantly."""

def explain_davies_bouldin_score(score):
    """Provide an explanation based on the Davies-Bouldin score."""
    if score < 0.5:
        return """📈 **Excellent Score (< 0.5)** - Well-separated and compact clusters."""
    elif score < 1.0:
        return """✅ **Good Score (0.5 - 1.0)** - Reasonable cluster separation."""
    elif score < 2.0:
        return """📊 **Moderate Score (1.0 - 2.0)** - Acceptable clustering performance."""
    else:
        return """⚠️ **High Score (> 2.0)** - Poorly separated clusters."""

def evaluate_clusters(data, weighted_data):
    """Evaluate clustering using multiple metrics."""
    silhouette_avg = silhouette_score(weighted_data, data['Group'])
    calinski_harabasz = calinski_harabasz_score(weighted_data, data['Group'])
    davies_bouldin = davies_bouldin_score(weighted_data, data['Group'])
    return silhouette_avg, calinski_harabasz, davies_bouldin

def visualize_clusters(data, weighted_data, kmeans):
    """Visualize clusters using t-SNE for better insights."""
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(weighted_data)
    
    reduced_data_df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
    reduced_data_df['Group'] = data['Group']

    fig = px.scatter(
        reduced_data_df, 
        x='Component 1', 
        y='Component 2', 
        color='Group', 
        title="Group Visualization (t-SNE Reduced)",
        labels={'Group': 'Group'},
        template="plotly_white"
    )
    st.plotly_chart(fig)

# Streamlit App
st.title("Clustering App")
st.write("Please open sidenav and start uploading dataset and metadata file.")

# File Upload
st.sidebar.header("Upload Files")
data_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")
meta_file = st.sidebar.file_uploader("Upload Metadata (CSV)", type="csv")

# Input constraints
st.sidebar.header("Constraints")
min_group_size = st.sidebar.number_input("Minimum Group Size", min_value=1, value=8)
max_group_size = st.sidebar.number_input("Maximum Group Size", min_value=1, value=10)

if data_file and meta_file:
    data = pd.read_csv(data_file)
    metadata = pd.read_csv(meta_file)
    meta_dict = metadata.set_index('column_name').to_dict(orient='index')

    # Display metadata configuration in the main page
    st.header("Metadata Configuration")
    with st.expander("Configure Metadata", expanded=True):
        for column in data.columns:
            if column in meta_dict:
                # Heading for the column
                st.text(f"{column}")
                # Row with weight and formula
                col1, col2 = st.columns(2)
                with col1:
                    weight = st.number_input(
                        f"Weight",
                        value=float(meta_dict[column]["weight"]),
                        key=f"weight_{column}"
                    )
                with col2:
                    formula = st.selectbox(
                        f"Formula",
                        ["Even Distribution", "Group Similarity"],
                        index=["Even Distribution", "Group Similarity"].index(meta_dict[column]["formula"]),
                        key=f"formula_{column}"
                    )
                meta_dict[column]["formula"] = formula
                meta_dict[column]["weight"] = weight

    if st.button("Run Clustering"):
        clustered_data, kmeans_model, processed_data = cluster_data(data, meta_dict, min_group_size, max_group_size)
        silhouette, calinski_harabasz, davies_bouldin = evaluate_clusters(clustered_data, processed_data)

        # Display results
        st.header("Results")
        
        with st.container(border=True):
            st.write(f"Silhouette Score: {silhouette:.3f}")
            st.markdown(explain_silhouette_score(silhouette))

        with st.container(border=True):
            st.write(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
            st.markdown(explain_calinski_harabasz_score(calinski_harabasz))
            
        with st.container(border=True):
            st.write(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
            st.markdown(explain_davies_bouldin_score(davies_bouldin))

        # Display cluster visualization
        st.header("Visualization")
        visualize_clusters(clustered_data, processed_data, kmeans_model)
        
        # Show data points per cluster in horizontal format
        st.header("Sizes")
        cluster_sizes = clustered_data['Group'].value_counts().reset_index()
        cluster_sizes.columns = ['Group', 'Size']

        # Transpose for horizontal view
        transposed_sizes = cluster_sizes.set_index('Group').T

        st.write("Group Sizes (Horizontal View):")
        st.dataframe(transposed_sizes)

        # Convert clustered_data to CSV
        csv = clustered_data.to_csv(index=False)
        b64 = io.BytesIO(csv.encode()).getvalue()  # Encode as bytes

        # Add a download button
        st.download_button(
            label="Download Clustered Data as CSV",
            data=b64,
            file_name="clustered_data.csv",
            mime="text/csv",
        )