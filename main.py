import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from analysis import cluster_analysis, plot_clusters, plot_elbow, plot_heatmap

def download_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name='Clusters')
    output.seek(0)
    return output

def fetch_google_sheet(sheet_url):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)
    data = pd.DataFrame(worksheet.get_all_records())
    return data

def explain_silhouette_score(score):
    if score > 0.7:
        return """📈 **Excellent Clustering (> 0.7)**
                 - Very well-defined clusters
                 - Strong structure found in the data
                 - High confidence in the grouping results"""
    elif score > 0.5:
        return """✅ **Good Clustering (0.5 - 0.7)**
                 - Clear cluster structure
                 - Reliable groupings
                 - Some overlap between groups, but generally distinct"""
    elif score > 0.25:
        return """📊 **Moderate Clustering (0.25 - 0.5)**
                 - Reasonable cluster structure
                 - Groups are somewhat distinct
                 - Some overlap between clusters"""
    elif score > 0:
        return """⚠️ **Weak Clustering (0 - 0.25)**
                 - Clusters have some structure
                 - Significant overlap between groups
                 - Consider adjusting parameters or features"""
    else:
        return """❌ **Poor Clustering (< 0)**
                 - Clusters may be incorrectly assigned
                 - Consider different features or parameters
                 - Data might not have clear group structure"""

# Streamlit UI
st.title("Grouping Form Data")

# Sidebar for configuration
st.sidebar.header("Analysis Configuration")

# File Upload
file = st.file_uploader("Upload an Excel file", type=['xlsx', 'csv'])
google_sheet_url = st.text_input("Or enter Google Sheet URL")

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

# Load data first
if file or google_sheet_url:
    try:
        if file:
            file_ext = file.name.split('.')[-1]
            if file_ext == 'csv':
                st.session_state.data = pd.read_csv(file)
            elif file_ext in ['xls', 'xlsx']:
                st.session_state.data = pd.read_excel(file)
        elif google_sheet_url:
            st.session_state.data = fetch_google_sheet(google_sheet_url)
            st.success("Data fetched successfully from Google Sheets!")
        
        # Show data preview
        if st.session_state.data is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head())
            
            # Column selection
            st.sidebar.subheader("Feature Selection")
            all_columns = st.session_state.data.columns.tolist()
            selected_columns = st.sidebar.multiselect(
                "Select columns for analysis",
                all_columns,
                default=all_columns
            )
            
            # Group size parameters
            st.sidebar.subheader("Group Size Parameters")
            col1, col2 = st.sidebar.columns(2)
            min_group_size = col1.number_input("Min Group Size", 
                                             min_value=2, 
                                             max_value=20, 
                                             value=7)
            max_group_size = col2.number_input("Max Group Size", 
                                             min_value=min_group_size, 
                                             max_value=30, 
                                             value=12)
            
            if st.button("Run Analysis"):
                try:
                    # Filter data for selected columns
                    analysis_data = st.session_state.data[selected_columns].copy()
                    
                    # Run cluster analysis with parameters
                    clustered_data, silhouette_pca, elbow_data, cluster_data, corr_matrix = \
                        cluster_analysis(analysis_data, min_size=min_group_size, max_size=max_group_size)
                    
                    # Display Results
                    st.header("Analysis Results")
                    
                    # Silhouette Score with explanation
                    st.subheader("Clustering Quality")
                    score_col, explain_col = st.columns([1, 2])
                    with score_col:
                        st.metric("Silhouette Score", f"{silhouette_pca:.3f}")
                    with explain_col:
                        st.markdown(explain_silhouette_score(silhouette_pca))
                    
                    # Display Interactive Plots
                    st.subheader("Interactive Elbow Method Plot")
                    st.plotly_chart(plot_elbow(elbow_data["wcss"], 
                                             elbow_data["range_values"]), 
                                  use_container_width=True)
                    
                    st.subheader("Interactive Clusters After PCA")
                    st.plotly_chart(plot_clusters(cluster_data["reduced_data"], 
                                                cluster_data["cluster_labels"]), 
                                  use_container_width=True)
                    
                    st.subheader("Feature Correlation Heatmap")
                    st.plotly_chart(plot_heatmap(corr_matrix), 
                                  use_container_width=True)
                    
                    # Download results
                    st.download_button(
                        label="Download Clustered Data as Excel",
                        data=download_excel(clustered_data),
                        file_name="clustered_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during clustering: {e}")
                    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    st.info("Please upload a file or enter a Google Sheet URL to begin.")