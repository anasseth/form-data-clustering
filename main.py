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
    worksheet = sheet.get_worksheet(0)  # Fetch the first sheet
    data = pd.DataFrame(worksheet.get_all_records())
    return data

# Streamlit UI
st.title("Grouping Form Data")

# File Upload
file = st.file_uploader("Upload an Excel file", type=['xlsx', 'csv'])
google_sheet_url = st.text_input("Or enter Google Sheet URL")

if st.button("Run Analysis"):
    if file:
        try:
            file_ext = file.name.split('.')[-1]
            if file_ext == 'csv':
                data = pd.read_csv(file)
            elif file_ext in ['xls', 'xlsx']:
                data = pd.read_excel(file)
            else:
                st.error("Unsupported file format! Please upload a CSV or Excel file.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    elif google_sheet_url:
        try:
            data = fetch_google_sheet(google_sheet_url)
            st.success("Data fetched successfully from Google Sheets!")
        except Exception as e:
            st.error(f"Error fetching Google Sheet: {e}")
            st.stop()
    else:
        st.warning("Please upload a file or enter a Google Sheet URL.")
        st.stop()

    try:
        # Run cluster analysis and fetch outputs
        clustered_data, silhouette_pca, elbow_data, cluster_data, corr_matrix = cluster_analysis(data)

        # Display Silhouette Score
        st.success(f"Silhouette Score with PCA: {silhouette_pca:.2f}")

        # Display Interactive Elbow Plot
        st.subheader("Interactive Elbow Method Plot")
        st.plotly_chart(plot_elbow(elbow_data["wcss"], elbow_data["range_values"]), key="elbow_plot")

        # Display Interactive Clusters Plot
        st.subheader("Interactive Clusters After PCA")
        st.plotly_chart(plot_clusters(cluster_data["reduced_data"], cluster_data["cluster_labels"]), key="clusters_plot")

        # Display Interactive Heatmap
        st.subheader("Interactive Correlation Heatmap")
        st.plotly_chart(plot_heatmap(corr_matrix), key="heatmap_plot")

        # Download Processed Data
        output_file = BytesIO()
        clustered_data.to_excel(output_file, index=False, engine='openpyxl')
        output_file.seek(0)
        st.download_button(
            label="Download Clustered Data as Excel",
            data=output_file,
            file_name="clustered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"An error occurred during clustering: {e}")