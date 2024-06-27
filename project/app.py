import os
import streamlit as st
import tarfile
import urllib.request
import shutil
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, to_date, lit, max
import logging
import sys

@st.cache_resource
def setup_java():
    java_url = "https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz"
    java_tar = "openjdk-11.0.2_linux-x64_bin.tar.gz"
    java_dir = "jdk-11.0.2"

    if not os.path.exists(java_dir):
        try:
            if shutil.which('wget'):
                os.system(f"wget {java_url}")
            else:
                urllib.request.urlretrieve(java_url, java_tar)
        except Exception as e:
            st.error(f"Failed to download Java: {str(e)}")
            return None

        try:
            with tarfile.open(java_tar, "r:gz") as tar:
                tar.extractall()
            os.remove(java_tar)
        except Exception as e:
            st.error(f"Failed to extract Java: {str(e)}")
            return None

    java_home = os.path.abspath(java_dir)
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"
    return java_home

java_home = setup_java()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['JAVA_HOME'] = java_home
logger.info(f"JAVA_HOME set to: {java_home}")

@st.cache_resource
def get_spark_session():
    try:
        logger.info(f"Current PATH: {os.environ.get('PATH', 'Not set')}")
        logger.info(f"Current LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        
        spark = SparkSession.builder \
            .appName("CustomerSegmentation") \
            .config("spark.driver.extraJavaOptions", "-Xss4M") \
            .config("spark.executor.extraJavaOptions", "-Xss4M") \
            .getOrCreate()
        logger.info("SparkSession created successfully")
        return spark
    except Exception as e:
        logger.error(f"Error creating SparkSession: {str(e)}")
        st.error(f"Failed to initialize Spark: {str(e)}")
        sys.exit(1)

try:
    spark = get_spark_session()
    logger.info("Spark initialization completed")
except Exception as e:
    logger.error(f"Unhandled exception during Spark initialization: {str(e)}")
    st.error(f"An unexpected error occurred: {str(e)}")
    sys.exit(1)

@st.cache_data
def load_data():
    df_features = spark.read.parquet("project/df_features.parquet")
    reference_date = df_features.agg(max("LastPurchasesDate")).collect()[0][0]
    reference_date_lit = lit(reference_date)
    df_features = df_features.withColumn("DaysSinceLastPurchase", datediff(to_date(reference_date_lit), "LastPurchasesDate"))
    df_features = df_features.withColumn("DaysSinceFirstPurchase", datediff(to_date(reference_date_lit), "FirstPurchasesDate"))
    df_features = df_features.drop("LastPurchasesDate", "FirstPurchasesDate")
    return df_features.toPandas()

df_features = load_data()

def main():
    st.title("Interactive Customer Segmentation")
    st.sidebar.header("Clustering Settings")
    
    algorithm = st.sidebar.selectbox(
        "Algorithm", 
        ("K-means", "Gaussian Mixture", "Bisecting K-Means"),
        help="K-means: Partitions data into k clusters. Simple and fast, but assumes spherical clusters.\n\n"
             "Gaussian Mixture: Probabilistic model, allows for clusters of different shapes and sizes.\n\n"
             "Bisecting K-Means: Hierarchical variant of K-means, can handle non-globular clusters better."
    )

    max_iter = 20
    if algorithm == "Bisecting K-Means":
        max_iter = st.sidebar.slider(
            "Max Iterations", 
            5, 50, 20,
            help="Maximum number of iterations for each split in Bisecting K-Means. Higher values may result in better clusters but increase computation time."
        )

    k = st.sidebar.slider(
        "Number of Clusters (k)", 
        2, 10, 3,
        help="The number of clusters to form. Increasing k will create more, smaller clusters. Decreasing k will create fewer, larger clusters."
    )

    standardize = st.sidebar.checkbox(
        "Standardize Features", 
        True,
        help="Scales features to have zero mean and unit variance. Recommended when features are on different scales."
    )

    numeric_columns = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    available_features = [col for col in numeric_columns if col != "Customer ID"]
    features_to_use = st.sidebar.multiselect(
        "Features to Use", 
        available_features, 
        default=available_features,
        help="Select the features to use for clustering. More features can capture more complexity but may also introduce noise."
    )

    apply_pca = False
    pca_components = 2
    if len(features_to_use) > 2:
        apply_pca = st.sidebar.checkbox(
            "Apply PCA", 
            False,
            help="Principal Component Analysis reduces the number of features while preserving most of the variance in the data. Useful for visualizing high-dimensional data."
        )
        if apply_pca:
            pca_components = st.sidebar.slider(
                "Number of PCA Components", 
                2, len(features_to_use), 2,
                help="Number of principal components to keep. More components retain more information but increase complexity."
            )
    if features_to_use:
        df_selected = df_features[["Customer ID"] + features_to_use]
        
        if standardize:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df_selected[features_to_use]), columns=features_to_use)
        else:
            df_scaled = df_selected[features_to_use]

        if apply_pca and len(features_to_use) > 1:
            pca = PCA(n_components=min(pca_components, len(features_to_use)))
            df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=[f"PC{i+1}" for i in range(pca.n_components_)])
            features_for_clustering = df_pca.columns.tolist()
        else:
            df_pca = df_scaled
            features_for_clustering = features_to_use

        if len(features_for_clustering) < 2:
            st.error("Please select at least two features for clustering.")
            return

        if algorithm == "K-means":
            model = KMeans(n_clusters=k, random_state=1)
        elif algorithm == "Gaussian Mixture":
            model = GaussianMixture(n_components=k, random_state=1)
        elif algorithm == "Bisecting K-Means":
            model = BisectingKMeans(n_clusters=k, random_state=1, max_iter=max_iter)

        predictions = model.fit_predict(df_pca[features_for_clustering])
        
        df_segmented = df_selected.copy()
        df_segmented['prediction'] = predictions

        st.subheader("Cluster Sizes")
        st.info("This bar chart shows the number of customers in each cluster. Balanced clusters are generally preferred, but uneven sizes can sometimes reveal meaningful segments.")
        cluster_sizes = df_segmented['prediction'].value_counts().sort_index().reset_index()
        cluster_sizes.columns = ['prediction', 'count']
        st.bar_chart(cluster_sizes, x='prediction', y='count')

        if algorithm == "K-means":
            st.subheader("Cluster Centers")
            st.info("These are the average values of each feature for each cluster. They represent the 'typical' member of each cluster. Features with notably different values across clusters can help characterize what makes each cluster unique.")
            if apply_pca:
                center_df = pd.DataFrame(model.cluster_centers_, columns=features_for_clustering)
            else:
                center_df = pd.DataFrame(model.cluster_centers_, columns=features_to_use)
            center_df = center_df.round(2)
            st.dataframe(center_df, use_container_width=True)

        st.subheader("Cluster Analysis")

        if len(features_for_clustering) >= 2:
            st.info("This scatter plot shows how the data points are grouped into clusters. Each point represents a customer, and the color indicates the cluster. Clear separations between colors suggest well-defined clusters.")
            fig = px.scatter(df_pca, x=features_for_clustering[0], y=features_for_clustering[1], color=predictions, 
                             title="Scatterplot Matrix (All Clusters)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough features for a scatter plot. Please select at least two features.")

        st.subheader("Cluster Feature Distributions")

        # Create combined histograms for each cluster
        for cluster in sorted(df_segmented['prediction'].unique()):
            st.markdown(f"### Cluster {cluster}")
            cluster_data = df_segmented[df_segmented['prediction'] == cluster]
            
            fig = go.Figure()
            for feature in features_to_use:
                fig.add_trace(go.Histogram(
                    x=cluster_data[feature],
                    name=feature,
                    opacity=0.75
                ))
            
            fig.update_layout(
                barmode='overlay',
                title=f"Combined Feature Distributions for Cluster {cluster}",
                xaxis_title="Value",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
