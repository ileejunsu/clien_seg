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
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, to_date, lit, max
import logging
import sys
from scipy import stats
@st.cache_resource
def setup_java():
    java_url = "https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz"
    java_tar = "openjdk-11.0.2_linux-x64_bin.tar.gz"
    java_dir = "jdk-11.0.2"

    if not os.path.exists(java_dir):
        #st.info("Downloading and setting up Java. This may take a few minutes...")
        
        # Download Java
        try:
            if shutil.which('wget'):
                os.system(f"wget {java_url}")
            else:
                #st.warning("wget not found. Using urllib for download.")
                urllib.request.urlretrieve(java_url, java_tar)
        except Exception as e:
            st.error(f"Failed to download Java: {str(e)}")
            return None

        # Extract Java
        try:
            with tarfile.open(java_tar, "r:gz") as tar:
                tar.extractall()
            
            # Clean up the tar file
            os.remove(java_tar)
        except Exception as e:
            st.error(f"Failed to extract Java: {str(e)}")
            return None

    # Set JAVA_HOME and update PATH
    java_home = os.path.abspath(java_dir)
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"

    #st.success("Java setup completed successfully.")
    return java_home

# Call the function to set up Java
java_home = setup_java()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['JAVA_HOME'] = java_home
logger.info(f"JAVA_HOME set to: {java_home}")


# Initialize SparkSession
@st.cache_resource
def get_spark_session():
    try:
        # Add these lines to help debug
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

# Load data from Parquet
@st.cache_data
def load_data():
    df_features = spark.read.parquet("project/df_features.parquet")
    reference_date = df_features.agg(max("LastPurchasesDate")).collect()[0][0]
    reference_date_lit = lit(reference_date)
    df_features = df_features.withColumn("DaysSinceLastPurchase", datediff(to_date(reference_date_lit), "LastPurchasesDate"))
    df_features = df_features.withColumn("DaysSinceFirstPurchase", datediff(to_date(reference_date_lit), "FirstPurchasesDate"))
    df_features = df_features.drop("LastPurchasesDate", "FirstPurchasesDate")
    
    # Convert to Pandas DataFrame before returning
    return df_features.toPandas()

df_features = load_data()

def main():
    st.title("Interactive Customer Segmentation")

    # Sidebar for parameter selection
    st.sidebar.header("Clustering Settings")

    # Clustering Algorithm Selection
    algorithm = st.sidebar.selectbox(
        "Algorithm", 
        ("K-means", "Gaussian Mixture", "Bisecting K-Means"),
        help="K-means: Partitions data into k clusters. Simple and fast, but assumes spherical clusters.\n\n"
             "Gaussian Mixture: Probabilistic model, allows for clusters of different shapes and sizes.\n\n"
             "Bisecting K-Means: Hierarchical variant of K-means, can handle non-globular clusters better."
    )

    # Max Iterations for Bisecting K-Means
    max_iter = 20
    if algorithm == "Bisecting K-Means":
        max_iter = st.sidebar.slider(
            "Max Iterations", 
            5, 50, 20,
            help="Maximum number of iterations for each split in Bisecting K-Means. Higher values may result in better clusters but increase computation time."
        )

    # Number of Clusters (k)
    k = st.sidebar.slider(
        "Number of Clusters (k)", 
        2, 10, 3,
        help="The number of clusters to form. Increasing k will create more, smaller clusters. Decreasing k will create fewer, larger clusters."
    )

    # Feature Scaling Option
    standardize = st.sidebar.checkbox(
        "Standardize Features", 
        True,
        help="Scales features to have zero mean and unit variance. Recommended when features are on different scales."
    )

    # Select Features
    numeric_columns = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    available_features = [col for col in numeric_columns if col != "Customer ID"]
    features_to_use = st.sidebar.multiselect(
        "Features to Use", 
        available_features, 
        default=available_features,
        help="Select the features to use for clustering. More features can capture more complexity but may also introduce noise."
    )

    # Apply PCA if selected (with dynamic range adjustment)
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
    # Clustering Execution
    if features_to_use:
        # Filter and Prepare Features
        df_selected = df_features[["Customer ID"] + features_to_use]
        
        # Standardize Features
        if standardize:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df_selected[features_to_use]), columns=features_to_use)
        else:
            df_scaled = df_selected[features_to_use]

        # Apply PCA
        if apply_pca and len(features_to_use) > 1:
            pca = PCA(n_components=min(pca_components, len(features_to_use)))
            df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=[f"PC{i+1}" for i in range(pca.n_components_)])
            features_for_clustering = df_pca.columns.tolist()
        else:
            df_pca = df_scaled
            features_for_clustering = features_to_use

        # Check if we have enough features for clustering
        if len(features_for_clustering) < 2:
            st.error("Please select at least two features for clustering.")
            return

        # Perform Clustering
        if algorithm == "K-means":
            model = KMeans(n_clusters=k, random_state=1)
        elif algorithm == "Gaussian Mixture":
            model = GaussianMixture(n_components=k, random_state=1)
        elif algorithm == "Bisecting K-Means":
            model = BisectingKMeans(n_clusters=k, random_state=1, max_iter=max_iter)

        # Fit model and predict clusters
        predictions = model.fit_predict(df_pca[features_for_clustering])
        
        # Add predictions to the dataframe
        df_segmented = df_selected.copy()
        df_segmented['prediction'] = predictions

        # Display Results
        st.subheader("Cluster Sizes")
        st.info("This bar chart shows the number of customers in each cluster. Balanced clusters are generally preferred, but uneven sizes can sometimes reveal meaningful segments.")
        cluster_sizes = df_segmented['prediction'].value_counts().sort_index().reset_index()
        cluster_sizes.columns = ['prediction', 'count']
        st.bar_chart(cluster_sizes, x='prediction', y='count')

        # Calculate and display cluster centers (for K-means)
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

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                if apply_pca:
                    x_col, y_col = 'PC1', 'PC2'
                    x_label, y_label = 'First Principal Component', 'Second Principal Component'
                else:
                    x_col = features_for_clustering[0]
                    y_col = features_for_clustering[1]
                    x_label, y_label = x_col, y_col

                fig = px.scatter(df_pca, x=x_col, y=y_col, color=predictions, 
                                 labels={'color': 'Cluster'},
                                 title="Scatterplot of Clusters")
                fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

            with col2:
                st.write("X-axis")
                x_col = st.selectbox("", features_for_clustering, key="x_axis")

            with col3:
                st.write("Y-axis")
                y_col = st.selectbox("", [f for f in features_for_clustering if f != x_col], key="y_axis")

            # Update the plot based on selection
            if not apply_pca:
                fig.update_traces(x=df_pca[x_col], y=df_pca[y_col])
                fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough features for a scatter plot. Please select at least two features.")

        # Cluster Profiles
        st.subheader("Cluster Profiles")
        st.info("These tabs provide detailed information about each cluster. You can explore the characteristics of each customer segment here.")
        
        # Ensure correct cluster order
        cluster_order = sorted(df_segmented['prediction'].unique())
        
        # Use tabs for each cluster
        tabs = st.tabs([f"Cluster {i}" for i in cluster_order])
        for tab, cluster_id in zip(tabs, cluster_order):
            with tab:
                st.subheader(f"Cluster {cluster_id}")
        
                # Filter data for the specific cluster
                cluster_data = df_segmented[df_segmented['prediction'] == cluster_id]
        
                # Summary statistics
                st.info("This table shows summary statistics for each feature within this cluster. Compare these values to other clusters to understand what makes this segment unique.")
               # Calculate summary statistics
                cluster_stats = cluster_data[features_to_use].describe()
                
                # Replace NaN with 0 and round to integers
                cluster_stats = cluster_stats.fillna(0).round(0).astype(int)


                # Transpose the dataframe for display
                cluster_stats = cluster_stats.transpose()

                st.table(cluster_stats)
        
                # Visualize distributions
                # Inside your cluster tab loop:
                st.write("Distribution of Features:")
                st.info("Select features from the multiselect box to view their distributions within this cluster.")

                # Create a multiselect box for feature selection with a unique key
                selected_features = st.multiselect(
                    "Select features to display:",
                    features_to_use,
                    default=[features_to_use[0]],
                    key=f"feature_select_{cluster_id}"
                )
                
                if selected_features:
                    fig = go.Figure()
                
                    for feature in selected_features:
                        # Normalize the data
                        data = cluster_data[feature]
                        data_norm = (data - data.min()) / (data.max() - data.min())
                        
                        # Calculate kernel density estimation
                        kde = stats.gaussian_kde(data_norm)
                        x_range = np.linspace(0, 1, 1000)
                        y_kde = kde(x_range)
                
                        # Add line trace
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_kde,
                            mode='lines',
                            name=feature,
                            line=dict(width=2)
                        ))
                
                    # Update layout for better readability
                    fig.update_layout(
                        title="Distribution of Selected Features (Normalized)",
                        xaxis_title="Normalized Value",
                        yaxis_title="Density",
                        legend_title="Features",
                        height=500,
                        hovermode="x unified"
                    )
                
                    # Update axes to automatically zoom and fit the data
                    fig.update_xaxes(range=[0, 1])
                    fig.update_yaxes(title_text="Density")
                
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                
                    # Add a note about normalization
                    st.info("Note: Feature values have been normalized to a 0-1 scale for comparison. The lines represent the probability density function for each feature.")
                else:
                    st.warning("Please select at least one feature to display.")


        # Evaluate Clustering (Silhouette Score)
        silhouette = silhouette_score(df_pca[features_for_clustering], predictions)
        st.subheader(f"Silhouette Score ({algorithm})")
        st.info("The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. Scores range from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.")
        st.metric("Score", round(silhouette, 3))

    else:
        st.warning("Please select at least one feature for clustering.")

if __name__ == "__main__":
    main()