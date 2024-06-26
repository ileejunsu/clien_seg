#!/bin/bash

# setup.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Update package list and install necessary packages
log "Updating package list and installing necessary packages..."
sudo apt-get update
sudo apt-get install -y procps openjdk-11-jdk

# Find and set Java home
log "Setting up Java environment..."
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
if [ -z "$JAVA_HOME" ]; then
    log "ERROR: Unable to find Java installation. Please install Java 11."
    exit 1
fi
log "JAVA_HOME set to: $JAVA_HOME"

# Set up Spark environment
log "Setting up Spark environment..."
export SPARK_HOME=$CONDA_PREFIX/lib/python3.9/site-packages/pyspark
if [ ! -d "$SPARK_HOME" ]; then
    log "ERROR: Spark installation not found. Please ensure PySpark is installed."
    exit 1
fi
log "SPARK_HOME set to: $SPARK_HOME"

# Set Python paths for PySpark
export PYSPARK_PYTHON=$CONDA_PREFIX/bin/python
export PYSPARK_DRIVER_PYTHON=$CONDA_PREFIX/bin/python

# Update PATH and LD_LIBRARY_PATH
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=$JAVA_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Print environment variables for debugging
log "Environment variables:"
log "JAVA_HOME: $JAVA_HOME"
log "SPARK_HOME: $SPARK_HOME"
log "PYSPARK_PYTHON: $PYSPARK_PYTHON"
log "PATH: $PATH"
log "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check if Java is installed and its version
if command -v java &> /dev/null; then
    log "Java version:"
    java -version
else
    log "ERROR: Java is not installed or not in PATH"
    exit 1
fi

# Check if pyspark is installed
log "Checking PySpark installation..."
if pip list | grep pyspark; then
    log "PySpark is installed"
else
    log "ERROR: PySpark is not installed"
    exit 1
fi

# Check if 'ps' command is available
if command -v ps &> /dev/null; then
    log "'ps' command is available"
else
    log "ERROR: 'ps' command is not found"
    exit 1
fi

# Check for the presence of necessary files
log "Checking for necessary files..."
if [ ! -f "app.py" ]; then
    log "ERROR: app.py not found in the current directory"
    exit 1
fi

if [ ! -f "df_features.parquet" ]; then
    log "WARNING: df_features.parquet not found. Make sure it's available before running the app."
fi

# Set Streamlit configuration
log "Setting Streamlit configuration..."
mkdir -p ~/.streamlit
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

# Run Streamlit
log "Starting Streamlit app..."
streamlit run app.py