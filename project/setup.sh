#!/bin/bash
# setup.sh

# Set environment variables
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_HOME=$CONDA_PREFIX/lib/python3.9/site-packages/pyspark
export PYSPARK_PYTHON=$CONDA_PREFIX/bin/python
export PYSPARK_DRIVER_PYTHON=$CONDA_PREFIX/bin/python
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=$JAVA_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Print environment variables for debugging
echo "JAVA_HOME: $JAVA_HOME"
echo "SPARK_HOME: $SPARK_HOME"
echo "PYSPARK_PYTHON: $PYSPARK_PYTHON"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check if Java is installed and its version
if command -v java &> /dev/null; then
    java -version
else
    echo "Java is not installed or not in PATH"
fi

# Check if pyspark is installed
pip list | grep pyspark

# Check if 'ps' command is available
if command -v ps &> /dev/null; then
    echo "'ps' command is available"
else
    echo "'ps' command is not found"
fi

# Run Streamlit
streamlit run app.py