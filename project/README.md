Customer Segmentation App

Overview
--------
This Streamlit application performs customer segmentation using various clustering algorithms on retail data. It allows users to interactively explore customer segments based on their purchasing behavior.

Features
--------
- Data preprocessing and feature engineering
- Multiple clustering algorithms: K-means, Gaussian Mixture, Bisecting K-Means
- Interactive parameter selection
- Visual representation of clusters
- Detailed cluster analysis and profiling

Prerequisites
-------------
- Python 3.7+
- PySpark
- Streamlit
- Pandas
- Scikit-learn
- Plotly

Installation
------------
1. Clone the repository:
   git clone https://github.com/ileejunsu/Customer_Segmentation.git
   cd Customer_Segmentation

2. Install required packages:
   pip install -r requirements.txt

Data Preparation
----------------
1. Download the 'Online Retail II' dataset:
   - Go to the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
   - Download the 'Online Retail II.xlsx' file
   - Convert the Excel file to CSV format and name it 'online_retail_II.csv'
   - Place the 'online_retail_II.csv' file in the project's root directory

2. Run the data preparation script: python retail_prep.py
   - This will generate 'df_features.parquet', which is used by the main application.


Usage
-----
1. Start the Streamlit app:
   streamlit run app.py

2. Open your web browser and navigate to http://localhost:8501.

3. Use the sidebar to select clustering parameters and features.

4. Explore the resulting clusters through visualizations and detailed profiles.

File Descriptions
-----------------
- app.py: Main Streamlit application file
- retail_prep.py: Data preprocessing and feature engineering script
- requirements.txt: List of Python dependencies
- setup.sh: Setup script for Streamlit configuration (used in deployment)
- Procfile: Specifies the commands that are executed by the app on startup (used in deployment)

Deployment
----------
This app is ready for deployment on platforms like Heroku. The `setup.sh` and `Procfile` are included for this purpose.

Contributing
------------
Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

License
-------
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
-------
For any queries or suggestions, please open an issue on this GitHub repository.

Happy Clustering!