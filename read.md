E-commerce Customer Segmentation Project
=====================================

Table of Contents
Project Overview
Dataset
Tools and Technologies
Project Structure
Workflow Steps
Step-by-Step Instructions
Project Overview
This project aims to segment customers based on their purchasing behavior to enable targeted marketing strategies, enhance customer satisfaction, and boost sales.

Dataset
The Online Retail Dataset from the UCI Machine Learning Repository is used for this project.
The dataset contains over 500,000 transactions from a UK-based online retailer between 2010 and 2011.
Tools and Technologies
Programming Language: Python
Libraries:
Data Manipulation: pandas, numpy
Visualization: matplotlib, seaborn, plotly
Machine Learning: scikit-learn
Dimensionality Reduction: PCA, t-SNE
Environment: Jupyter Notebook
Project Structure
data/: contains the online retail dataset (online_retail.csv)
notebooks/: contains Jupyter notebooks for each step of the workflow
src/: contains Python scripts for data preprocessing, feature engineering, and clustering
reports/: contains the final customer segmentation report (customer_segmentation_report.pdf)
requirements.txt: lists the required libraries and dependencies
Workflow Steps
Data Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Clustering
Visualization
Step-by-Step Instructions
Step 1: Data Preprocessing
Open the 01_data_preprocessing.ipynb notebook in Jupyter Notebook.
Import the necessary libraries: pandas, numpy, and matplotlib.
Load the online retail dataset (online_retail.csv) into a Pandas dataframe.
Handle missing values and duplicates.
Convert data types appropriately.
Remove canceled transactions.
Save the preprocessed data to a new CSV file (preprocessed_data.csv).
Step 2: Exploratory Data Analysis (EDA)
Open the 02_eda.ipynb notebook in Jupyter Notebook.
Import the necessary libraries: pandas, matplotlib, and seaborn.
Load the preprocessed data (preprocessed_data.csv) into a Pandas dataframe.
Analyze sales trends over time.
Identify top-selling products.
Examine customer purchasing patterns.
Step 3: Feature Engineering
Open the 03_feature_engineering.ipynb notebook in Jupyter Notebook.
Import the necessary libraries: pandas and numpy.
Load the preprocessed data (preprocessed_data.csv) into a Pandas dataframe.
Calculate Recency, Frequency, and Monetary (RFM) values for each customer.
Normalize RFM features.
Step 4: Clustering
Open the 04_clustering.ipynb notebook in Jupyter Notebook.
Import the necessary libraries: scikit-learn and matplotlib.
Load the feature-engineered data (feature_engineered_data.csv) into a Pandas dataframe.
Determine the optimal number of clusters using the Elbow Method and Silhouette Score.
Apply K-Means clustering algorithm.
Assign cluster labels to customers.
Step 5: Visualization
Open the 05_visualization.ipynb notebook in Jupyter Notebook.
Import the necessary libraries: matplotlib, seaborn, and plotly.
Load the clustered data (clustered_data.csv) into a Pandas dataframe.
Use PCA or t-SNE for dimensionality reduction.
Visualize clusters in 2D space.
Create bar plots to compare RFM values across clusters.
Step 6: Report Generation
Open the customer_segmentation_report.ipynb notebook in Jupyter Notebook.
Import the necessary libraries: matplotlib and seaborn.
Load the clustered data (clustered_data.csv) into a Pandas dataframe.
Generate the final customer segmentation report (customer_segmentation_report.pdf).