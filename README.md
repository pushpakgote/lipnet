# Fraud Detection And Recommendation

#### Checkout live implementation at [https://frauddetectionandrecommendation.streamlit.app/](https://frauddetectionandrecommendation.streamlit.app/) 

This repository contains Python scripts for data cleaning, unsupervised clustering analysis, and product recommendations based on an Amazon product dataset. The repository is organized into three main files:

1. [**cleaning.ipynb**](https://github.com/pushpakgote/fraud_detection_and_recommendation/blob/main/cleaning.ipynb) : This notebook is used to clean and preprocess the Amazon product dataset.
2. [**clustering.ipynb**](https://github.com/pushpakgote/fraud_detection_and_recommendation/blob/main/clustering.ipynb) : This notebook performs unsupervised clustering analysis on the cleaned dataset using K-Means clustering and includes graphical representations of the analysis.
3. [**recommendation.ipynb**](https://github.com/pushpakgote/fraud_detection_and_recommendation/blob/main/recommendation.ipynb) : This notebook generates product recommendations based on clusters and cosine similarity between products.

## Cleaning Data (cleaning.ipynb)

The `cleaning.ipynb` notebook performs the following data cleaning tasks:

- Reads the Amazon product dataset (`Amazon-Products.csv`).
- Removes unnecessary columns.
- Handles missing values and outliers.
- Cleans and transforms specific columns (e.g., 'ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'discount%').
- Saves the cleaned data to a new CSV file (`cleaned_amazon_products.csv`).

## Unsupervised Clustering Analysis (clustering.ipynb)

The `clustering.ipynb` notebook performs unsupervised clustering analysis on the cleaned dataset:

- Reads the cleaned dataset (`cleaned_amazon_products.csv`).
- Standardizes numeric features for clustering.
- Applies K-Means clustering with a specified number of clusters.
- Redefines clusters for better product recommendations.
- Generates graphical representations of the clustering analysis, including a silhouette score plot.
- Saves the clustering model and scaler for future use (`kmeans_model.pkl` and `standard_scaler_model.pkl`).
- Provides functions to load the clustering model and predict clusters for new data.
- provides function to detect if product is suspicious or not.
## Product Recommendations (recommendation.ipynb)

The `recommendation.ipynb` notebook generates product recommendations based on cosine similarity:

- Reads the cleaned dataset with clusters and without duplicates (`cleaned_amazon_products_with_cluster_without_duplicates.csv`).
- Converts categorical columns to numeric values.
- Scales numeric features.
- Utilizes CountVectorizer to transform product names into vectors.
- Defines functions for generating recommendations.
- Calculates cosine similarity between products and provides recommendations for similar products.

## Usage

To use these scripts, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the necessary Python packages installed (e.g., pandas, scikit-learn, matplotlib).
3. Run the scripts in the following order:
   - `extract_zip.py` to extract compressed csv files.
   - `cleaning.ipynb` to clean and preprocess the data.
   - `clustering.ipynb` to perform unsupervised clustering analysis, including graphical analysis.
   - `recommendation.ipynb` to generate product recommendations.

## Note

- The clustering model (`kmeans_model.pkl`) and scaler (`standard_scaler_model.pkl`) generated in the clustering step are required for the recommendation step.

- The clusters are made using ratings and discount % of the products. These 2 features are used for checking suspicious products and recommend better products in better cluster.

- The recommendation script can be used to obtain product recommendations for specific products by providing their features to the `recommendation` function.

- This is an unsupervised learning problem, and the clustering analysis in `clustering.py` includes graphical representations, making it easy to see clusters.

- The provided clustering and recommendation scripts are designed for educational purposes and can be further customized and extended based on your specific requirements.

