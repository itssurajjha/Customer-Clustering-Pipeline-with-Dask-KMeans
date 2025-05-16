import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
PARQUET_PATH = r'D:/python_journy/linkdinpythoncourse/Ex_Files_Intermediate_Python_Non_Programmers_2024/coustomer_clusturing/data/preprocessed_data.parquet'
LABELS_PATH = PARQUET_PATH.replace('.parquet', '_labels.csv')
OUTPUT_DIR = "eda_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    logging.info("Loading processed data and cluster labels...")
    df = pd.read_parquet(PARQUET_PATH)
    labels = pd.read_csv(LABELS_PATH)['cluster']
    df['cluster'] = labels
    return df


def plot_distributions(df):
    logging.info("Generating boxplots and KDE distribution plots per feature by cluster...")
    features = ['price', 'hour', 'day', 'month']
    for feature in features:
        # Boxplot with Plotly
        fig_box = px.box(df, x='cluster', y=feature, color='cluster',
                         title=f'{feature.capitalize()} Distribution by Cluster (Boxplot)')
        box_html_path = os.path.join(OUTPUT_DIR, f'{feature}_boxplot.html')
        fig_box.write_html(box_html_path)
        logging.info(f"Saved boxplot to {box_html_path}")

        # KDE plot with Seaborn + Matplotlib (saved as PNG)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x=feature, hue='cluster', fill=True, common_norm=False, alpha=0.5, palette='tab10')
        plt.title(f'{feature.capitalize()} Distribution by Cluster (KDE)')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Density')
        plt.tight_layout()
        kde_png_path = os.path.join(OUTPUT_DIR, f'{feature}_kde.png')
        plt.savefig(kde_png_path)
        plt.close()
        logging.info(f"Saved KDE plot to {kde_png_path}")


def plot_cluster_counts(df):
    logging.info("Plotting cluster sizes...")
    counts = df['cluster'].value_counts().sort_index()
    fig = go.Figure(data=[go.Bar(x=counts.index.astype(str), y=counts.values, marker_color='teal')])
    fig.update_layout(title="Number of Customers per Cluster", xaxis_title="Cluster", yaxis_title="Count")
    counts_html_path = os.path.join(OUTPUT_DIR, 'cluster_counts.html')
    fig.write_html(counts_html_path)
    logging.info(f"Saved cluster counts bar chart to {counts_html_path}")


def plot_pairplot(df):
    logging.info("Generating seaborn pairplot (sampling 10%)...")
    sample_df = df.sample(frac=0.1, random_state=42)
    sns.pairplot(sample_df, hue='cluster', palette='tab10')
    plt.suptitle("Pairplot of Features by Cluster", y=1.02)
    pairplot_png_path = os.path.join(OUTPUT_DIR, 'pairplot.png')
    plt.savefig(pairplot_png_path)
    plt.close()
    logging.info(f"Saved pairplot to {pairplot_png_path}")


def plot_correlation_heatmaps(df):
    logging.info("Generating correlation heatmaps per cluster...")
    features = ['price', 'hour', 'day', 'month']
    clusters = df['cluster'].unique()
    for cluster in clusters:
        cluster_df = df[df['cluster'] == cluster][features]
        corr = cluster_df.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Heatmap - Cluster {cluster}')
        plt.tight_layout()
        corr_png_path = os.path.join(OUTPUT_DIR, f'correlation_heatmap_cluster_{cluster}.png')
        plt.savefig(corr_png_path)
        plt.close()
        logging.info(f"Saved correlation heatmap for cluster {cluster} to {corr_png_path}")


def summary_stats(df):
    logging.info("Computing and displaying cluster centroid statistics...")
    features = ['price', 'hour', 'day', 'month']
    centroids = df.groupby('cluster')[features].mean()
    print("\nCluster Centroid (Mean) Feature Values:\n")
    print(centroids)
    
    # Plot centroids for visual overview
    centroids_reset = centroids.reset_index()
    fig = px.bar(centroids_reset.melt(id_vars='cluster'), 
                 x='variable', y='value', color='cluster', barmode='group',
                 title='Cluster Centroid Mean Feature Values')
    centroids_html_path = os.path.join(OUTPUT_DIR, 'cluster_centroids.html')
    fig.write_html(centroids_html_path)
    logging.info(f"Saved cluster centroids bar chart to {centroids_html_path}")


def main():
    df = load_data()
    plot_distributions(df)
    plot_cluster_counts(df)
    plot_pairplot(df)
    plot_correlation_heatmaps(df)
    summary_stats(df)
    logging.info("EDA complete. All reports saved in folder: %s", OUTPUT_DIR)


if __name__ == '__main__':
    main()
