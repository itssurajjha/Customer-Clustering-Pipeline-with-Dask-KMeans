import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.cluster import KMeans
from kneed import KneeLocator
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import sys
from sklearn.decomposition import PCA  # <-- use sklearn PCA, NOT dask_ml PCA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = r'D:/python_journy/linkdinpythoncourse/Ex_Files_Intermediate_Python_Non_Programmers_2024/coustomer_clusturing/data/online_retail.csv'
PARQUET_PATH = DATA_PATH.replace('online_retail.csv', 'preprocessed_data.parquet')
LABELS_PATH = PARQUET_PATH.replace('.parquet', '_labels.csv')


def preprocess_dask(df: dd.DataFrame) -> dd.DataFrame:
    try:
        df = df.dropna(subset=['event_time', 'price'])
        df['event_time'] = dd.to_datetime(df['event_time'], errors='coerce')
        df = df.dropna(subset=['event_time'])

        df['hour'] = df['event_time'].dt.hour
        df['day'] = df['event_time'].dt.day
        df['month'] = df['event_time'].dt.month

        df['category_code'] = df['category_code'].fillna('Unknown').astype('category')
        df['event_type'] = df['event_type'].fillna('Unknown').astype('category')

        df = df.categorize(columns=['event_type', 'category_code'])
        df['event_type'] = df['event_type'].cat.as_known().cat.codes
        df['category_code'] = df['category_code'].cat.as_known().cat.codes

        features = ['price', 'hour', 'day', 'month']
        return df[features]
    except Exception as e:
        logging.error(f"Error in preprocess_dask: {e}")
        raise


def auto_tune_kmeans(data, k_range=(2, 10)):
    logging.info("Auto-tuning K using Elbow Method...")
    distortions = []
    for k in range(k_range[0], k_range[1] + 1):
        model = KMeans(n_clusters=k, init_max_iter=10, oversampling_factor=2, random_state=42)
        model.fit(data)
        distortions.append(model.inertia_)
    k_vals = list(range(k_range[0], k_range[1] + 1))
    knee = KneeLocator(k_vals, distortions, curve="convex", direction="decreasing")
    best_k = knee.knee or k_vals[np.argmin(distortions)]  # fallback
    logging.info(f"Best K={best_k}")
    final_model = KMeans(n_clusters=best_k, init_max_iter=20, oversampling_factor=2, random_state=42)
    final_model.fit(data)
    return final_model, best_k


def save_pdf_report(df, labels, pca_data, cluster_centers, filename='cluster_report.pdf'):
    df_report = df.copy()
    df_report['cluster'] = labels

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    # Scatter matrix of original features colored by cluster
    fig_plotly = px.scatter_matrix(df_report,
                                   dimensions=['price', 'hour', 'day', 'month'],
                                   color='cluster',
                                   title="Cluster Visualization")
    img_bytes = pio.to_image(fig_plotly, format='png', width=800, height=600, scale=2)

    with PdfPages(filename) as pdf:
        # Add plotly scatter matrix image to PDF
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        img = plt.imread(io.BytesIO(img_bytes))
        ax.imshow(img)
        pdf.savefig(fig)
        plt.close(fig)

        # Add cluster counts table
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('tight')
        ax.axis('off')
        table_data = [["Cluster", "Count"]]
        for cluster_id, count in cluster_counts.items():
            table_data.append([str(cluster_id), str(count)])
        table = ax.table(cellText=table_data,
                         loc='center',
                         cellLoc='center',
                         colWidths=[0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 3)
        ax.set_title('Cluster Counts', fontsize=18, pad=20)
        pdf.savefig(fig)
        plt.close(fig)

        # Add PCA plot showing clusters in 2D
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax.set_title('PCA Projection of Clusters')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        # Optionally, plot cluster centers in PCA space if cluster_centers given
        if cluster_centers is not None:
            pca_centers = PCA(n_components=2).fit_transform(cluster_centers)
            ax.scatter(pca_centers[:, 0], pca_centers[:, 1], c='red', marker='X', s=100, label='Centroids')
            ax.legend()

        pdf.savefig(fig)
        plt.close(fig)

    logging.info(f"Saved clustering report as PDF: {filename}")


def run_pipeline():
    from dask.distributed import Client
    client = Client(n_workers=4, threads_per_worker=2, memory_limit="4GB")
    logging.info(f"Dask dashboard available at: {client.dashboard_link}")

    try:
        logging.info("Reading CSV...")
        df = dd.read_csv(DATA_PATH, blocksize="50MB")

        logging.info("Preprocessing...")
        df_processed = preprocess_dask(df)

        logging.info("Scaling...")
        scaler = StandardScaler()
        X = df_processed.to_dask_array(lengths=True)
        X_scaled = scaler.fit_transform(X)
        df_scaled = dd.from_dask_array(X_scaled, columns=df_processed.columns)

        logging.info("Saving preprocessed data...")
        df_scaled.to_parquet(PARQUET_PATH, write_index=False, engine='pyarrow', compression='snappy', overwrite=True)

        df_clustering = dd.read_parquet(PARQUET_PATH)

        logging.info("Auto-tuning KMeans...")
        model, best_k = auto_tune_kmeans(df_clustering)

        logging.info("Computing final labels...")
        labels = model.labels_.compute()
        pd.Series(labels).to_csv(LABELS_PATH, index=False, header=['cluster'])
        logging.info("Saving final labels...")

        logging.info("Generating report from sample...")
        sample_df = df_clustering.sample(frac=0.05).compute()
        sample_scaled = sample_df.to_numpy()

        logging.info("Computing PCA...")
        pca = PCA(n_components=2)  # sklearn PCA, on in-memory numpy array
        pca_data = pca.fit_transform(sample_scaled)

        logging.info("Creating report PDF...")
        save_pdf_report(sample_df, labels[:len(sample_df)], pca_data, model.cluster_centers_)

    except Exception as e:
        logging.error("Pipeline failed:")
        logging.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    run_pipeline()
