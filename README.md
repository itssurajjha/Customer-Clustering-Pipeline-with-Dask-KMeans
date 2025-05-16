# 🧠 Customer Segmentation with KMeans Clustering

Welcome to my unsupervised learning project, where I deep-dived into **customer behavior analysis** using **KMeans Clustering**, **Dask**, and a powerful **EDA pipeline** for insights. This is a modular, performance-aware machine learning pipeline designed for clarity, scalability, and storytelling through data.

---

## 🚀 Project Highlights

* ⚙️ **Scalable Data Pipeline** using `Dask` for handling large datasets
* 🧠 **KMeans Clustering** with auto-cluster tuning (Elbow + Silhouette)
* 📊 **Exploratory Data Analysis (EDA)** with `Plotly`, `Seaborn`, and `Matplotlib`
* 🧾 Detailed **cluster-wise statistics** and distribution plots
* 🧠 Suitable for **real-world customer behavior segmentation**

---

## 📂 Project Structure

```
customer-clustering-project/
├── data/
│   ├── preprocessed_data.parquet         # Cleaned feature set
│   └── preprocessed_data_labels.csv      # Cluster labels
├── eda.py                                # EDA and visual storytelling
├── data_preprocessing.py                 # Full pipeline: load → clean → cluster
├── requirements.txt                      # Reproducibility
├── README.md                             # You're here!
└── .gitignore
```

---

## 🛠️ Technologies Used

| Tool             | Purpose                      |
| ---------------- | ---------------------------- |
| **Python**       | Core programming language    |
| **Pandas**       | Data manipulation            |
| **Dask**         | Scalable parallel processing |
| **Scikit-learn** | KMeans, scaling, metrics     |
| **Plotly**       | Interactive visualization    |
| **Seaborn**      | Statistical plotting         |
| **Matplotlib**   | Supplementary visuals        |

---

## 📈 Key Insights

* Cluster segmentation reveals distinct purchasing behaviors based on:

  * **Price sensitivity**
  * **Time of purchase (hour/day/month)**
* Data visualizations highlight outlier segments and seasonal trends
* Provides a foundation for:

  * Targeted marketing
  * Inventory planning
  * Customer lifecycle analysis

---

## ⚡ How to Run

> Ensure Python 3.8+ and dependencies from `requirements.txt` are installed.

```bash
git clone https://github.com/yourusername/customer-clustering-project.git
cd customer-clustering-project
python eda.py
```

---

## 📌 TODO / Future Enhancements

* [ ] Integrate cluster summary reporting to PDF/HTML
* [ ] Build a dashboard using **Streamlit** or **Dash**
* [ ] Add GCP/AWS data pipeline integration

---

## 🧠 About Me

I'm a coding maniac & data storyteller. I build ML workflows that are clean, powerful, and production-aware.
If you're hiring or collaborating, [let's connect on LinkedIn](https://www.linkedin.com/in/yourprofile)!

---

## ⭐ Star this project if it helped you!
