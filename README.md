 # Unsupervised Face Clustering & Explainable AI (XAI) Interpretation
This project implements an end-to-end unsupervised learning pipeline to cluster facial images from the Labeled Faces in the Wild (LFW) dataset. Beyond standard clustering, it incorporates Explainable AI (XAI) techniques to interpret the latent features (such as lighting, pose, and expression), driving the grouping decisions without utilizing ground-truth identity labels.

📌 Project Overview
The primary objective is to organize facial data based on intrinsic visual similarities and provide transparency into the "black box" of unsupervised algorithms.

Dimensionality Reduction: Compressed 16,000+ pixel features into compact embeddings while retaining 95% variance.

Clustering: Evaluated multiple algorithms to group faces based on latent structures.

Interpretability (XAI): Applied post-hoc analysis to understand why specific faces were clustered together using surrogate modeling and feature attribution.

🛠️ Methodology & Pipeline
1. Data Preprocessing & Balancing
Dataset: LFW (Deepfunneled), resized to 250x250 grayscale.

Imbalance Handling: Implemented a hybrid sampling strategy to ensure fair representation:


Undersampling: Applied Tomek Links to clean overlapping boundary points.



Oversampling: Used ADASYN to generate synthetic samples for underrepresented individuals.


2. Unsupervised Learning (Clustering)

Feature Extraction: Principal Component Analysis (PCA) reduced dimensions from 62,500 to ~248 components.

Algorithms Evaluated:


K-Means (k=7): Achieved the best separation and cohesion.

Hierarchical Clustering: Used Ward’s linkage for dendrogram analysis.


DBSCAN: Tested for density-based grouping (found unsuitable for this high-dimensional feature space).


Self-Organizing Maps (SOM): Utilized for topological visualization and similarity search.

3. Knowledge Representation & XAI
To interpret the clusters, a surrogate Random Forest classifier was trained on the cluster labels to enable feature attribution:


SHAP (SHapley Additive exPlanations): Quantified the global impact of specific PCA components on cluster formation.


LIME (Local Interpretable Model-agnostic Explanations): Explained individual predictions to detect local variances.


Pixel-Level Heatmaps: Back-projected important components to visualize facial regions (e.g., eyes, mouth) influencing the grouping.

📂 Repository Structure
Bash

├── data-preparation.ipynb  # Dataset curation, filtering, and initial processing
├── model-dev.ipynb         # Sampling (Tomek/ADASYN), PCA, and Clustering implementation
├── visual.ipynb            # Interactive 3D visualizations, Surrogate Modeling, and XAI (SHAP/LIME) analysis
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
📊 Key Results
Metric	K-Means	Hierarchical	DBSCAN
Silhouette Score (Higher is better)	0.040	0.031	0.024
Davies-Bouldin (Lower is better)	3.35	3.97	2.98
Calinski-Harabasz (Higher is better)	252.14	199.71	46.84

Export to Sheets

XAI Insights:

Clusters were primarily driven by lighting conditions and head pose rather than distinct facial identities.

Feature heatmaps confirmed that the algorithms focused heavily on high-contrast regions like the eyes and mouth to separate groups.

🚀 Usage
Clone the repo:

Bash

git clone [https://github.com/UmarAftab174/unsupervised-face-clustering-and-analysis-using-lfw-dataset/](https://github.com/UmarAftab174/unsupervised-face-clustering-and-analysis-using-lfw-dataset/tree/main)
Install dependencies:

Bash

pip install -r requirements.txt
Run the analysis:

Start with data-preparation.ipynb to process the raw LFW images.

Run model-dev.ipynb to train the clustering models.

Use visual.ipynb to generate SHAP plots and LIME explanations.

🧩 Technologies Used
Core: Python, NumPy, Pandas

ML/Clustering: Scikit-learn, Imbalanced-learn (ADASYN/Tomek)

XAI: SHAP, LIME

Visualization: Matplotlib, Seaborn