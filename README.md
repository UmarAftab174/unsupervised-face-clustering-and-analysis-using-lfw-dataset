# **Unsupervised Face Clustering & Explainable AI (XAI) Interpretation**

This project builds an end-to-end unsupervised learning pipeline to cluster facial images from the **Labeled Faces in the Wild (LFW)** dataset. Instead of relying on identity labels, the system groups faces based purely on visual similarity—then explains *why* those groups emerged using modern XAI techniques.

The goal is to make unsupervised systems more transparent by uncovering the hidden visual factors (lighting, expression, pose) that drive clustering behavior.

---

## ⭐ **Project Highlights**

* ✔️ 16,000+ pixel features compressed into compact PCA embeddings
* ✔️ Multiple clustering algorithms benchmarked (KMeans, Hierarchical, DBSCAN, SOM)
* ✔️ XAI analysis using SHAP, LIME, and PCA component back-projection
* ✔️ Synthetic rebalancing to correct dataset imbalance (ADASYN + Tomek Links)
* ✔️ Pixel-level heatmaps showing which facial regions influence clustering

---

## 🧠 **Pipeline Overview**

### **1. Data Preparation & Balancing**

* **Dataset**: LFW (deepfunneled), resized to **250×250 grayscale**
* **Imbalance Handling**:

  * **Tomek Links undersampling** to clean boundary noise
  * **ADASYN oversampling** to generate synthetic examples for rare classes

This ensures more stable clustering by reducing class dominance from highly photographed individuals.

---

### **2. Dimensionality Reduction**

Faces (250×250 = 62,500 pixels) are transformed into compact embeddings using:

* **Principal Component Analysis (PCA)**
* Retained **95% variance**, resulting in **~248 principal components**

These components represent latent facial factors such as lighting gradients, pose variations, and structural symmetry.

---

### **3. Unsupervised Clustering**

Multiple algorithms were evaluated:

| Algorithm    | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
| ------------ | ------------ | ---------------- | ------------------- |
| **K-Means**  | **0.040**    | **3.35**         | **252.14**          |
| Hierarchical | 0.031        | 3.97             | 199.71              |
| DBSCAN       | 0.024        | 2.98             | 46.84               |

**K-Means with k=7** offered the most stable grouping in the latent space.

Additionally:

* **Hierarchical Clustering** helped visualize dendrogram relationships
* **Self-Organizing Maps (SOM)** provided a topology-preserving projection
* **DBSCAN** proved unsuitable due to high-dimensional sparsity

---

### **4. Explainable AI (XAI) Analysis**

A **Random Forest surrogate model** was trained on cluster labels to enable interpretation.

#### 🔍 **Global Explanations (SHAP)**

* Identified which PCA components most influenced clustering
* Revealed that lighting direction, contrast patterns, and head tilt dominated decisions

#### 🎯 **Local Explanations (LIME)**

* Explained individual image cluster assignments
* Highlighted subtle distinctions between similar-looking faces

#### 🎨 **Pixel-Level Heatmaps**

PCA components were reconstructed back into image space to reveal:

* Eyes
* Mouth shape
* Forehead lighting
* Shadow/color gradients

These were the strongest drivers of clustering.

---

## 📁 **Repository Structure**

```
📂 unsupervised-face-clustering-and-analysis-using-lfw-dataset/
│
├── data-preparation.ipynb       # Filtering, balancing, preprocessing
├── model-dev.ipynb              # PCA, clustering algorithms, performance metrics
├── visual.ipynb                 # SHAP, LIME, heatmaps, interactive visualization
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---

## 🧠 **Key Insights from XAI**

* Clusters correlate more strongly with **lighting direction**, **pose**, and **contrast** than with true identity.
* High-contrast regions—**eyes, eyebrows, mouth**—carry the highest explanatory weight.
* Subtle PCA components encode facial geometry, while early components capture lighting and shading trends.
* Unsupervised models form *perceptual* clusters, not identity clusters.

---

## 🚀 **How to Use**

### **1. Clone the Repository**

```bash
git clone https://github.com/UmarAftab174/unsupervised-face-clustering-and-analysis-using-lfw-dataset/
```

### **2. Install Requirements**

```bash
pip install -r requirements.txt
```

### **3. Run the Pipeline**

1. **data-preparation.ipynb** for cleaning & balancing
2. **model-dev.ipynb** for PCA + clustering
3. **visual.ipynb** for SHAP/LIME and visualization

---

## 🧩 **Tech Stack**

**Core:** Python, NumPy, Pandas
**Clustering:** scikit-learn, MiniSom
**Sampling:** imbalanced-learn (ADASYN, Tomek Links)
**XAI:** SHAP, LIME
**Visualization:** Matplotlib, Seaborn