# **Autism Detection from fMRI Using Graph Neural Networks (GIN)**

This project builds a **graph-based pipeline** to detect Autism Spectrum Disorder (ASD) using **functional MRI (fMRI) data**. Each brain is modeled as a **functional connectivity graph**, where:

* **Nodes = brain regions (ROIs)**
* **Edges = functional connections derived using Graph-Regularized LASSO**
* **Edge weights = strength of connectivity**

Using these subject-specific brain graphs, a **Graph Isomorphism Network (GIN)** is trained to classify subjects as **ASD or Healthy Control**.

---

## 🔍 **Key Steps**

### **1. Preprocessing fMRI Time-Series**

* Extract ROI-wise time-series
* Standardize data
* Compute correlation structure

### **2. Graph Construction (Graph-Regularized LASSO)**

* Encourages **sparse + smooth** connectivity
* Uses graph Laplacian to reduce noise
* Produces subject-specific weighted adjacency matrices

### **3. Graph Creation (PyTorch Geometric)**

* Node features = identity matrix
* Edges = non-zero connections
* Edge attributes = connection strengths
* Labels = ASD / Control

### **4. GNN Model**

* **GINConv** layers for expressive graph learning
* **Global pooling** to aggregate ROI information
* Fully connected layer for classification

---

## 🧠 **Why GNNs?**

* Brain is naturally a **network**, not an image
* GNNs capture **inter-regional dependencies**
* More robust to noise than CNN-based approaches
* Handles sparse and irregular connectivity patterns

---

## 📦 **Tech Stack**

* Python
* NumPy, SciPy
* PyTorch
* PyTorch Geometric
* scikit-learn
* NetworkX (visualization)

---

## 📈 **Results**

* 88.89% accuracy using sparse functional connectivity graphs
* Highlights key disrupted connections in ASD
* Shows promise for neuroimaging-based diagnosis


