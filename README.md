# 🧠 MRI Brain Tumor Analysis & Diagnostics

An advanced **Streamlit-based medical imaging platform** designed for automated brain tumor detection, segmentation, and clinical feature extraction from MRI scans. This project bridges the gap between traditional Computer Vision and modern Deep Learning.

---

## 🚀 Key Features

### 1. 🎯 Tumor Segmentation (Hybrid Approach)
Compare state-of-the-art Deep Learning with traditional CV algorithms:
- **AI-Powered (U-Net)**: High-precision neural network segmentation.
- **Traditional CV Suite**:
  - **Otsu's Thresholding**: Global intensity-based segmentation.
  - **K-Means Clustering**: Unsupervised region grouping.
  - **Contour Detection**: Advanced skull stripping followed by ROI extraction.
  - **GrabCut**: Iterative foreground/background estimation.

### 2. 📊 Clinical Feature Extraction (Metrics)
Automatically extracts quantitative data from detected tumors:
- **Shape Analysis**: Area, Perimeter, and Circularity (regularity score).
- **Spatial Metrics**: Bounding box dimensions (W x H) and Center-of-Mass (X, Y).
- **Intensity Profile**: Mean and Maximum brightness within the tumor region.

### 3. 🧪 Robust Preprocessing
- **Multi-stage Filtering**: Gaussian Blur for high-frequency noise and Median filtering for salt-and-pepper noise removal.
- **Standardization**: Automated resizing and normalization for consistent model performance.

### 4. 📈 Classification
- Predicts presence of tumor with high-confidence probability scores.
- Powered by deep learning architectures (ResNet/EfficientNet/GoogleNet backends).

### 5. 🎨 Modern Dashboard
- Interactive UI with custom CSS (Glassmorphism inspired).
- Real-time comparison side-by-side.
- Metric cards for clinical review.

---

## 📂 Project Structure

```text
📦 brain-tumor-analysis
├── app.py                # Main Streamlit Dashboard logic
├── animations.js         # Frontend UI enhancements
├── requirements.txt      # Dependency manifest
├── models/               # Pre-trained H5 weights (External)
│   ├── BrainTumor_Segmentation_Unet.h5
│   └── BrainTumor_classification_model.h5
├── Classification/       # Training notebooks for classification
├── Segmentation/         # Training notebooks for U-Net
├── test_photos/          # Sample MRI scans for testing
└── predictions/          # Saved masks and results
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone and Install
```bash
# Clone the repository
git clone https://github.com/Abdo0213/Brain-Tumor-Analysis.git
cd Brain-Tumor-Analysis

# Install requirements
pip install -r requirements.txt
```

### 2️⃣ Download Models
The pre-trained models are required for the "AI" features. Download them from the link below and place them in the root directory:
🔗 [Download Pre-trained Models](https://drive.google.com/drive/folders/1c7QqNMkogn2zRylDNzwlzGxPXvBX-BT7?usp=sharing)

### 3️⃣ Run Application
```bash
streamlit run app.py
```

---

## 🔬 How It Works

1. **Upload**: Drag and drop a `.tif`, `.jpg`, or `.png` MRI slice.
2. **Select Task**: Choose between **Segmentation** (detect region) or **Classification** (diagnose).
3. **Analyze**: 
   - View the U-Net mask vs. Traditional methods.
   - Read the extracted **Tumor Metrics** (Area, Circularity, etc.).
4. **Export**: (Optional) Download the resulting masks for further analysis.

---

## 📌 Requirements

- **Python**: 3.11+
- **Frameworks**: Streamlit, TensorFlow, OpenCV, Scikit-Image, NumPy, Pillow.

---

## 👨‍💻 Development Team
Developed by **Team CV**.

---
*Note: This tool is for research purposes and should not be used as a primary diagnostic tool without clinical validation.*

