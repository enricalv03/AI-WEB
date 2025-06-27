# 👕 AI Clothing Web - Machine Learning Fashion Classifier

A modern web application that demonstrates the practical application of **K-Nearest Neighbors (KNN)** and **K-means** clustering algorithms for clothing classification and retrieval. This project provides an interactive interface to explore how these fundamental machine learning algorithms work in real-world scenarios.

## 🎯 Project Overview

This university project showcases two essential machine learning algorithms through a clothing classification system:

- **K-means Clustering**: For color-based classification and grouping
- **K-Nearest Neighbors (KNN)**: For shape and clothing type classification

The web interface allows users to visually understand how these algorithms filter and categorize clothing items based on different criteria, making abstract ML concepts tangible and interactive.

## ✨ Features

### 🔍 Smart Filtering System
- **Color-based filtering**: Uses K-means clustering to identify dominant colors
- **Shape/Type classification**: Employs KNN to categorize clothing types (shirts, dresses, jeans, etc.)
- **Combined filtering**: Merge both algorithms for precise results
- **Confidence scoring**: See how certain the algorithms are about their predictions

### 🎨 Modern User Interface
- **Responsive design**: Works on desktop, tablet, and mobile devices
- **Dark/Light mode**: Toggle between themes for comfortable viewing
- **Grid/List views**: Choose your preferred layout
- **Real-time results**: Instant feedback as you adjust filters
- **Visual color palette**: Interactive color selection interface

### 📊 Educational Value
- **Algorithm visualization**: See how K-means and KNN process data
- **Performance metrics**: Understand classification accuracy
- **Interactive learning**: Experiment with different parameters
- **Real dataset**: Work with actual clothing images and labels

## 🏗️ Architecture

```
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API        │    │   ML Algorithms │
│                 │    │                    │    │                 │
│ • HTML/CSS/JS   │◄──►│ • RESTful API      │◄──►│ • K-means       │
│ • Responsive UI │    │ • Image Processing │    │ • KNN           │
│ • Interactive   │    │ • Data Utils       │    │ • Classification│
└─────────────────┘    └────────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python 3.7+** 
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Web browser** (Chrome, Firefox, Safari, or Edge)

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AI-Clothing-Web.git
cd AI-Clothing-Web
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Dataset Structure

Ensure your project structure looks like this:

```
AI-Clothing-Web/
├── images/
│   ├── train/          # Training images (3000+ images)
│   ├── test/           # Test images (1000+ images)
│   ├── gt.json         # Ground truth labels
│   └── gt_reduced.json # Reduced dataset labels
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
└── ...
```

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Access the Web Interface

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## 💻 Usage Examples

### Basic Color Filtering
1. Select colors from the palette (e.g., "Blue", "Red")
2. Click "Search" to see items with those colors
3. View results with confidence scores

### Shape Classification
1. Choose a clothing type from the dropdown (e.g., "Shirts")
2. Observe how KNN classifies similar shapes
3. Experiment with different K values

### Combined Filtering
1. Select both colors and clothing type
2. See how both algorithms work together
3. Compare results with single-algorithm filtering

## 🔬 Machine Learning Algorithms

### K-means Clustering (Color Classification)

```python
# Simplified example of how K-means works in the project
kmeans = KMeans(X, K=5)  # 5 color clusters
kmeans.fit()
colors = get_colors(kmeans.centroids)
```

**Purpose**: Groups pixels by color similarity to identify dominant colors in clothing items.

**Parameters**:
- `K=5`: Number of color clusters
- `tolerance`: Convergence threshold
- `max_iter`: Maximum iterations

### K-Nearest Neighbors (Shape Classification)

```python
# Simplified example of KNN implementation
knn = KNN(train_data, train_labels)
predictions = knn.predict(test_data, k=5)
```

**Purpose**: Classifies clothing types based on shape similarity to training examples.

**Parameters**:
- `k=5`: Number of neighbors to consider
- `metric`: Distance calculation method (euclidean, manhattan)

## 📁 Project Structure

```
AI-Clothing-Web/
├── 📄 app.py                 # Flask web server & API routes
├── 🧠 Kmeans.py             # K-means clustering implementation
├── 🔍 KNN.py                # K-Nearest Neighbors implementation
├── 🏷️ my_labeling.py        # Image retrieval & classification logic
├── 🛠️ utils_data.py         # Data processing utilities
├── 🧪 TestCases_*.py        # Unit tests for algorithms
├── 📊 requirements.txt       # Python dependencies
├── 📝 README.md             # Project documentation
├── 🌐 templates/
│   └── index.html           # Main web interface
├── 🎨 static/
│   ├── css/style.css        # Styling & responsive design
│   ├── js/app.js           # Frontend JavaScript logic
│   ├── img/                # UI images & icons
│   └── temp_imgs/          # Temporary processed images
├── 🖼️ images/
│   ├── train/              # Training dataset (~3,750 images) "Recomended"
│   ├── test/               # Test dataset (~1,000 images) "Recomended"
│   ├── gt.json             # Complete ground truth labels
│   └── gt_reduced.json     # Subset labels for testing
└── 📊 output/              # Generated results & analysis
```

## 🔧 API Documentation

### Filter by Color
```http
POST /api/filter/color
Content-Type: application/json

{
  "colors": ["blue", "red"],
  "threshold": 0.15
}
```

### Filter by Shape
```http
POST /api/filter/shape
Content-Type: application/json

{
  "shape": "Shirts",
  "k": 5,
  "metric": "euclidean"
}
```

### Combined Filtering
```http
POST /api/filter/combined
Content-Type: application/json

{
  "colors": ["blue"],
  "shape": "Shirts",
  "min_coverage": 0.15,
  "k": 5
}
```

## 📊 Dataset Information

- **Total Images**: ~4,750 clothing items
- **Categories**: Dresses, Flip Flops, Heels, Jeans, Sandals, Shirts, Shorts, Socks
- **Colors**: Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow
- **Format**: JPG images (60x80 pixels, resized for processing)
- **Labels**: JSON format with shape and color annotations

## 🧪 Testing the Algorithms

Run the test suites to verify algorithm performance:

```bash
# Test K-means clustering
python TestCases_kmeans.py

# Test KNN classification  
python TestCases_knn.py
```

## ⚙️ Configuration Options

Modify algorithm parameters in `app.py`:

```python
# K-means settings
COLOR_THRESHOLD = 0.15  # Minimum color coverage
K_CLUSTERS = 5          # Number of color clusters

# KNN settings  
DEFAULT_K = 5           # Default neighbors count
DISTANCE_METRIC = 'euclidean'  # Distance calculation
```

## 🎓 Educational Goals

This project demonstrates:

1. **Practical ML Application**: Real-world use of fundamental algorithms
2. **Algorithm Comparison**: Visual comparison between unsupervised (K-means) and supervised (KNN) learning
3. **Parameter Tuning**: How different parameters affect results
4. **Data Preprocessing**: Image processing and feature extraction
5. **Web Development**: Creating ML-powered web applications

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'flask'"
**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Issue**: "No images found" or empty results
**Solution**: Verify the images directory structure and ensure gt.json exists

**Issue**: Slow performance
**Solution**: The first run may be slower due to model initialization. Subsequent searches are faster due to caching.

## 🤝 Contributing

This is a university project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📚 Further Learning

To deepen your understanding:

- **K-means**: Study clustering algorithms and their applications
- **KNN**: Explore lazy learning and distance metrics
- **Computer Vision**: Learn about image processing and feature extraction
- **Web Development**: Flask framework and RESTful API design

## 📄 License

This project is created for educational purposes as part of university coursework. Feel free to use it for learning and non-commercial purposes.

## 👥 Authors
- **Names**: Enric Alvarez, Federico, Arnau
- **Course**: Introduction to Artificial Intelligence
- **Institution**: Autonomous University of Barcelona

---

**Happy Learning! 🎓** Explore the fascinating world of machine learning through this interactive clothing classifier! 
