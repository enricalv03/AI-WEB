from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
import os
import json
import numpy as np
import shutil
from my_labeling import Retrieval_by_color, Retrieval_by_shape, Retrieval_combined
from utils_data import read_dataset, read_extended_dataset, crop_images
from utils_data import convert_to_grayscale, downscale_images, analyze_color_coverage, verify_dominant_colors
from Kmeans import KMeans, get_colors
from KNN import KNN
import time

app = Flask(__name__)

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Load data and initialize models
print("Loading dataset...")
train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
    test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

classes = list(set(list(train_class_labels) + list(test_class_labels)))

# Initialize KNN for shape classification
print("Initializing KNN model...")
train_gray = np.mean(train_imgs, axis=3) if train_imgs.ndim == 4 else train_imgs
knn = KNN(train_gray, train_class_labels)

# Initialize KMeans for color clustering - use sampling for better efficiency
print("Initializing KMeans model for color clustering...")
# Use a sample of the training images to fit KMeans (for better performance)
sample_size = min(500, train_imgs.shape[0])  # Use at most 500 images
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(train_imgs.shape[0], size=sample_size, replace=False)
sample_imgs = train_imgs[sample_indices]

# Reshape the sample to pixels
X = sample_imgs.reshape(-1, 3)
# Further sample pixels if there are too many
if X.shape[0] > 100000:  # Limit to 100k pixels
    pixel_indices = np.random.choice(X.shape[0], size=100000, replace=False)
    X = X[pixel_indices]

print(f"Fitting KMeans on {X.shape[0]} pixel samples...")
kmeans = KMeans(X, K=5)
kmeans.fit()
centroids_colors = get_colors(kmeans.centroids)

# Directory to save temporary images
TEMP_IMG_DIR = 'static/temp_imgs'
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# Global variable to track the last search type
last_search_type = None
last_search_params = None

# Function to clear temporary images
def clear_temp_images():
    try:
        # First try to remove the entire directory and recreate it
        if os.path.exists(TEMP_IMG_DIR):
            shutil.rmtree(TEMP_IMG_DIR)
        os.makedirs(TEMP_IMG_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error removing directory: {str(e)}")
        try:
            for file in os.listdir(TEMP_IMG_DIR):
                file_path = os.path.join(TEMP_IMG_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        except Exception as e:
            print(f"Error listing directory: {str(e)}")

# Convert image to file and save
def save_image(img, filename):
    from PIL import Image
    
    # Add timestamp to prevent browser caching
    timestamp = int(time.time())
    filename_with_timestamp = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
    
    img_pil = Image.fromarray(img)
    filepath = os.path.join(TEMP_IMG_DIR, filename_with_timestamp)
    img_pil.save(filepath)
    return filepath

# Config parameters
COLOR_THRESHOLD = 0.15  # Minimum color coverage (15%)
VERIFY_COLORS = True    # Perform secondary verification
USE_CONFIDENCE = True   # Include confidence scores in results

# API endpoint for color filtering
@app.route('/api/filter/color', methods=['POST'])
def filter_by_color():
    global last_search_type, last_search_params
    
    # Clear previous temporary images
    clear_temp_images()
    
    # Reset last search info
    last_search_type = 'color'
    
    data = request.json
    colors = data.get('colors', [])
    threshold = data.get('threshold', COLOR_THRESHOLD)
    
    # Store search params
    last_search_params = {
        'colors': colors,
        'threshold': threshold
    }
    
    if not colors:
        return jsonify({'error': 'No colors provided'}), 400
    
    # Convert to list if a single color is provided
    if isinstance(colors, str):
        colors = [colors]
    
    # Ensure colors is always a list
    if not isinstance(colors, list):
        return jsonify({'error': 'Colors must be provided as a list'}), 400
    
    # Log the request for debugging
    print(f"Color filter request: {colors}, threshold: {threshold}")
    
    try:
        result_imgs, result_coverages, result_indices = Retrieval_by_color(
            test_imgs, test_color_labels, colors, 
            min_coverage=threshold, 
            verify_colors=VERIFY_COLORS
        )
        
        # Save images and collect metadata
        results = []
        for i, idx in enumerate(result_indices):
            img_path = save_image(result_imgs[i], f'color_{i}.jpg')
            # Extract just the filename from the path
            img_filename = os.path.basename(img_path)
            
            # Get item dimensions
            height, width = result_imgs[i].shape[:2] if len(result_imgs[i].shape) >= 2 else (0, 0)
            
            # Calculate a quality score based on coverage
            quality_score = min(100, int(result_coverages[i] * 150))  # Scale up coverage for better UX
            
            results.append({
                'image_url': f'/static/temp_imgs/{img_filename}',
                'class': test_class_labels[idx],
                'colors': list(test_color_labels[idx]),
                'coverage': float(result_coverages[i]),
                'confidence': float(result_coverages[i]) if USE_CONFIDENCE else None,
                'metadata': {
                    'id': f"item_{idx}",
                    'dimensions': f"{width}x{height}",
                    'quality_score': quality_score,
                    'matched_colors': colors
                }
            })
        
        print(f"Found {len(results)} results for color query: {colors}")
        return jsonify({
            'results': results,
            'count': len(results),
            'query': {
                'type': 'color',
                'colors': colors,
                'threshold': threshold
            }
        })
    except Exception as e:
        import traceback
        print(f"Error in color filtering: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# API endpoint for shape filtering
@app.route('/api/filter/shape', methods=['POST'])
def filter_by_shape():
    global last_search_type, last_search_params
    
    # Clear previous temporary images
    clear_temp_images()
    
    # Reset last search info
    last_search_type = 'shape'
    
    data = request.json
    shape = data.get('shape', '')
    k = data.get('k', 5)  # Number of neighbors for KNN
    metric = data.get('metric', 'euclidean')  # Distance metric
    
    # Store search params
    last_search_params = {
        'shape': shape,
        'k': k,
        'metric': metric
    }
    
    if not shape:
        return jsonify({'error': 'No shape provided'}), 400
    
    try:
        result_imgs, result_indices = Retrieval_by_shape(test_imgs, test_class_labels, shape)
        
        # Optionally get confidence scores for shape classification
        confidences = []
        if USE_CONFIDENCE:
            # Convert matching test images to grayscale for KNN prediction
            gray_imgs = convert_to_grayscale(result_imgs)
            _, confidences = knn.predict_with_confidence(gray_imgs, k, metric)
        else:
            # If not using confidence, create placeholder values
            confidences = [0.75] * len(result_imgs)  # Default 75% confidence
        
        # Save images and collect metadata
        results = []
        for i, idx in enumerate(result_indices):
            img_path = save_image(result_imgs[i], f'shape_{i}.jpg')
            # Extract just the filename from the path
            img_filename = os.path.basename(img_path)
            
            # Get item dimensions
            height, width = result_imgs[i].shape[:2] if len(result_imgs[i].shape) >= 2 else (0, 0)
            
            results.append({
                'image_url': f'/static/temp_imgs/{img_filename}',
                'class': test_class_labels[idx],
                'colors': list(test_color_labels[idx]),
                'confidence': float(confidences[i]),
                'metadata': {
                    'id': f"item_{idx}",
                    'dimensions': f"{width}x{height}",
                    'quality_score': min(100, int(confidences[i] * 100)),
                    'shape_match': shape
                }
            })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': {
                'type': 'shape',
                'shape': shape,
                'metric': metric
            }
        })
    except Exception as e:
        import traceback
        print(f"Error in shape filtering: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# API endpoint for combined filtering
@app.route('/api/filter/combined', methods=['POST'])
def filter_combined():
    global last_search_type, last_search_params
    
    # Clear previous temporary images
    clear_temp_images()
    
    # Reset last search info
    last_search_type = 'combined'
    
    data = request.json
    shape = data.get('shape', '')
    colors = data.get('colors', [])
    threshold = data.get('threshold', COLOR_THRESHOLD)
    
    # Store search params
    last_search_params = {
        'shape': shape,
        'colors': colors,
        'threshold': threshold
    }
    
    if not shape or not colors:
        return jsonify({'error': 'Shape and colors must be provided'}), 400
    
    # Convert to list if a single color is provided
    if isinstance(colors, str):
        colors = [colors]
    
    # Ensure colors is always a list
    if not isinstance(colors, list):
        return jsonify({'error': 'Colors must be provided as a list'}), 400
    
    # Log the request for debugging
    print(f"Combined filter request: shape={shape}, colors={colors}, threshold={threshold}")
    
    # Set a more permissive threshold for combined searches
    combined_threshold = threshold * 0.8  # 20% more permissive
    
    try:
        result_imgs, result_info, result_indices = Retrieval_combined(
            test_imgs, test_class_labels, test_color_labels, shape, colors,
            min_coverage=combined_threshold,
            verify_colors=VERIFY_COLORS
        )
        
        # Save images and collect metadata
        results = []
        for i, idx in enumerate(result_indices):
            img_path = save_image(result_imgs[i], f'combined_{i}.jpg')
            # Extract just the filename from the path
            img_filename = os.path.basename(img_path)
            
            # Get item dimensions
            height, width = result_imgs[i].shape[:2] if len(result_imgs[i].shape) >= 2 else (0, 0)
            
            # Calculate a combined score
            color_score = min(100, int(result_info[i]['color_coverage'] * 150))
            shape_score = min(100, int(result_info[i]['confidence'] * 100))
            combined_score = (color_score + shape_score) // 2
            
            results.append({
                'image_url': f'/static/temp_imgs/{img_filename}',
                'class': test_class_labels[idx],
                'colors': list(test_color_labels[idx]),
                'coverage': float(result_info[i]['color_coverage']),
                'confidence': float(result_info[i]['confidence']),
                'matched_colors': result_info[i].get('matched_colors', []),
                'metadata': {
                    'id': f"item_{idx}",
                    'dimensions': f"{width}x{height}",
                    'quality_score': combined_score,
                    'shape_match': shape,
                    'color_match': result_info[i].get('matched_colors', [])
                }
            })
        
        print(f"Found {len(results)} results for combined query: shape={shape}, colors={colors}")
        return jsonify({
            'results': results,
            'count': len(results),
            'query': {
                'type': 'combined',
                'shape': shape,
                'colors': colors,
                'threshold': threshold
            }
        })
    except Exception as e:
        import traceback
        print(f"Error in combined filtering: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Get available classes and colors
@app.route('/api/options', methods=['GET'])
def get_options():
    all_colors = set()
    for colors in train_color_labels:
        for color in colors:
            all_colors.add(color.lower())
    
    return jsonify({
        'shapes': sorted(classes),
        'colors': sorted(list(all_colors)),
        'metrics': ['euclidean', 'manhattan', 'minkowski']
    })

# API endpoint for threshold configuration
@app.route('/api/config', methods=['GET', 'POST'])
def configure():
    global COLOR_THRESHOLD, VERIFY_COLORS, USE_CONFIDENCE
    
    if request.method == 'POST':
        data = request.json
        COLOR_THRESHOLD = data.get('color_threshold', 0.15)
        VERIFY_COLORS = data.get('verify_colors', True)
        USE_CONFIDENCE = data.get('use_confidence', True)
        
        return jsonify({
            'status': 'success',
            'config': {
                'color_threshold': COLOR_THRESHOLD,
                'verify_colors': VERIFY_COLORS,
                'use_confidence': USE_CONFIDENCE
            }
        })
    else:
        return jsonify({
            'color_threshold': COLOR_THRESHOLD,
            'verify_colors': VERIFY_COLORS,
            'use_confidence': USE_CONFIDENCE
        })

# API endpoint for reset
@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset endpoint to clear server-side state and temporary files"""
    global last_search_type, last_search_params
    
    try:
        # Clear temporary images
        clear_temp_images()
        
        # Reset search state
        last_search_type = None
        last_search_params = None
        
        return jsonify({
            'status': 'success',
            'message': 'Server state reset successfully'
        })
    except Exception as e:
        import traceback
        print(f"Error in reset: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Serve the main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001) 