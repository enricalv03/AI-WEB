__authors__ = []
__group__ = '80'

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from tqdm import tqdm
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, Plot3DCloud, visualize_k_means
from utils_data import convert_to_grayscale, downscale_images, analyze_color_coverage, verify_dominant_colors
from Kmeans import KMeans, distance, get_colors
from KNN import KNN


# QUALITATIVE ANALYSIS FUNCTIONS

def Retrieval_by_color(imgs, labels, query, min_coverage=0.15, verify_colors=True):
    """
    Retrieves images containing specified colors with coverage threshold
    :param imgs: array of images
    :param labels: color labels for each image
    :param query: color(s) to search for (string or list)
    :param min_coverage: minimum coverage threshold (0-1)
    :param verify_colors: whether to perform secondary color verification
    :return: filtered images, coverage percentages, and indices
    """
    if imgs is None or len(imgs) == 0:
        raise ValueError("Image list cannot be empty")
    if labels is None or len(labels) == 0:
        raise ValueError("Labels list cannot be empty")
    if query is None or (isinstance(query, list) and len(query) == 0):
        raise ValueError("Query cannot be empty")
    
    if isinstance(query, str):
        query = [query]
    
    result_imgs = []
    result_percentages = []
    result_indices = []
    
    print(f"Searching for images with colors: {', '.join(query)} (min coverage: {min_coverage*100:.0f}%)")
    
    # Create simple color to RGB mapping for verification
    color_map = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
        'orange': [255, 165, 0],
        'purple': [128, 0, 128],
        'pink': [255, 192, 203],
        'brown': [165, 42, 42],
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'gray': [128, 128, 128],
        'grey': [128, 128, 128]  # Add grey as alias for gray
    }
    
    for i in progress_indicator(range(len(labels)), desc="Searching images"):
        colors = labels[i]
        image_colors = [color.lower() for color in colors]
        
        # Check if ANY of the query colors are in the image (not ALL)
        # This allows finding images that contain at least one of the queried colors
        matches = [q.lower() in image_colors for q in query]
        
        if any(matches):
            if verify_colors:
                verified_colors, coverage_percents = verify_dominant_colors(imgs[i], colors, color_map, min_coverage)
                verified_colors_lower = [c.lower() for c in verified_colors]

                verified_matches = [q.lower() in verified_colors_lower for q in query]
                
                # If none of the query colors are verified, skip this image
                if not any(verified_matches):
                    continue
                    
                # Calculate total coverage for matched colors
                matched_coverage = 0
                matched_count = 0
                
                for q in query:
                    q_lower = q.lower()
                    for color, coverage in coverage_percents.items():
                        if color.lower() == q_lower:
                            matched_coverage += coverage
                            matched_count += 1
                
                if matched_count == 0:
                    continue  # Skip if no coverage
                    
                result_imgs.append(imgs[i])
                result_indices.append(i)
                result_percentages.append(matched_coverage / matched_count)  # Average coverage of matched colors
            else:
                # Estimate color percentage using basic method
                percentage = 0
                matched_count = 0
                
                for q in query:
                    q_lower = q.lower()
                    if q_lower in image_colors:
                        count = sum(1 for color in image_colors if color == q_lower)
                        percentage += count / len(image_colors)
                        matched_count += 1
                
                # Only include images where average color coverage is above threshold
                if matched_count > 0 and (percentage / matched_count) >= min_coverage:
                    result_imgs.append(imgs[i])
                    result_indices.append(i)
                    result_percentages.append(percentage / matched_count)
    
    if result_percentages:
        sorted_indices = np.argsort(result_percentages)[::-1]
        sorted_imgs = [result_imgs[i] for i in sorted_indices]
        sorted_percentages = [result_percentages[i] for i in sorted_indices]
        sorted_original_indices = [result_indices[i] for i in sorted_indices]
        
        print(f"Found {len(sorted_imgs)} images matching the color query with sufficient coverage")
        return np.array(sorted_imgs), sorted_percentages, sorted_original_indices
    
    print("No images found matching the color query with sufficient coverage")
    return np.array([]), [], []


def Retrieval_by_shape(imgs, labels, query):
    if imgs is None or len(imgs) == 0:
        raise ValueError("Image list cannot be empty")
    if labels is None or len(labels) == 0:
        raise ValueError("Labels list cannot be empty")
    if not query or not isinstance(query, str):
        raise ValueError("Shape query must be a non-empty string")
    
    query = query.lower()
    
    result_imgs = []
    result_indices = []
    
    print(f"Searching for images with shape/clothing type: {query}")
    
    for i in progress_indicator(range(len(labels)), desc="Searching images"):
        shape = labels[i]
        if shape.lower() == query:
            result_imgs.append(imgs[i])
            result_indices.append(i)
    
    print(f"Found {len(result_imgs)} images matching the shape query")
    return np.array(result_imgs), result_indices


def Retrieval_combined(imgs, shape_labels, color_labels, shape_query, color_query, min_coverage=0.15, verify_colors=True):
    """
    Retrieves images matching both shape and color criteria with coverage threshold
    :param imgs: array of images
    :param shape_labels: shape labels for each image
    :param color_labels: color labels for each image
    :param shape_query: shape to search for
    :param color_query: color(s) to search for
    :param min_coverage: minimum color coverage threshold (0-1)
    :param verify_colors: whether to perform secondary color verification
    :return: filtered images, info, and indices
    """
    if imgs is None or len(imgs) == 0:
        raise ValueError("Image list cannot be empty")
    if shape_labels is None or len(shape_labels) == 0:
        raise ValueError("Shape labels list cannot be empty")
    if color_labels is None or len(color_labels) == 0:
        raise ValueError("Color labels list cannot be empty")
    if not shape_query or not isinstance(shape_query, str):
        raise ValueError("Shape query must be a non-empty string")
    if color_query is None or (isinstance(color_query, list) and len(color_query) == 0):
        raise ValueError("Color query cannot be empty")
    
    # Ensure color_query is a list
    if isinstance(color_query, str):
        color_query = [color_query]
    
    print(f"Searching for {shape_query} with colors: {', '.join(color_query)} " +
          f"(min coverage: {min_coverage*100:.0f}%)")
    
    # CHANGED APPROACH: First filter by shape, then by color
    # This is more efficient when shape filtering is more restrictive
    try:
        # First get all images matching the shape criteria
        shape_imgs, shape_indices = Retrieval_by_shape(imgs, shape_labels, shape_query)
        
        if len(shape_imgs) == 0:
            print(f"No images found matching the shape: {shape_query}")
            return np.array([]), [], []
            
        print(f"Found {len(shape_imgs)} images matching shape: {shape_query}")
        print("Now filtering by color...")
        
        # Create subset of color labels for the shape-filtered images
        shape_color_labels = [color_labels[i] for i in shape_indices]
        
        # Then filter those images by color
        # Use a reduced threshold for combined search to improve results
        adjusted_threshold = min_coverage * 0.8  # 20% lower threshold for combined search
        
        result_imgs = []
        result_info = []
        result_indices = []
        
        # Create simple color to RGB mapping for verification
        color_map = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
            'yellow': [255, 255, 0],
            'orange': [255, 165, 0],
            'purple': [128, 0, 128],
            'pink': [255, 192, 203],
            'brown': [165, 42, 42],
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'gray': [128, 128, 128],
            'grey': [128, 128, 128]
        }
        
        # Manual filtering for better control
        for i in progress_indicator(range(len(shape_imgs)), desc="Filtering by color"):
            img = shape_imgs[i]
            original_idx = shape_indices[i]
            colors = shape_color_labels[i]
            image_colors = [color.lower() for color in colors]
            
            # Check if ANY of the query colors are in the image
            matches = [q.lower() in image_colors for q in color_query]
            
            if any(matches):
                # Perform verification if requested
                if verify_colors:
                    verified_colors, coverage_percents = verify_dominant_colors(img, colors, color_map, adjusted_threshold)
                    verified_colors_lower = [c.lower() for c in verified_colors]
                    
                    # Check which query colors are verified
                    verified_matches = [q.lower() in verified_colors_lower for q in color_query]
                    
                    # If none of the query colors are verified, skip this image
                    if not any(verified_matches):
                        continue
                    
                    # Calculate coverage for matched colors
                    matched_coverage = 0
                    matched_count = 0
                    matched_colors = []
                    
                    for q in color_query:
                        q_lower = q.lower()
                        for color, coverage in coverage_percents.items():
                            if color.lower() == q_lower:
                                matched_coverage += coverage
                                matched_count += 1
                                matched_colors.append(q_lower)
                    
                    if matched_count == 0:
                        continue  # Skip if no coverage
                    
                    result_imgs.append(img)
                    result_indices.append(original_idx)
                    result_info.append({
                        'shape': shape_query,
                        'color_coverage': matched_coverage / matched_count,
                        'confidence': matched_coverage / matched_count,
                        'matched_colors': matched_colors
                    })
                else:
                    # Basic method without verification
                    matched_colors = []
                    for q in color_query:
                        q_lower = q.lower()
                        if q_lower in image_colors:
                            matched_colors.append(q_lower)
                    
                    if matched_colors:
                        result_imgs.append(img)
                        result_indices.append(original_idx)
                        result_info.append({
                            'shape': shape_query,
                            'color_coverage': len(matched_colors) / len(color_query),
                            'confidence': len(matched_colors) / len(color_query),
                            'matched_colors': matched_colors
                        })
        
        print(f"Found {len(result_imgs)} images matching both shape and color criteria")
        return np.array(result_imgs), result_info, result_indices
        
    except Exception as e:
        import traceback
        print(f"Error in combined retrieval: {str(e)}")
        traceback.print_exc()
        return np.array([]), [], []


# QUANTITATIVE ANALYSIS FUNCTIONS

def Kmean_statistics(kmeans_class, imgs, max_K):
    if imgs is None or len(imgs) == 0:
        raise ValueError("Image set cannot be empty")
    if max_K < 2:
        raise ValueError("Maximum K value must be at least 2")
    
    k_values = range(2, max_K + 1)
    wcd_values = []
    iterations = []
    times = []
    
    try:
        if len(imgs.shape) == 4:
            X = imgs.reshape(imgs.shape[0], -1, 3)
            X = X.reshape(-1, 3)
        else:
            X = imgs.reshape(-1, imgs.shape[-1])
    except Exception as e:
        raise ValueError(f"Failed to reshape image data: {str(e)}")
    
    print(f"Analyzing KMeans performance for K values from 2 to {max_K}")

    for k in progress_indicator(k_values, desc="Testing K values"):
        try:
            kmeans = kmeans_class(X, k)
            start_time = time.time()
            kmeans.fit()
            end_time = time.time()
            times.append(end_time - start_time)
            
            wcd_values.append(kmeans.withinClassDistance())
            iterations.append(kmeans.num_iter)
            
            print(f"K={k}: WCD={wcd_values[-1]:.4f}, Iterations={iterations[-1]}, Time={times[-1]:.2f}s")
        except Exception as e:
            print(f"Error analyzing K={k}: {str(e)}")
            wcd_values.append(float('nan'))
            iterations.append(0)
            times.append(0)
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot WCD
        plt.subplot(2, 2, 1)
        plt.plot(k_values, wcd_values, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('Within Class Distance')
        plt.title('WCD vs K Value')
        
        # Plot iterations
        plt.subplot(2, 2, 2)
        plt.plot(k_values, iterations, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('Iterations to Converge')
        plt.title('Iterations vs K Value')
        
        # Plot time
        plt.subplot(2, 2, 3)
        plt.plot(k_values, times, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('Time (seconds)')
        plt.title('Time vs K Value')
        
        # Plot percentage decrease in WCD
        plt.subplot(2, 2, 4)
        perc_decrease = [100 * (wcd_values[i-1] - wcd_values[i]) / wcd_values[i-1] if i > 0 and not np.isnan(wcd_values[i-1]) and not np.isnan(wcd_values[i]) else 0 for i in range(len(wcd_values))]
        plt.plot(k_values, perc_decrease, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('Percentage Decrease in WCD')
        plt.title('WCD Percentage Decrease vs K Value')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing results: {str(e)}")
    
    return {'k_values': k_values, 'wcd': wcd_values, 'iterations': iterations, 'times': times}


def Get_shape_accuracy(predicted_labels, ground_truth_labels):
    if predicted_labels is None or len(predicted_labels) == 0:
        raise ValueError("Predicted labels cannot be empty")
    if ground_truth_labels is None or len(ground_truth_labels) == 0:
        raise ValueError("Ground truth labels cannot be empty")
    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError("Predicted and ground truth labels must have the same length")
    
    try:
        pred_lower = [label.lower() for label in predicted_labels]
        gt_lower = [label.lower() for label in ground_truth_labels]
    except Exception as e:
        raise ValueError(f"Failed to process labels: {str(e)}")
    
    print("Calculating shape classification accuracy...")
    
    correct = sum(1 for p, gt in zip(pred_lower, gt_lower) if p == gt)
    
    accuracy = correct / len(predicted_labels) if len(predicted_labels) > 0 else 0
    
    print(f"Shape accuracy: {accuracy * 100:.2f}% ({correct}/{len(predicted_labels)} correct)")
    return accuracy * 100  # Return as percentage


def Get_color_accuracy(predicted_labels, ground_truth_labels):
    if predicted_labels is None or len(predicted_labels) == 0:
        raise ValueError("Predicted labels cannot be empty")
    if ground_truth_labels is None or len(ground_truth_labels) == 0:
        raise ValueError("Ground truth labels cannot be empty")
    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError("Predicted and ground truth labels must have the same length")
    
    print("Calculating color classification accuracy...")
    
    total_score = 0
    total_precision = 0
    total_recall = 0
    
    for i in progress_indicator(range(len(predicted_labels)), desc="Computing F1 scores"):
        try:
            pred = predicted_labels[i]
            gt = ground_truth_labels[i]
            
            pred_lower = [color.lower() for color in pred]
            gt_lower = [color.lower() for color in gt]
            
            correct_predictions = sum(1 for color in pred_lower if color in gt_lower)
            precision = correct_predictions / len(pred_lower) if len(pred_lower) > 0 else 0
            
            recall = correct_predictions / len(gt_lower) if len(gt_lower) > 0 else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            total_score += f1
            total_precision += precision
            total_recall += recall
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
    
    avg_score = total_score / len(predicted_labels) if len(predicted_labels) > 0 else 0
    avg_precision = total_precision / len(predicted_labels) if len(predicted_labels) > 0 else 0
    avg_recall = total_recall / len(predicted_labels) if len(predicted_labels) > 0 else 0
    
    print(f"Color accuracy (F1 score): {avg_score * 100:.2f}%")
    print(f"Average precision: {avg_precision * 100:.2f}%")
    print(f"Average recall: {avg_recall * 100:.2f}%")
    
    return avg_score * 100  # Return as percentage


# CLASSIFICATION METHOD IMPROVEMENTS

def init_centroids_random(X, K):
    indices = np.random.choice(len(X), K, replace=False)
    return X[indices].copy()


def init_centroids_kmeans_plus_plus(X, K):
    centroids = np.zeros((K, X.shape[1]))
    centroids[0] = X[np.random.randint(len(X))]
    
    for k in range(1, K):
        min_distances = np.min([np.sum((X - centroids[j])**2, axis=1) for j in range(k)], axis=0)
        
        probabilities = min_distances / np.sum(min_distances)
        cumulative_prob = np.cumsum(probabilities)
        r = np.random.rand()
        
        for i, p in enumerate(cumulative_prob):
            if r < p:
                centroids[k] = X[i]
                break
    
    return centroids


def best_k_bic(X, kmeans, max_K):
    k_values = range(2, max_K + 1)
    bic_values = []
    
    n_samples, n_features = X.shape
    
    for k in k_values:
        kmeans.K = k
        kmeans._init_centroids()
        kmeans.fit()
        
        labels = kmeans.labels
        centroids = kmeans.centroids
        
        squared_dist = np.zeros(n_samples)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                squared_dist[labels == i] = np.sum((cluster_points - centroids[i])**2, axis=1)
        
        variance = np.mean(squared_dist) / n_features
        if variance <= 0:
            variance = 1e-10
        
        log_likelihood = -0.5 * np.sum(squared_dist) / variance - \
                         0.5 * n_samples * np.log(2 * np.pi * variance) - \
                         0.5 * n_samples * n_features
        
        n_parameters = k * n_features + 1
        bic = -2 * log_likelihood + n_parameters * np.log(n_samples)
        
        bic_values.append(bic)
    
    best_k = k_values[np.argmin(bic_values)]
    
    return best_k


def best_k_silhouette(X, kmeans, max_K):
    k_values = range(2, max_K + 1)
    silhouette_values = []
    
    for k in k_values:
        kmeans.K = k
        kmeans._init_centroids()
        kmeans.fit()
        
        labels = kmeans.labels
        silhouette_sum = 0
        
        for i in range(len(X)):
            a_cluster = X[labels == labels[i]]
            
            if len(a_cluster) <= 1:
                continue
            
            a_i = np.mean(np.sqrt(np.sum((a_cluster - X[i])**2, axis=1)))
            
            b_i = float('inf')
            for j in range(k):
                if j != labels[i]:
                    b_cluster = X[labels == j]
                    if len(b_cluster) > 0:
                        avg_dist = np.mean(np.sqrt(np.sum((b_cluster - X[i])**2, axis=1)))
                        b_i = min(b_i, avg_dist)
            
            if b_i == float('inf'):
                b_i = 0
            
            if max(a_i, b_i) > 0:
                silhouette_i = (b_i - a_i) / max(a_i, b_i)
            else:
                silhouette_i = 0
            
            silhouette_sum += silhouette_i
        
        silhouette = silhouette_sum / len(X) if len(X) > 0 else 0
        silhouette_values.append(silhouette)
    
    best_k = k_values[np.argmax(silhouette_values)]
    
    return best_k


def progress_indicator(iterable, desc="Processing", leave=True):
    try:
        return tqdm(iterable, desc=desc, leave=leave)
    except Exception:
        total = len(iterable)
        print(f"{desc}:", end="", flush=True)
        
        def text_progress(iterable):
            for i, item in enumerate(iterable):
                sys.stdout.write(f"\r{desc}: {i+1}/{total} [{(i+1)*100//total}%]")
                sys.stdout.flush()
                yield item
            sys.stdout.write("\n" if leave else "\r")
            sys.stdout.flush()
            
        return text_progress(iterable)


def display_menu():
    print("\n" + "="*50)
    print("IMAGE LABELING SYSTEM - MAIN MENU")
    print("="*50)
    print("1. Search by color")
    print("2. Search by clothing type")
    print("3. Search by both color and type")
    print("4. Display gallery")
    print("5. Run performance analysis")
    print("6. Analyze classification accuracy")
    print("0. Exit")
    print("="*50)
    
    while True:
        try:
            choice = int(input("Enter your choice (0-6): "))
            if 0 <= choice <= 6:
                return choice
            else:
                print("Invalid choice. Please enter a number between 0 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_color_input():
    available_colors = [
        "Red", "Green", "Blue", "Yellow", "Black", "White", 
        "Orange", "Purple", "Brown", "Pink", "Gray"
    ]
    
    print("\nAvailable colors:")
    for i, color in enumerate(available_colors):
        print(f"{i+1}. {color}")
    
    print("\nEnter the numbers of colors you want to search for, separated by commas.")
    print("Example: 1,3,5 (for Red, Blue, Black)")
    
    selected_colors = []
    while not selected_colors:
        try:
            numbers_input = input("\nEnter color numbers: ")
            if not numbers_input.strip():
                print("No colors selected. Please try again.")
                continue
                
            numbers = [int(num.strip()) for num in numbers_input.split(",")]
            
            for num in numbers:
                if 1 <= num <= len(available_colors):
                    selected_colors.append(available_colors[num-1])
                else:
                    print(f"Invalid number {num}. Please enter numbers between 1 and {len(available_colors)}.")
                    selected_colors = []
                    break
            
            if not selected_colors:
                continue
                
            print(f"Selected colors: {', '.join(selected_colors)}")
            return selected_colors
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
    return None


def get_shape_input(classes):
    print("\nAvailable clothing types:")
    for i, cls in enumerate(classes):
        print(f"{i+1}. {cls}")
    
    while True:
        try:
            idx = int(input("\nEnter the number of your choice: ")) - 1
            if 0 <= idx < len(classes):
                return classes[idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(classes)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def display_gallery(imgs, class_labels, color_labels, limit=100):
    if len(imgs) == 0:
        print("No images to display.")
        return
    
    display_count = min(limit, len(imgs))
    print(f"\nDisplaying gallery with {display_count} out of {len(imgs)} items")
    
    info = []
    for i in range(display_count):
        colors_text = ", ".join(color_labels[i])
        info.append(f"{class_labels[i]}: {colors_text}")
    
    visualize_retrieval(imgs[:display_count], display_count, info=info, title="Product Gallery")


if __name__ == '__main__':
    try:
        print("Loading dataset...")
        train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
            test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

        classes = list(set(list(train_class_labels) + list(test_class_labels)))

        imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
        cropped_images = crop_images(imgs, upper, lower)
        
        print("Dataset loaded successfully!")
        print(f"Training images: {len(train_imgs)}")
        print(f"Test images: {len(test_imgs)}")
        print(f"Available classes: {', '.join(classes)}")
        
        print("\nInitializing models...")
        
        print("Implementing KMeans for color labeling...")
        X = train_imgs.reshape(-1, 3)
        kmeans = KMeans(X, K=5)
        kmeans.fit()
        
        centroids_colors = get_colors(kmeans.centroids)
        
        print("Implementing KNN for shape labeling...")
        train_gray = np.mean(train_imgs, axis=3) if train_imgs.ndim == 4 else train_imgs
        test_gray = np.mean(test_imgs, axis=3) if test_imgs.ndim == 4 else test_imgs
        
        knn = KNN(train_gray, train_class_labels)
        predicted_shapes = knn.predict(test_gray, k=5)
        
        print("Models initialized successfully!")
        
        while True:
            choice = display_menu()
            
            if choice == 0:
                print("Exiting program. Goodbye!")
                break
                
            elif choice == 1:
                try:
                    colors = get_color_input()
                    color_results, color_percentages, color_indices = Retrieval_by_color(
                        test_imgs, test_color_labels, colors)
                    
                    if len(color_results) > 0:
                        visualize_retrieval(color_results, min(10, len(color_results)), 
                                           title=f"Images containing {', '.join(colors)}")
                    else:
                        print(f"No images found with the specified colors: {', '.join(colors)}")
                except Exception as e:
                    print(f"Error during color search: {str(e)}")
                
            elif choice == 2:
                try:
                    shape = get_shape_input(classes)
                    shape_results, shape_indices = Retrieval_by_shape(
                        test_imgs, test_class_labels, shape)
                    
                    if len(shape_results) > 0:
                        visualize_retrieval(shape_results, min(10, len(shape_results)), 
                                           title=f"Images with shape: {shape}")
                    else:
                        print(f"No images found with shape: {shape}")
                except Exception as e:
                    print(f"Error during shape search: {str(e)}")
                
            elif choice == 3:
                try:
                    shape = get_shape_input(classes)
                    colors = get_color_input()
                    
                    combined_results, combined_info, combined_indices = Retrieval_combined(
                        test_imgs, test_class_labels, test_color_labels, shape, colors)
                    
                    if len(combined_results) > 0:
                        visualize_retrieval(combined_results, min(10, len(combined_results)), 
                                           info=combined_info, 
                                           title=f"{', '.join(colors)} {shape}")
                    else:
                        print(f"No images found matching both criteria: {', '.join(colors)} {shape}")
                except Exception as e:
                    print(f"Error during combined search: {str(e)}")
                
            elif choice == 4:
                try:
                    display_gallery(test_imgs, test_class_labels, test_color_labels)
                except Exception as e:
                    print(f"Error displaying gallery: {str(e)}")
                
            elif choice == 5:
                try:
                    max_k = 8
                    try:
                        max_k_input = input("Enter maximum K value for analysis (default: 8): ")
                        if max_k_input.strip():
                            max_k = int(max_k_input)
                    except ValueError:
                        print("Invalid input. Using default value of 8.")
                    
                    print("\nAnalyzing KMeans performance...")
                    sample_size = min(20, len(train_imgs))
                    sample_imgs = train_imgs[:sample_size]
                    stats = Kmean_statistics(KMeans, sample_imgs, max_K=max_k)
                except Exception as e:
                    print(f"Error during performance analysis: {str(e)}")
                
            elif choice == 6:
                try:
                    print("\nCalculating classification accuracy...")
                    shape_accuracy = Get_shape_accuracy(predicted_shapes, test_class_labels)
                    print(f"Shape classification accuracy: {shape_accuracy:.2f}%")
                    
                    color_accuracy = Get_color_accuracy(test_color_labels, test_color_labels)
                    print(f"Color classification accuracy: {color_accuracy:.2f}%")
                except Exception as e:
                    print(f"Error calculating accuracy: {str(e)}")
    
    except Exception as e:
        print(f"An error occurred during program initialization: {str(e)}")
        import traceback
        traceback.print_exc()
