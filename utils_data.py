import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import cv2


def crop_images(images, upper, lower):
    cropped_image = []
    for image, top_cord, bottom_cord in zip(images, upper, lower):
        cropped_image.append(image[top_cord[1]:bottom_cord[1], top_cord[0]:bottom_cord[0], :])
    return np.array(cropped_image, dtype=object)


def read_extended_dataset(root_folder='./images/', extended_gt_json='./images/gt_reduced.json', w=60, h=80):
    """
        reads the extended ground truth, returns:
            images: the images in color (80x60x3)
            shape labels: array of strings
            color labels: array of arrays of strings
            upper_left_coord: (x, y) coordinates of the window top left
            lower_right_coord: (x, y) coordinates of the window bottom right
            background: array of booleans indicating if the defined window contains background or not
    """
    ground_truth_extended = json.load(open(extended_gt_json, 'r'))
    img_names, class_labels, color_labels, upper, lower, background = [], [], [], [], [], []

    for k, v in ground_truth_extended.items():
        img_names.append(os.path.join(root_folder, 'train', k))
        class_labels.append(v[0])
        color_labels.append(v[1])
        upper.append(v[2])
        lower.append(v[3])
        background.append(True if v[4] == 1 else False)

    imgs = load_imgs(img_names, w, h, True)

    idxs = np.arange(imgs.shape[0])
    np.random.seed(42)
    np.random.shuffle(idxs)

    imgs = imgs[idxs]
    class_labels = np.array(class_labels)[idxs]
    color_labels = np.array(color_labels, dtype=object)[idxs]
    upper = np.array(upper)[idxs]
    lower = np.array(lower)[idxs]
    background = np.array(background)[idxs]

    return imgs, class_labels, color_labels, upper, lower, background


def read_dataset(root_folder='./images/', gt_json='./test/gt.json', w=60, h=80, with_color=True):
    """
        reads the dataset (train and test), returns the images and labels (class and colors) for both sets
    """
    np.random.seed(123)
    ground_truth = json.load(open(gt_json, 'r'))

    train_img_names, train_class_labels, train_color_labels = [], [], []
    test_img_names, test_class_labels, test_color_labels = [], [], []
    for k, v in ground_truth['train'].items():
        train_img_names.append(os.path.join(root_folder, 'train', k))
        train_class_labels.append(v[0])
        train_color_labels.append(v[1])

    for k, v in ground_truth['test'].items():
        test_img_names.append(os.path.join(root_folder, 'test', k))
        test_class_labels.append(v[0])
        test_color_labels.append(v[1])

    train_imgs = load_imgs(train_img_names, w, h, with_color)
    test_imgs = load_imgs(test_img_names, w, h, with_color)

    np.random.seed(42)

    idxs = np.arange(train_imgs.shape[0])
    np.random.shuffle(idxs)
    train_imgs = train_imgs[idxs]
    train_class_labels = np.array(train_class_labels)[idxs]
    train_color_labels = np.array(train_color_labels, dtype=object)[idxs]

    idxs = np.arange(test_imgs.shape[0])
    np.random.shuffle(idxs)
    test_imgs = test_imgs[idxs]
    test_class_labels = np.array(test_class_labels)[idxs]
    test_color_labels = np.array(test_color_labels, dtype=object)[idxs]

    return train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels


def load_imgs(img_names, w, h, with_color):
    imgs = []
    for tr in img_names:
        imgs.append(read_one_img(tr + '.jpg', w, h, with_color))
    return np.array(imgs)


def read_one_img(img_name, w, h, with_color):
    img = Image.open(img_name)

    if with_color:
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    if img.size != (w, h):
        img = img.resize((w, h))
    return np.array(img)


def visualize_retrieval(imgs, topN, info=None, ok=None, title='', query=None):
    def add_border(color):
        return np.stack(
            [np.pad(imgs[i, :, :, c], 3, mode='constant', constant_values=color[c]) for c in range(3)], axis=2
        )

    columns = 4
    rows = math.ceil(topN/columns)
    if query is not None:
        fig = plt.figure(figsize=(10, 8*6/8))
        columns += 1
        fig.add_subplot(rows, columns, 1+columns)
        plt.imshow(query)
        plt.axis('off')
        plt.title(f'query', fontsize=8)
    else:
        fig = plt.figure(figsize=(8, 8*6/8))

    for i in range(min(topN, len(imgs))):
        sp = i+1
        if query is not None:
            sp = (sp - 1) // (columns-1) + 1 + sp
        fig.add_subplot(rows, columns, sp)
        if ok is not None:
            im = add_border([0, 255, 0] if ok[i] else [255, 0, 0])
        else:
            im = imgs[i]
        plt.imshow(im)
        plt.axis('off')
        if info is not None:
            plt.title(f'{info[i]}', fontsize=8)
    plt.gcf().suptitle(title)
    plt.show()


# Visualize k-mean with 3D plot
def Plot3DCloud(km, rows=1, cols=1, spl_id=1):
    ax = plt.gcf().add_subplot(rows, cols, spl_id, projection='3d')

    for k in range(km.K):
        Xl = km.X[km.labels == k, :]
        ax.scatter(
            Xl[:, 0], Xl[:, 1], Xl[:, 2], marker='.', c=km.centroids[np.ones((Xl.shape[0]), dtype='int') * k, :] / 255
        )

    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    ax.set_zlabel('dim 3')
    return ax


def visualize_k_means(kmeans, img_shape):
    def prepare_img(x, img_shape):
        x = np.clip(x.astype('uint8'), 0, 255)
        x = x.reshape(img_shape)
        return x

    fig = plt.figure(figsize=(8, 8))

    X_compressed = kmeans.centroids[kmeans.labels]
    X_compressed = prepare_img(X_compressed, img_shape)

    org_img = prepare_img(kmeans.X, img_shape)

    fig.add_subplot(131)
    plt.imshow(org_img)
    plt.title('original')
    plt.axis('off')

    fig.add_subplot(132)
    plt.imshow(X_compressed)
    plt.axis('off')
    plt.title('kmeans')

    Plot3DCloud(kmeans, 1, 3, 3)
    plt.title('núvol de punts')
    plt.show()


# Add image preprocessing functions
def convert_to_grayscale(images):
    """
    Convert color images to grayscale
    :param images: array of images (N, H, W, 3)
    :return: grayscale images (N, H, W)
    """
    if images.ndim == 4:  # batch of images
        return np.mean(images, axis=3)
    elif images.ndim == 3 and images.shape[2] == 3:  # single image
        return np.mean(images, axis=2)
    return images  # already grayscale

def downscale_images(images, scale_factor=0.5):
    """
    Downscale images by a given factor
    :param images: array of images (N, H, W, C)
    :param scale_factor: factor to downscale by (0.5 = half size)
    :return: downscaled images
    """
    if images.ndim != 4:
        raise ValueError("Expected 4D array (N, H, W, C)")
    
    N, H, W, C = images.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    
    resized_images = np.zeros((N, new_H, new_W, C), dtype=images.dtype)
    
    for i in range(N):
        # Use cv2 for efficient image resizing
        resized_images[i] = cv2.resize(images[i], (new_W, new_H))
    
    return resized_images

def analyze_color_coverage(image, color_centroids, threshold=0.15):
    """
    Analyze color coverage in an image
    :param image: image array (H, W, 3)
    :param color_centroids: array of color centroids
    :param threshold: minimum coverage threshold (0-1)
    :return: list of color indices with coverage above threshold, color percentages
    """
    # Reshape image for clustering
    pixels = image.reshape(-1, 3)
    
    # Calculate distances to each centroid
    distances = np.zeros((pixels.shape[0], color_centroids.shape[0]))
    for i, centroid in enumerate(color_centroids):
        distances[:, i] = np.sqrt(np.sum((pixels - centroid)**2, axis=1))
    
    # Assign each pixel to closest centroid
    labels = np.argmin(distances, axis=1)
    
    # Calculate coverage percentage for each color
    total_pixels = pixels.shape[0]
    color_coverage = np.zeros(color_centroids.shape[0])
    
    for i in range(color_centroids.shape[0]):
        color_coverage[i] = np.sum(labels == i) / total_pixels
    
    # Get colors above threshold
    colors_above_threshold = np.where(color_coverage >= threshold)[0]
    
    return colors_above_threshold, color_coverage

def verify_dominant_colors(image, color_labels, color_map, min_coverage=0.15):
    """
    Secondary verification of dominant colors
    :param image: image array (H, W, 3)
    :param color_labels: list of color labels
    :param color_map: mapping of color names to RGB values
    :param min_coverage: minimum coverage threshold (0-1)
    :return: verified colors, coverage percentages
    """
    H, W, _ = image.shape
    total_pixels = H * W
    verified_colors = []
    coverage_percent = {}
    
    # Convert image to HSV for better color segmentation
    hsv_img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    
    # Get all ranges from color_ranges dictionary
    color_ranges = {
        'red': [(np.array([0, 70, 50]), np.array([10, 255, 255])),
                (np.array([170, 70, 50]), np.array([180, 255, 255]))],
        'green': [(np.array([35, 25, 25]), np.array([85, 255, 255]))],
        'blue': [(np.array([100, 50, 50]), np.array([140, 255, 255]))],
        'yellow': [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
        'orange': [(np.array([5, 80, 100]), np.array([25, 255, 255]))],
        'purple': [(np.array([140, 40, 40]), np.array([170, 255, 255]))],
        'pink': [(np.array([140, 10, 100]), np.array([175, 255, 255])),  # Main pink range
                (np.array([0, 10, 150]), np.array([10, 255, 255])),      # Light pinks
                (np.array([175, 10, 100]), np.array([180, 255, 255])),   # Pink near red
                (np.array([140, 10, 150]), np.array([170, 100, 255]))],  # Pale/pastel pinks
        'brown': [(np.array([10, 60, 20]), np.array([30, 255, 200]))],
        'black': [(np.array([0, 0, 0]), np.array([180, 255, 40])),       # Expanded black range
                 (np.array([0, 0, 0]), np.array([180, 30, 60]))],        # Additional range for dark grays
        'white': [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
        'gray': [(np.array([0, 0, 40]), np.array([180, 50, 200]))],
        'grey': [(np.array([0, 0, 40]), np.array([180, 50, 200]))]
    }
    
    # Define color-specific thresholds
    color_thresholds = {
        'pink': min_coverage * 0.6,    # 40% lower threshold for pink
        'orange': min_coverage * 0.7,  # 30% lower threshold for orange
        'grey': min_coverage * 0.7,    # 30% lower threshold for grey
        'gray': min_coverage * 0.7,    # 30% lower threshold for gray
        'green': min_coverage * 0.7,   # 30% lower threshold for green
        'black': min_coverage * 0.5,   # 50% lower threshold for black
        'white': min_coverage * 0.7    # 30% lower threshold for white
    }
    
    for color in color_labels:
        color_lower = color.lower()
        
        if color_lower in color_ranges:
            # Get all ranges for this color
            ranges = color_ranges[color_lower]
            
            # Create a combined mask for all ranges
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            
            for lower_bound, upper_bound in ranges:
                mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            coverage = np.sum(combined_mask > 0) / total_pixels
            
            # Use color-specific threshold if available
            adjusted_threshold = color_thresholds.get(color_lower, min_coverage)
            
            # Special handling for black in dark images
            if color_lower == 'black':
                # Calculate average brightness of the image
                avg_brightness = np.mean(hsv_img[:,:,2])
                
                # If the image is generally dark, lower the threshold even more
                if avg_brightness < 100:
                    adjusted_threshold = min_coverage * 0.4  # 60% lower threshold for black in dark images
            
            if coverage >= adjusted_threshold:
                verified_colors.append(color)
                coverage_percent[color] = coverage
    
    return verified_colors, coverage_percent

def get_color_hsv_range(color_name):
    """Get HSV range for common colors"""
    color_ranges = {
        'red': [(np.array([0, 70, 50]), np.array([10, 255, 255])),
                (np.array([170, 70, 50]), np.array([180, 255, 255]))],
        'green': [(np.array([35, 25, 25]), np.array([85, 255, 255]))],  # Expanded green range
        'blue': [(np.array([100, 50, 50]), np.array([140, 255, 255]))],
        'yellow': [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
        'orange': [(np.array([5, 80, 100]), np.array([25, 255, 255]))],  # Expanded orange range
        'purple': [(np.array([140, 40, 40]), np.array([170, 255, 255]))],
        'pink': [(np.array([140, 10, 100]), np.array([175, 255, 255])),  # Main pink range
                (np.array([0, 10, 150]), np.array([10, 255, 255])),      # Light pinks
                (np.array([175, 10, 100]), np.array([180, 255, 255])),   # Third range for pink near red
                (np.array([140, 10, 150]), np.array([170, 100, 255]))],  # Pale/pastel pinks
        'brown': [(np.array([10, 60, 20]), np.array([30, 255, 200]))],
        'black': [(np.array([0, 0, 0]), np.array([180, 255, 40])),       # Expanded black range
                 (np.array([0, 0, 0]), np.array([180, 30, 60]))],        # Additional range for dark grays
        'white': [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
        'gray': [(np.array([0, 0, 40]), np.array([180, 50, 200]))],  # Expanded gray range
        'grey': [(np.array([0, 0, 40]), np.array([180, 50, 200]))]   # Added 'grey' as alias
    }
    
    color_name = color_name.lower()
    
    if color_name in color_ranges:
        ranges = color_ranges[color_name]
        if len(ranges) == 1:
            return ranges[0][0], ranges[0][1]
        else:  # For colors like red that wrap around the hue
            # Return the first range by default, but we need to modify the verification function
            # to handle multiple ranges for better detection
            return ranges[0][0], ranges[0][1]
    
    return None, None
