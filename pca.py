import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path, image_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, image_size)
                images.append(img_resized.flatten())
                labels.append(filename)
    return np.array(images), labels

def apply_pca(data, n_components=50):
    pca = PCA(n_components=n_components, whiten=True)
    transformed_data = pca.fit_transform(data)
    return pca, transformed_data

def find_most_similar(pca, dataset_features, input_image_vector):
    input_feature = pca.transform([input_image_vector])
    distances = np.linalg.norm(dataset_features - input_feature, axis=1)
    return np.argmin(distances), distances

def plot_pca_bases(pca, image_shape=(100, 100), num_components=9):
    plt.figure(figsize=(12, 6))
    for i in range(num_components):
        plt.subplot(3, 3, i + 1)
        plt.imshow(pca.components_[i].reshape(image_shape), cmap='gray')
        plt.title(f"Basis {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
