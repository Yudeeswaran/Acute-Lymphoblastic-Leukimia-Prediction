import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt

# Load the VGG16 model pre-trained on ImageNet
model = VGG16(weights='imagenet', include_top=False)

# Load and preprocess the image
img_path = 'train/[Malignant] early Pre-B/Sap_148 (1).jpg'  # Path to the uploaded image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Extract features
features = model.predict(x)

# Flatten the features to create a 1D array
flattened_features = features.flatten()

# Output the 1D array
print(flattened_features)

# Save the 1D array to a file if needed
np.savetxt('vgg16_features.txt', flattened_features)

def visualize_feature_maps(feature_maps):
    square = 8  # define the square dimensions for plotting
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')
            ix += 1
    plt.show()

# Extract features from a specific layer
layer_outputs = [layer.output for layer in model.layers[:12]]  # Change 12 to any layer number you want to inspect
from tensorflow.keras.models import Model
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x)

# Visualize feature maps from a specific layer (e.g., the 4th layer)
feature_maps = activations[4]  # Change 4 to the index of the layer you want to visualize
visualize_feature_maps(feature_maps)