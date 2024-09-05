from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

def load_and_preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x
    except Exception as e:
        print(f'Error loading image {img_path}: {e}')
        return None, None

def classify_images_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img, x = load_and_preprocess_image(img_path)
            if x is not None:
                preds = model.predict(x)
                decoded_preds = decode_predictions(preds, top=3)[0]
                
                # Get the class names and probabilities for top 3 predictions
                classes = [pred[1] for pred in decoded_preds]
                probabilities = [pred[2] for pred in decoded_preds]

                # Plotting the predictions for each image
                plt.figure(figsize=(8, 4))
                plt.barh(classes, probabilities, color='skyblue')
                plt.xlabel('Probability')
                plt.title(f'Predictions for {filename}')
                plt.xlim(0, 1)
                plt.gca().invert_yaxis()  # Invert y-axis to have the highest probability at the top
                plt.show()

# Putanja do foldera sa slikama
folder_path = '/Users/nadjacvetkovic/Desktop/zaBanane'
classify_images_from_folder(folder_path)
