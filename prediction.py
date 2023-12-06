import os
import numpy as np
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

img_height = 128  
img_width = 128   


# Set the paths
model_weights_path = r"models/best_model.keras"
test_data_dir = r"split/test"
output_csv_path = r"predictions.csv"

# Load the model
model = load_model(model_weights_path)

# Get class names
class_names = sorted(os.listdir(test_data_dir))

# Open a CSV file for writing predictions
with open(output_csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image', 'Actual Class', 'Predicted Class', 'Verdict'])

    # Iterate over test images
    for class_name in class_names:
        class_path = os.path.join(test_data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Load and preprocess the test image
            img = image.load_img(img_path, target_size=(img_height, img_width))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values to between 0 and 1

            # Make predictions
            predictions = model.predict(img_array)

            # Interpret predictions
            predicted_class = np.argmax(predictions)
            predicted_class_name = class_names[predicted_class]

            # Determine verdict
            verdict = 'True' if class_name == predicted_class_name else 'False'

            # Write results to CSV
            csv_writer.writerow([img_name, class_name, predicted_class_name, verdict])

print(f"Predictions written to {output_csv_path}")
