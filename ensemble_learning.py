import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np

weights = []
weights = [0.11609369542378466, 0.2441298715454647, 0.2063605433094278, 0.10432392840613987, 0.10339322202709791, 0.22569873928808507]
loaded_models = []

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image to the required input shape of the model
    image = cv2.resize(image, (96, 96))
    # Convert the image to float32 and normalize its values
    image = image.astype(np.float32) / 255.0
    # Expand the dimensions to match the input shape expected by the model
    #image = np.expand_dims(image, axis=0)
    return image



# Function to perform weighted averaging
def weighted_average(predictions, weights):
    weighted_predictions = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        weighted_predictions += pred * weight
    return weighted_predictions

class Ensemble_Learning:
    @staticmethod
    def build():
        agedb_test_images_path = [f for f in glob.glob("test_images" + "/*.jpg", recursive=True)]
        trained_model_paths= [
            "resnet_V3.keras",
            "effnet_V3.keras",
            "alexnet_V3.keras",
            "vggnet_V2.keras",
            "googlenet_V3.keras",
            "CNN_V3.keras"
        ]

        for model_path in trained_model_paths:
            model = tf.keras.models.load_model(model_path)
            print(type(model))
            loaded_models.append(model)

        #print(loaded_models)
        # Split preprocessed data into training and testing sets
        validation_files, _ = train_test_split(agedb_test_images_path, test_size=0.2, random_state=42)

        def extract_age_label(filename):
            # Extract age label from filename
            age_label = int(filename.split("/")[-1].split("_")[2])
            return int(age_label)


        #print(len(validation_files))
        validation_labels = [extract_age_label(validation_file) for validation_file in validation_files]

        #print(validation_labels)

        validation_files= [preprocess_image(file) for file in validation_files]
        # Evaluate models on validation data and calculate weights
        computed_weights = []
        for model, validation_data in zip(loaded_models, validation_files):
            #validation_data = np.expand_dims(validation_data, axis=0)  # Add batch dimension
            _, accuracy = model.evaluate(np.array([validation_data]), np.array([validation_labels]))  # Evaluate with validation labels
            computed_weights.append(accuracy)

        # Normalize weights
        total_weight = sum(computed_weights)
        for weight in computed_weights:
            weights.append(weight / total_weight)
        #print(weights)

    def model_prediction(image):
        predictions = []
        for model in loaded_models:
            prediction = model.predict(np.array(image))
            predictions.append(prediction)
        final_predictions = weighted_average(predictions, weights)
        return final_predictions
