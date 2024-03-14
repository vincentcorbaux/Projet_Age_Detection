import cv2
import numpy as np
import tensorflow as tf
import time
from ensemble_learning import Ensemble_Learning


print(tf.__version__)

def predict(image):
        # Redimensionner l'image
        #resized_img = cv2.resize(image, (96, 96))
        # Normaliser l'image
        #normalized_img = resized_img / 255.0

        # Ajouter une dimension de lot supplémentaire
        #normalized_img_batch = np.expand_dims(image, axis=0)

        predictions = np.array(vgg_classifier.predict(image))
        max_index = np.argmax(predictions)

        if max_index == 0:
            pred = model_resnet_0.predict(image)
            
        elif max_index == 1:
            pred = model_effnet.predict(image)
            
        elif max_index == 2:
            pred = model_resnet_1.predict(image)
            
        elif max_index == 3:
            pred = model_cnn.predict(image)
            
        elif max_index == 4:
            pred = model_vggnet.predict(image)

        return pred


# Load OpenCV's face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
vgg_classifier = tf.keras.models.load_model("trained_models/vgg_classifier.keras")
model_resnet_0 = tf.keras.models.load_model("trained_models/Resnet_0_17.keras")
model_effnet = tf.keras.models.load_model("trained_models/EfficientNet_18_25.keras")
model_resnet_1 = tf.keras.models.load_model("trained_models/Resnet_26_50.keras")
model_cnn = tf.keras.models.load_model("trained_models/CNN_51_70.keras")
model_vggnet = tf.keras.models.load_model("trained_models/VGG_71_120.keras")


# Load the pre-trained models for age estimation
#model_resnet = tf.keras.models.load_model('ResNet50_age_detection.model')
#model_effnet = tf.keras.models.load_model('EfficientNetB0.keras')
#model_googlenet = tf.keras.models.load_model('GoogleNet_age_detection.model')
#model_alexnet = tf.keras.models.load_model('AlexNet2_age_detection.model')
#model_cnn = tf.keras.models.load_model('CNNModel.keras')
#model_vggnet = tf.keras.models.load_model('VGGNet_age_detection.model')


# Choose a model to use for prediction
#Modify image size for cnn (200,200)
#Ensemble_Learning.build() # or model_effnet

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for calculating the average age
ages = []
last_update = time.time()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (96, 96))  # Assuming the model input size is 96x96
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        # Prediction
        #age = Ensemble_Learning.model_prediction(face)
        age = predict(face)
        age_value = round(age[0][0])

        # Display the predicted age
        label = f"Age: {age_value} years"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Age Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
