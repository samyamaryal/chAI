import os
import cv2
import json
import numpy as np
from deepface import DeepFace 
from deepface.models.demography import Emotion

model = DeepFace.build_model(task="facial_attribute", model_name="Emotion")

folder_path = "video_frames/"  # Path to images
labels = Emotion.labels
predictions_json = []

# List all image files (you can extend this to support more formats)
image_files = [f for f in os.listdir(folder_path)]
image_files.sort()

for image in image_files:
    image_path = os.path.join(folder_path, image)
    face = DeepFace.extract_faces(img_path=image_path, detector_backend='opencv', enforce_detection=False)[0] # Capture only one face

    face_img = face['face'].astype('float32') # This was encoded as float64 - not compatible with CV2. 

    face_img_batch = np.expand_dims(face_img, axis=0) # Add a batch dimension to the image for Keras compatibility
    prediction_probabilities = model.predict(face_img_batch)
    
    # Fetch emotion probabilities and store it in a dictionary of the format {emotion: probability}
    prob_dict = {emotion: "{:.3f}".format(prob) for emotion, prob in zip(labels, prediction_probabilities)} 

    # Convert probabilities to a dictionary with emotion names
    predictions_json.append(json.dumps({image: prob_dict}))
