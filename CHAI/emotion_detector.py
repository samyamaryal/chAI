import os
import json
import numpy as np
from deepface import DeepFace 
from deepface.models.demography import Emotion

def analyze_emotions_in_recording(combined_log_path="./frames/combined_log.json", webcam_folder="./frames/webcam/"):
    """
    Analyze emotions in webcam frames and add emotion data to the combined log
    
    Parameters:
    - combined_log_path: Path to the combined log JSON file
    - webcam_folder: Path to the webcam frames folder
    
    Returns:
    - dict: Updated combined log with emotion data
    """
    try:
        # Load the combined log
        with open(combined_log_path, 'r') as f:
            combined_log = json.load(f)
        
        # Initialize emotion model
        try:
            model = DeepFace.build_model(task="facial_attribute", model_name="Emotion")
            labels = Emotion.labels
            print(f"Model initialized successfully. Labels: {labels}")
        except Exception as model_init_error:
            print(f"Error initializing emotion model: {model_init_error}")
            return None
        
        print(f"Starting emotion analysis for {len(combined_log['webcam'])} webcam frames...")
        
        # Counters for summary
        processed_frames = 0
        skipped_frames = 0
        failed_frames = 0
        
        # Process each webcam frame
        for i, image_entry in enumerate(combined_log["webcam"]):
            try:
                # Check if emotion data already exists for this frame
                if "emotion" in image_entry:
                    print(f"Skipping {image_entry['file']} - emotion data already exists")
                    skipped_frames += 1
                    continue
                
                image_path = os.path.join(webcam_folder, image_entry["file"])
                
                # Check if image file exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found: {image_path}")
                    continue
                
                # Extract face from image
                faces = DeepFace.extract_faces(
                    img_path=image_path, 
                    detector_backend='opencv', 
                    enforce_detection=False
                )
                
                if not faces:
                    print(f"Warning: No face detected in {image_entry['file']}")
                    # Add default emotion data for frames without faces
                    combined_log["webcam"][i]["emotion"] = {
                        "angry": "0.000",
                        "disgust": "0.000", 
                        "fear": "0.000",
                        "happy": "0.000",
                        "sad": "0.000",
                        "surprise": "0.000",
                        "neutral": "1.000"
                    }
                    failed_frames += 1
                    continue
                
                # Try alternative method using DeepFace's built-in emotion analysis
                try:
                    emotion_result = DeepFace.analyze(
                        img_path=image_path,
                        actions=['emotion'],
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    
                    if emotion_result and len(emotion_result) > 0:
                        # Extract emotion data from the result
                        emotion_data = emotion_result[0].get('emotion', {})
                        
                        # Convert to our format
                        prob_dict = {}
                        for emotion, prob in emotion_data.items():
                            prob_dict[emotion] = "{:.3f}".format(float(prob))
                        
                        # Add emotion data to the webcam entry
                        combined_log["webcam"][i]["emotion"] = prob_dict
                        print(f"Used DeepFace.analyze for {image_entry['file']}")
                        processed_frames += 1
                        continue
                        
                except Exception as deepface_error:
                    print(f"DeepFace.analyze failed for {image_entry['file']}: {deepface_error}")
                    # Continue with manual model approach
                
                # Use the first detected face
                face = faces[0]
                face_img = face['face'].astype('float32')
                
                # Add batch dimension for model prediction
                face_img_batch = np.expand_dims(face_img, axis=0)
                
                try:
                    prediction_probabilities = model.predict(face_img_batch)
                    
                    # Debug information
                    print(f"Prediction shape: {prediction_probabilities.shape}")
                    print(f"Prediction type: {type(prediction_probabilities)}")
                    print(f"Labels: {labels}")
                    
                    # Handle the prediction probabilities properly
                    # The model returns a 2D array, we need to flatten it
                    if prediction_probabilities.ndim > 1:
                        prediction_probabilities = prediction_probabilities.flatten()
                    
                    print(f"Flattened prediction shape: {prediction_probabilities.shape}")
                    
                    # Create emotion probability dictionary
                    prob_dict = {}
                    for i, emotion in enumerate(labels):
                        if i < len(prediction_probabilities):
                            prob_value = float(prediction_probabilities[i])
                            prob_dict[emotion] = "{:.3f}".format(prob_value/100)
                        else:
                            prob_dict[emotion] = "0.000"
                    
                    print(f"Created prob_dict: {prob_dict}")
                    
                    # Add emotion data to the webcam entry
                    combined_log["webcam"][i]["emotion"] = prob_dict
                    processed_frames += 1
                    
                except Exception as model_error:
                    print(f"Model prediction error for {image_entry['file']}: {model_error}")
                    # Add default emotion data for model prediction failures
                    combined_log["webcam"][i]["emotion"] = {
                        "angry": "0.000",
                        "disgust": "0.000",
                        "fear": "0.000", 
                        "happy": "0.000",
                        "sad": "0.000",
                        "surprise": "0.000",
                        "neutral": "1.000"
                    }
                    failed_frames += 1
                
                # Print progress every 10 frames
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(combined_log['webcam'])} frames")
                    
            except Exception as e:
                print(f"Error processing frame {image_entry['file']}: {e}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                # Add default emotion data for failed frames
                combined_log["webcam"][i]["emotion"] = {
                    "angry": "0.000",
                    "disgust": "0.000",
                    "fear": "0.000", 
                    "happy": "0.000",
                    "sad": "0.000",
                    "surprise": "0.000",
                    "neutral": "1.000"
                }
                failed_frames += 1
        
        # Print summary
        total_frames = len(combined_log["webcam"])
        print(f"\n📊 Emotion Analysis Summary:")
        print(f"   Total frames: {total_frames}")
        print(f"   Processed: {processed_frames}")
        print(f"   Skipped (already had emotion data): {skipped_frames}")
        print(f"   Failed: {failed_frames}")
        print(f"   Success rate: {(processed_frames / (processed_frames + failed_frames) * 100):.1f}%" if (processed_frames + failed_frames) > 0 else "   Success rate: N/A")
        
        # Save updated log back to the same file
        with open(combined_log_path, 'w') as f:
            json.dump(combined_log, f, indent=2)
        
        print(f"Emotion analysis complete! Updated {combined_log_path}")
        return combined_log
        
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return None

def get_dominant_emotion(emotion_data):
    """
    Get the dominant emotion from emotion probabilities
    
    Parameters:
    - emotion_data: Dictionary with emotion probabilities
    
    Returns:
    - str: Name of the dominant emotion
    """
    if not emotion_data:
        return "neutral"
    
    # Convert string probabilities to floats
    emotions = {k: float(v) for k, v in emotion_data.items()}
    
    # Find the emotion with highest probability
    dominant_emotion = max(emotions, key=emotions.get)
    return dominant_emotion

# Legacy function for backward compatibility
def run_emotion_detection():
    """Legacy function that runs the original emotion detection workflow"""
    return analyze_emotions_in_recording()

if __name__ == "__main__":
    # Run emotion detection when script is executed directly
    result = analyze_emotions_in_recording()
    if result:
        print("Emotion detection completed successfully!")
    else:
        print("Emotion detection failed!")
