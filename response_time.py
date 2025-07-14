import json
import cv2
import numpy as np
from datetime import datetime
import os

def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error between two images
    
    Parameters:
    - img1, img2: Input images (grayscale)
    
    Returns:
    - MSE score (lower = more similar, 0 = identical images)
    """
    if img1 is None or img2 is None:
        return float('inf')
    
    # Ensure both images have the same dimensions
    if img1.shape != img2.shape:
        # Resize img2 to match img1
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to float for calculations
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    return mse

def load_image(image_path):
    """Load and preprocess image for MSE comparison"""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Convert to grayscale for MSE comparison
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def calculate_similarity(img1, img2):
    """Calculate similarity between two images using MSE"""
    if img1 is None or img2 is None:
        return 0.0
    
    try:
        mse_score = calculate_mse(img1, img2)
        # Convert MSE to similarity score (lower MSE = higher similarity)
        # Normalize to 0-1 range where 1 = identical, 0 = very different
        # Using exponential decay: similarity = exp(-mse / scale_factor)
        scale_factor = 1000.0  # Adjust this based on your image characteristics
        similarity = np.exp(-mse_score / scale_factor)
        
        # Ensure similarity is bounded between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        # Debug output
        print(f"[DEBUG] MSE: {mse_score:.6f}, Similarity: {similarity:.6f}")
        
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def find_click_frames(combined_log):
    """Find frames where mouse left button clicks occurred"""
    click_frames = []
    
    for mouse_event in combined_log.get('mouse', []):
        if (mouse_event.get('event') == 'click' and 
            mouse_event.get('button') == 'Button.left' and 
            mouse_event.get('pressed') == True):
            
            click_time = datetime.fromisoformat(mouse_event['time'])
            click_frames.append({
                'time': mouse_event['time'],
                'position': mouse_event['position'],
                'datetime': click_time
            })
    return click_frames

def find_screen_frame_at_time(combined_log, target_time, tolerance_seconds=0.1):
    """Find the closest screen frame to a given time"""
    target_dt = datetime.fromisoformat(target_time)
    closest_frame = None
    min_diff = float('inf')
    
    for frame_data in combined_log.get('screen', []):
        frame_time = datetime.fromisoformat(frame_data['time'])
        time_diff = abs((frame_time - target_dt).total_seconds())
        
        if time_diff < min_diff and time_diff <= tolerance_seconds:
            min_diff = time_diff
            closest_frame = frame_data
    
    return closest_frame

def find_change_frames(combined_log, click_frame, frames_dir, similarity_threshold=0.8, max_frames_after=50):
    """Find frames where similarity drops significantly after a click"""
    click_time = click_frame['datetime']
    screen_frames = combined_log.get('screen', [])
    
    # Find the screen frame closest to the click time
    click_screen_frame = find_screen_frame_at_time(combined_log, click_frame['time'])
    if not click_screen_frame:
        return None, None
    
    # Load the click frame image
    click_image_path = os.path.join(frames_dir, 'screen', click_screen_frame['file'])
    click_image = load_image(click_image_path)
    if click_image is None:
        return None, None
    
    # Find frames after the click
    click_frame_idx = None
    for i, frame in enumerate(screen_frames):
        if frame['file'] == click_screen_frame['file']:
            click_frame_idx = i
            break
    
    if click_frame_idx is None:
        return None, None
    
    # Check subsequent frames for similarity changes
    for i in range(click_frame_idx + 1, min(click_frame_idx + max_frames_after, len(screen_frames))):
        current_frame = screen_frames[i]
        current_image_path = os.path.join(frames_dir, 'screen', current_frame['file'])
        current_image = load_image(current_image_path)
        
        if current_image is not None:
            similarity_score = calculate_similarity(click_image, current_image)
            print("[DEBUG] similarity_score: ", similarity_score, " ", click_frame_idx, " ", current_frame['file'])
            
            # If similarity is below threshold, we found a significant change
            if similarity_score < similarity_threshold:
                return click_screen_frame, current_frame
    
    return click_screen_frame, None

def calculate_response_time(click_frame, change_frame):
    """Calculate response time between click and visual change"""
    if not click_frame or not change_frame:
        return None
    
    click_time = datetime.fromisoformat(click_frame['time'])
    change_time = datetime.fromisoformat(change_frame['time'])
    
    response_time = (change_time - click_time).total_seconds() * 1000  # Convert to milliseconds
    return response_time

def process_combined_log(combined_log_path, frames_dir, similarity_threshold=0.8):
    """Process the combined log to find click responses"""
    
    # Load the combined log
    with open(combined_log_path, 'r') as f:
        combined_log = json.load(f)
    
    # Find all mouse clicks
    click_frames = find_click_frames(combined_log)
    
    results = []
    
    for click_frame in click_frames:
        # Find the screen frame at click time and the frame where similarity changes
        click_screen_frame, change_screen_frame = find_change_frames(
            combined_log, click_frame, frames_dir, similarity_threshold
        )
        
        if click_screen_frame and change_screen_frame:
            response_time = calculate_response_time(click_screen_frame, change_screen_frame)
            
            result = {
                'click_time': click_frame['time'],
                'click_position': click_frame['position'],
                'click_frame': {
                    'frame': click_screen_frame['frame'],
                    'file': click_screen_frame['file'],
                    'time': click_screen_frame['time']
                },
                'change_frame': {
                    'frame': change_screen_frame['frame'],
                    'file': change_screen_frame['file'],
                    'time': change_screen_frame['time']
                },
                'response_time_ms': response_time
            }
            
            results.append(result)
    
    return results

def update_combined_log_with_responses(combined_log_path, frames_dir, output_path=None, similarity_threshold=0.8):
    """Update the combined log with response time analysis"""
    
    # Load the combined log
    with open(combined_log_path, 'r') as f:
        combined_log = json.load(f)
    
    # Process the log to find responses
    responses = process_combined_log(combined_log_path, frames_dir, similarity_threshold)
    
    # Add responses to the combined log
    combined_log['responses'] = responses
    
    # Save the updated log
    output_path = output_path or combined_log_path
    with open(output_path, 'w') as f:
        json.dump(combined_log, f, indent=4)
    
    return responses

def main():
    """Main function to process the combined log"""
    combined_log_path = "frames/combined_log.json"
    frames_dir = "frames"
    
    # Check if files exist
    if not os.path.exists(combined_log_path):
        print(f"Error: {combined_log_path} not found")
        return
    
    if not os.path.exists(frames_dir):
        print(f"Error: {frames_dir} directory not found")
        return
    
    # Process the log with different similarity thresholds
    thresholds = [0.8]
    
    for threshold in thresholds:
        print(f"\nProcessing with similarity threshold: {threshold}")
        responses = update_combined_log_with_responses(
            combined_log_path, 
            frames_dir, 
            f"frames/combined_log_with_responses.json",
            threshold
        )
        
        print(f"Found {len(responses)} responses")
        for response in responses:
            print(f"Click at {response['click_time']} -> Change at {response['change_frame']['time']} "
                  f"(Response time: {response['response_time_ms']:.2f}ms)")

if __name__ == "__main__":
    main()

