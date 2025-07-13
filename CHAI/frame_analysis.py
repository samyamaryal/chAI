import ollama
import json
from PIL import Image
from pathlib import Path

# Prompt for describing each individual frame (before or after)
single_frame_prompt = (
    "You are analyzing a full screenshot of a webpage. Your description will be used by another model to understand the user intent"
    "Generate a detailed, objective description of all visible UI elements and layout. "
    "Be concise but thorough. Use full sentences."
)

# Prompt to summarize the session based on the sequence of frame descriptions
session_summary_prompt = (
    "You are given a sequence of descriptions of consecutive webpage states representing a user's interactions on a site. "
    "Summarize what the user was trying to do and describe the overall flow of actions. "
    "Highlight key steps, user intent, and any signs of confusion or inefficiency. "
    "Output should be 3-6 bullet points describing the user journey clearly."
)

# Prompt to generate interaction summary between two frames
interaction_summary_prompt = (
    "You are analyzing a user interaction on a webpage. You have descriptions of the page before and after the interaction. "
    "Based on these descriptions, explain what the user likely did and what changed on the page. "
    "Be specific about the user action and the resulting changes. "
    "Output should be 1-2 sentences describing the interaction clearly."
)

def generate_interaction_summary(before_desc, after_desc):
    """Generate an interaction summary using Ollama"""
    try:
        response = ollama.chat(
            model="gemma3:1b",  # Lightweight LLM for on-device reasoning
            messages=[
                {
                    "role": "user",
                    "content": f"{interaction_summary_prompt}\n\nBefore: {before_desc}\n\nAfter: {after_desc}"
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating interaction summary: {e}")
        return f"**Before State:** {before_desc}\n\n**After State:** {after_desc}"


def compare_two(images_list):
    img1, img2 = images_list
    
    img1_desc = ollama.generate(
    model='llava:13b',
    prompt='Describe this webpage screenshot in 3 sentences. Only describe what you see in the image, not what you think it is. Do not make assumptions. Do not hallucinate.',
    images=[f"{img1}"],
    options={'temperature': 0.1}
    )

    # Describe image 2
    img2_desc = ollama.generate(
        model='llava:13b',
        prompt='Describe this webpage screenshot in 3 sentences. Only describe what you see in the image, not what you think it is. Do not make assumptions. Do not hallucinate.',
        images=[f"{img2}"],
        options={'temperature': 0.1}
    )
    return img1_desc['response'], img2_desc['response']


def summarize_sequence(steps, summary_prompt=session_summary_prompt):
    response = ollama.chat(
        model="gemma3:1b",  # Lightweight LLM for on-device reasoning
        messages=[
            {
                "role": "user",
                "content": f"{summary_prompt}\n\n{steps}"
            }
        ]
    )
    return response['message']['content']


def analyze_single_response(response, screen_dir):
    """Analyze a single response (click_frame and change_frame)"""
    frame1_path = f"{screen_dir}/{response['click_frame']['file']}"
    frame2_path = f"{screen_dir}/{response['change_frame']['file']}"
    
    try:
        desc1, desc2 = compare_two([frame1_path, frame2_path])
        
        # Try to get existing interaction summary from the log file
        interaction_summary = None
        try:
            with open("../CHAI/frames/combined_log.json", "r") as f:  
                combined_log = json.load(f)
                for response_itr in combined_log.get("responses", []):
                    if (response_itr.get("click_frame", {}).get("file") == response["click_frame"]["file"] and 
                        response_itr.get("change_frame", {}).get("file") == response["change_frame"]["file"]):
                        if "analysis" in response_itr and "interaction_summary" in response_itr["analysis"]:
                            interaction_summary = response_itr["analysis"]["interaction_summary"]
                            break
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
        
        # If no existing summary found, create a new one using Ollama
        if interaction_summary is None:
            interaction_summary = generate_interaction_summary(desc1, desc2)
        
        return {
            "before_description": desc1,
            "after_description": desc2,
            "interaction_summary": interaction_summary
        }
    except Exception as e:
        print(f"Error analyzing response: {e}")
        return {
            "before_description": "Error analyzing image",
            "after_description": "Error analyzing image", 
            "interaction_summary": "Error in analysis"
        }


def analyze_all_responses(responses, screen_dir):
    """Analyze all responses and add analysis to each response object"""
    all_step_descriptions = []
    
    for i, response in enumerate(responses):
        print(f"Analyzing response {i+1}/{len(responses)}")
        
        # Analyze this response
        analysis = analyze_single_response(response, screen_dir)
        
        # Add analysis to the response object
        response['analysis'] = analysis
        
        # Add descriptions to the overall sequence
        all_step_descriptions.append(analysis['before_description'])
        all_step_descriptions.append(analysis['after_description'])
    
    # Generate overall session summary
    try:
        overall_summary = summarize_sequence("\n\n".join(all_step_descriptions))
    except Exception as e:
        print(f"Error generating overall summary: {e}")
        overall_summary = "Error generating overall summary"
    
    return responses, overall_summary


def run_frame_analysis(log_file_path, screen_dir_path):
    """Main function to run frame analysis on a log file"""
    try:
        # Load the log file
        with open(log_file_path, 'r') as f:
            logs = json.load(f)
        
        responses = logs.get('responses', [])
        
        if not responses:
            return None, "No responses found in log file"
        
        # Analyze all responses
        updated_responses, session_summary = analyze_all_responses(responses, screen_dir_path)
        
        # Add session summary to the logs
        logs['session_analysis'] = {
            "summary": session_summary,
            "total_responses": len(updated_responses)
        }
        
        # Update the responses in the logs
        logs['responses'] = updated_responses
        
        # Save the updated logs back to the original file
        with open(log_file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return logs, session_summary
        
    except Exception as e:
        return None, f"Error in frame analysis: {e}"


# Only run if this script is executed directly
if __name__ == "__main__":
    # Current working directory
    ROOT_DIR = Path.cwd().parent 
    screen_dir = f"{ROOT_DIR}/CHAI/frames/screen"
    log_file = "../CHAI/frames/combined_log.json"
    
    result, summary = run_frame_analysis(log_file, screen_dir)
    if result:
        print("Analysis complete! Updated logs saved.")
        print(f"Session Summary: {summary}")
    else:
        print(f"Error: {summary}")
