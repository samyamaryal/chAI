import json
from gaze_heatmap import get_gaze_freq
import ollama

with open("./frames/gaze_log.json", "r") as f:
    gaze_log = json.load(f)
freq = get_gaze_freq(gaze_log)

with open("./frames/combined_log.json", "r") as f:
    combined_log = json.load(f)

def get_max_occurring_emotion(combined_log):
    max_emotion = {}
    for frame in combined_log["webcam"]:
        if frame["emotion"] is not None:
            emotion = frame["emotion"]
            frame_max_emotion = max(emotion, key=lambda k: float(emotion[k]))
            if frame_max_emotion not in max_emotion:
                max_emotion[frame_max_emotion] = 0
            max_emotion[frame_max_emotion] += 1
    return max(max_emotion, key=max_emotion.get)

def get_number_of_mouse_clicks(combined_log):
    mouse_clicks = 0
    for frame in combined_log["mouse"]:
        if frame["pressed"] is True:
            mouse_clicks += 1
    return mouse_clicks

def get_number_of_keyboard_events(combined_log):
    keyboard_events = 0
    for frame in combined_log["keyboard"]:
        if frame["event"] is "press":
            keyboard_events += 1
    return keyboard_events

# max_emotion = get_max_occurring_emotion(combined_log)
# print(max_emotion)

# number_of_mouse_clicks = get_number_of_mouse_clicks(combined_log)
# print(number_of_mouse_clicks)

# number_of_keyboard_events = get_number_of_keyboard_events(combined_log)
# print(number_of_keyboard_events)

def get_summary(path):
    with open(f"./{path}/gaze_log.json", "r") as f:
        gaze_log = json.load(f)

    with open(f"./{path}/combined_log.json", "r") as f:
        combined_log = json.load(f)


    gaze_freq = get_gaze_freq(gaze_log)
    print(gaze_freq)

    max_emotion = get_max_occurring_emotion(combined_log)
    print(max_emotion)

    number_of_mouse_clicks = get_number_of_mouse_clicks(combined_log)
    print(number_of_mouse_clicks)

    number_of_keyboard_events = get_number_of_keyboard_events(combined_log)
    print(number_of_keyboard_events)
    return gaze_freq, max_emotion, number_of_mouse_clicks, number_of_keyboard_events


def get_ollama_summary(gaze_freq, max_emotion, number_of_mouse_clicks, number_of_keyboard_events):
    prompt = f"""
    You are a helpful assistant that summarizes the user's interaction with the UI. The following are the analytics for the user's performed actions:
    Here is the gaze frequency in a 3x3 grid that is how many times the user looked at each sector (top left = 1, top middle = 2, top right = 3, middle left = 4, middle middle = 5, middle right = 6, bottom left = 7, bottom middle = 8, bottom right = 9): {gaze_freq}
    Here is the max emotion: {max_emotion}
    Here is the number of mouse clicks: {number_of_mouse_clicks}
    Here is the number of keyboard events: {number_of_keyboard_events}
    """

    response = ollama.chat(
    model='llama3.1:latest',
    messages=[
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]
    )
    print(response['message']['content'])
    return response['message']['content']


