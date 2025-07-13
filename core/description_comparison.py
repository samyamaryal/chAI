import ollama
from PIL import Image
import base64
import io
import json

descriptor_prompt = (
    "You are analyzing a full screenshot of a computer webpage to generate a detailed, structured description. "
    "Include visible elements such as: header, main content, buttons, forms, text sections, menus, images, popups, and visible changes like alerts or modals. "
    "Be objective and systematic. Use full sentences and group elements logically (e.g., layout, navigation, content, and calls to action). "
    "Assume this will be used for visual regression or change detection between versions of the same webpage, so accuracy is critical."
)

compare_captions_prompt = (
    "You are analyzing how a webpage has changed from an earlier state (caption 1) to a later state (caption 2). "
    "Identify only the differences, focusing on what has been added, removed, or changed. "
    "Frame the differences as transitions (e.g., 'X was added', 'Y changed to Z', 'A was removed'). "
    "Be extremely concise and use bullet points. Ignore elements that stayed the same. "
    "This is part of a visual tracking system that monitors updates between two webpages from the same site."
)

def generate_descriptions(img_path, prompt=descriptor_prompt):

    # Load image and convert to base64
    with Image.open(img_path).convert("RGB") as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")


    # Run inference using LLaVA
    response = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [img_base64]
            }
        ]
    )
    buffered.close()
    return response['message']['content'] 

def compare_captions(caption1, caption2, system_prompt=compare_captions_prompt):
    response = ollama.chat(
    model='gemma3:1b',
    messages=[
        {
            "role": "user",
            "content": f"{system_prompt}\n\n1. {caption1}\n2. {caption2}."
        }
    ]
    )
    return response['message']['content']   


def compare_images(image1_path, image2_path):

    desc1 = generate_descriptions(image1_path)
    desc2 = generate_descriptions(image2_path) 

    diff = compare_captions(desc1, desc2)

    return json.dumps({
        "image1": image1_path,
        "image2": image2_path,
        "image1_description": desc1,
        "image2_description": desc2,
        "differences": diff
    }) 

if __name__ == "__main__":
    print(compare_images("image.png", "image2.png"))

