import ollama
from PIL import Image
import base64
import io
import json

descriptor_prompt = "Give a succinct, detailed description of this website. Include all relevant information that a user would need to know about the website."
compare_captions_prompt = "Compare these captions about a webpage and highlight the differences. Keep it very concise & give bullet points only"

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

