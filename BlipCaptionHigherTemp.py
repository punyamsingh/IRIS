import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json

# Load the pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate long captions using BLIP
def get_image_tags(image_path):
    """Use BLIP to generate detailed captions for the image."""
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate a caption with increased max_length for more verbosity
    out = model.generate(**inputs, max_length=50, num_beams=5, temperature=0.7)  # num_beams and temperature can control output quality
    
    # Decode the output caption (this should be more verbose)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Generated Caption for {image_path}: {caption}")
    
    # Extract tags (split caption into words or use NLP techniques to extract key objects)
    tags = set(caption.lower().split())
    
    return list(tags)

# Process all images in a directory and save results to JSON
def process_directory(directory_path, output_file='image_tags.json'):
    """Process all images in a directory, generate tags using BLIP, and save to JSON file."""
    image_tags_mapping = {}
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            # Step 1: Get detailed tags from the image using BLIP
            tags = get_image_tags(image_path)
            # Step 2: Store the mapping
            image_tags_mapping[filename] = tags
            print(f"Processed {filename} with tags: {tags}")
    
    # Save the image-to-tags mapping to a JSON file
    with open(output_file, 'w') as f:
        json.dump(image_tags_mapping, f, indent=4)

# Example usage
process_directory('/content')
