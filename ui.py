import streamlit as st
import os
import json
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)


def get_image_tags(image_path):
    """Generate detailed captions and tags using the BLIP model."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(
        pixel_values, max_length=100, num_beams=5, early_stopping=True
    )[0]
    caption = processor.decode(output_ids, skip_special_tokens=True)

    # Extract tags (split caption into words or use NLP techniques to extract key objects)
    tags = set(caption.lower().split())
    return caption, list(tags)


def process_image(image_file):
    """Process a single image file: generate captions and tags."""
    image_path = f"temp_{int(time.time())}.jpg"
    with open(image_path, "wb") as f:
        f.write(image_file.getbuffer())

    caption, tags = get_image_tags(image_path)
    os.remove(image_path)  # Clean up the temp file
    return caption, tags


def save_image_to_db(image_url, caption, tags):
    """Save the image details (URL, caption, tags) to Supabase."""
    response = (
        supabase.table("images")
        .insert({"image_url": image_url, "caption": caption, "tags": tags})
        .execute()
    )

    if response.status_code == 201:
        st.success("Image uploaded and processed successfully!")
    else:
        st.error(f"Error saving image to the database: {response.error_message}")


# Streamlit Configuration
st.set_page_config(page_title="IRIS Smart Gallery", layout="wide")
st.title("IRIS Smart Gallery")
st.write("Upload images to view, tag, and search.")

# Sidebar for search
st.sidebar.title("Search")
search_query = st.sidebar.text_input("Search by caption or tags")

# Image Upload
st.subheader("Upload a New Image")
uploaded_files = st.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Step 1: Process the uploaded image and get the caption and tags
        caption, tags = process_image(uploaded_file)

        # Step 2: Upload the image to Supabase Storage
        file_name = f"{int(time.time())}_{uploaded_file.name}"
        file_data = uploaded_file.read()
        response = supabase.storage.from_("images").upload(
            file_name, file_data, {"contentType": "image/jpeg"}
        )

        if response:
            # Step 3: Generate the public URL for the uploaded image
            image_url = supabase.storage.from_("images").get_public_url(file_name)

            # Step 4: Save the image details (URL, caption, tags) to the database
            save_image_to_db(image_url, caption, tags)

        else:
            st.error(f"Error uploading image: {uploaded_file.name}")

    # Reload the page to update the gallery with newly uploaded images
    st.experimental_rerun()

# Display Image Gallery
st.subheader("Image Gallery")


# Function to fetch images based on search query
def fetch_images(query=None):
    """Fetch images from Supabase based on search query."""
    try:
        response = supabase.table("images").select("*").execute()
        if response:
            images = response.data
            if query:
                # Filter images by caption or tags based on the query
                images = [
                    image
                    for image in images
                    if query.lower() in image["caption"].lower()
                    or any(query.lower() in tag for tag in image["tags"])
                ]
            return images
        else:
            return []
    except Exception as e:
        print(f"Error fetching images: {e}")
        return []


# Fetch and display images in the gallery
images = fetch_images(search_query)

if images:
    cols = st.columns(3)
    for i, image in enumerate(images):
        with cols[i % 3]:
            st.image(
                image["image_url"],
                caption=f"{image['caption']} | Tags: {', '.join(image['tags'])}",
                use_container_width=True,
            )
else:
    st.warning("No images available matching your search.")
