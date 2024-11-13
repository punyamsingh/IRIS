import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client
import time

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Streamlit configuration
st.set_page_config(page_title="IRIS Smart Gallery", layout="wide")
st.title("IRIS Smart Gallery")
st.write("Upload images to view, tag, and search.")

# Image upload widget (allow multiple files)
st.subheader("Upload New Images")
uploaded_files = st.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Generate a unique file name based on timestamp and original name
        file_name = f"{int(time.time())}_{uploaded_file.name}"

        # Read the uploaded file content as bytes
        file_data = uploaded_file.read()

        # Upload the file to Supabase Storage
        response = supabase.storage.from_("images").upload(
            file_name, file_data, {"contentType": "image/jpeg"}
        )

        if response is not None:
            # Generate a public URL for the uploaded image
            image_url = supabase.storage.from_("images").get_public_url(file_name)

            # Save image info (including URL) to the Supabase database
            supabase.table("images").insert({"image_url": image_url}).execute()

            st.success(f"Image {uploaded_file.name} uploaded successfully!")
        else:
            st.error(
                f"Error uploading image {uploaded_file.name}: "
                + response["error"]["message"]
            )

# Display Image Gallery
st.subheader("Image Gallery")


# Function to fetch images from Supabase table
def fetch_all_images():
    try:
        response = supabase.table("images").select("*").execute()
        if response:
            images = response.data  # List of images with metadata
            if images:
                return images
            else:
                print("No images found.")
                return []
        else:
            print(f"Error fetching images: {response.error_message}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


images = fetch_all_images()

# Display images in a grid format
if images:
    cols = st.columns(3)
    for i, image in enumerate(images):
        with cols[i % 3]:
            image_url = image["image_url"]
            image_caption = image.get("caption", "No caption available")
            st.image(image_url, caption=image_caption, use_container_width=True)
else:
    st.warning("No images available.")
