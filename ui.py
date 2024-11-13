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

# Image upload widget
st.subheader("Upload a New Image")
uploaded_files = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files is not None:
    # Loop through the uploaded files and upload them one by one
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

            # Display a success message temporarily
            success_message = st.empty()  # Create an empty container for the message
            success_message.success(
                f"Image {uploaded_file.name} uploaded successfully!"
            )
            time.sleep(1)  # Wait for 1 second before clearing the message
            success_message.empty()  # Clear the success message
        else:
            st.error("Error uploading image: " + response["error"]["message"])

# Display Image Gallery
st.subheader("Image Gallery")


# Function to fetch images from Supabase table
def fetch_all_images():
    try:
        response = supabase.table("images").select("*").execute()
        if response:
            images = response.data  # List of images with metadata
            if images:
                print(f"Fetched images: {images}")  # Debugging line
            return images
        else:
            print(f"Error fetching images: {response.error_message}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


images = fetch_all_images()

# Display images in a grid format
cols = st.columns(3)
if images:
    for i, image in enumerate(images):
        with cols[i % 3]:
            image_url = image["image_url"]
            image_caption = image["caption"] if "caption" in image else ""

            st.image(image_url, caption=image_caption, use_container_width=True)
else:
    st.warning("No images available.")
