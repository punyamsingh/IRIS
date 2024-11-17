import streamlit as st
import os
import time
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Streamlit Configuration
st.set_page_config(
    page_title="IRIS Smart Gallery", layout="wide", page_icon="logo-modified.png"
)


# Load models
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    return processor, model


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-mpnet-base-v2")


processor, model = load_blip_model()
st_model = load_sentence_model()
tfidf_vectorizer = TfidfVectorizer()

# Sidebar search functionality
st.sidebar.title("Search")
search_query = st.sidebar.text_input("Search by caption or tags")
threshold = st.sidebar.slider(
    "Relevance Threshold", min_value=0.2, max_value=0.5, value=0.5, step=0.01
)


# Image compression function
def compress_image(image_file, max_size=500):
    image = Image.open(image_file).convert("RGB")
    output = BytesIO()
    image.save(output, format="JPEG", quality=75)
    output.seek(0)
    return output


# Function to generate image captions and tags
def get_image_tags(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(
        pixel_values, max_length=100, num_beams=5, early_stopping=True
    )[0]
    caption = processor.decode(output_ids, skip_special_tokens=True)
    tags = set(caption.lower().split())
    return caption, list(tags)


def save_image_to_db(image_url, caption, tags):
    response = (
        supabase.table("images")
        .insert({"image_url": image_url, "caption": caption, "tags": tags})
        .execute()
    )
    if not response:
        st.error("Error saving image to the database.")


def process_image(image_file):
    image_path = f"temp_{int(time.time())}.jpg"
    compressed_image = compress_image(image_file)
    with open(image_path, "wb") as f:
        f.write(compressed_image.getbuffer())
    caption, tags = get_image_tags(image_path)
    os.remove(image_path)
    return caption, tags


# Fetch and rank images
def fetch_images():
    response = supabase.table("images").select("*").execute()
    if response:
        return response.data
    else:
        return []


def get_literal_similarity(query, image_tags):
    all_tags = " ".join(image_tags)
    X = tfidf_vectorizer.fit_transform([query, all_tags])
    return 1 - cosine(X[0].toarray().flatten(), X[1].toarray().flatten())


def get_semantic_similarity(query, image_tags):
    query_embed = st_model.encode(query)
    tags_embed = st_model.encode(image_tags)
    tags_mean_embed = np.mean(tags_embed, axis=0)
    return 1 - cosine(query_embed, tags_mean_embed)


def rank_images(images, query, threshold):
    ranked_images = []
    for image in images:
        literal_sim = get_literal_similarity(query, image["tags"])
        semantic_sim = get_semantic_similarity(query, image["tags"])
        total_score = literal_sim + semantic_sim
        if total_score >= threshold:
            ranked_images.append((image, total_score))
    return [img[0] for img in sorted(ranked_images, key=lambda x: x[1], reverse=True)]


def delete_from_database(image_id):
    """
    Deletes an image record from the Supabase database.

    Args:
        image_id (str): The ID of the image to delete.
    """
    response = supabase.table("images").delete().eq("id", image_id).execute()

    st.success("Image deleted successfully.")
    # if response["error"] is None:
    #     st.success("Image deleted successfully.")
    # else:
    #     st.error("Failed to delete image.")
    #     st.error(response["error"])


# def delete_from_bucket(image_url):
#     """
#     Deletes an image file from the Supabase storage bucket.

#     Args:
#         image_url (str): The URL of the image to delete.
#     """
#     # Extract the bucket name and file path from the URL
#     bucket_name = "your_bucket_name"  # Replace with your Supabase bucket name
#     file_path = image_url.split(
#         f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/"
#     )[-1]

#     response = supabase.storage.from_(bucket_name).remove([file_path])

#     if response.get("error") is None:
#         st.success("Image deleted from the storage bucket.")
#     else:
#         st.error("Failed to delete image from the storage bucket.")
#         st.error(response.get("error"))


# def delete_image(image_id, image_url):
#     # Delete from database
#     delete_from_database(image_id)  # Custom function to delete entry from DB

# # Delete from storage bucket
# delete_from_bucket(image_url)  # Custom function to delete from Supabase bucket


@st.fragment
def gallery():
    images_per_page = 20
    all_images = fetch_images()

    # Sort images by upload time (newest first)
    all_images = sorted(
        all_images, key=lambda x: x.get("upload_time", ""), reverse=True
    )

    # Refresh button
    cola, colb = st.columns([0.93, 0.07])  # Adjust column widths
    with cola:
        st.subheader("Image Gallery")
    with colb:
        if st.button("🔄", key="refresh_gallery"):
            st.rerun()

    if search_query:
        all_images = rank_images(all_images, search_query, threshold)

    # Handle empty gallery
    if len(all_images) == 0:
        st.info("No images found in the gallery.")
        return

    # Pagination setup
    total_pages = max((len(all_images) - 1) // images_per_page + 1, 1)
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1

    current_page = st.session_state["current_page"]

    # Boundary checks
    current_page = max(1, min(current_page, total_pages))
    st.session_state["current_page"] = current_page

    # Paginated images
    start_idx = (current_page - 1) * images_per_page
    end_idx = start_idx + images_per_page
    images = all_images[start_idx:end_idx]

    # Pagination controls (top)
    st.divider()
    pagination_controls(total_pages, current_page, position="top")
    st.divider()

    # Display images in a grid layout
    num_columns = 3
    cols = st.columns(num_columns)

    # Define the delete confirmation dialog
    @st.dialog("Confirm Deletion")
    def confirm_delete(image_id, caption_text):
        st.warning(f"Are you sure you want to delete this image?\n\n**{caption_text}**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete"):
                delete_from_database(image_id)  # Function to delete the image
                st.success("Image deleted successfully.")
                st.rerun()
        with col2:
            if st.button("No, cancel"):
                st.info("Deletion canceled.")
                st.rerun()

    for i, image in enumerate(images):
        with cols[i % num_columns]:
            # Display the image with a delete button

            col_img, col_del = st.columns([0.9, 0.2])
            with col_img:
                caption_text = (
                    f"{image['caption']} | Tags: {', '.join(image['tags'])}"
                    if image["tags"]
                    else "None"
                )
                st.image(
                    image["image_url"],
                    caption=caption_text,
                    use_container_width=True,
                )
            with col_del:
                if st.button("❌", key=f"delete_{image['id']}"):
                    confirm_delete(image["id"], caption_text)

    # Pagination controls (bottom)
    st.divider()
    pagination_controls(total_pages, current_page, position="bottom")
    st.divider()


def pagination_controls(total_pages, current_page, position):
    """Renders pagination controls."""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        if st.button(f"<< First ({position})", disabled=current_page == 1):
            st.session_state["current_page"] = 1
            st.rerun()
    with col2:
        if st.button(f"< Previous ({position})", disabled=current_page == 1):
            st.session_state["current_page"] -= 1
            st.rerun()
    with col3:
        st.write(f"Page {current_page} of {total_pages}")
    with col4:
        if st.button(f"Next > ({position})", disabled=current_page == total_pages):
            st.session_state["current_page"] += 1
            st.rerun()
    with col5:
        if st.button(f"Last >> ({position})", disabled=current_page == total_pages):
            st.session_state["current_page"] = total_pages
            st.rerun()


@st.fragment
def uploader():
    st.subheader("Upload a New Image")

    # Initialize session state for uploaded files
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    uploaded_files = st.file_uploader(
        "Choose images...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    # Store newly uploaded files in session state
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in [f.name for f in st.session_state["uploaded_files"]]:
                st.session_state["uploaded_files"].append(file)

    # Process uploaded files
    if st.session_state["uploaded_files"]:
        uploaded_status = []
        progress_bar = st.progress(0)  # Progress bar for tracking uploads
        total_files = len(st.session_state["uploaded_files"])

        for idx, uploaded_file in enumerate(st.session_state["uploaded_files"]):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    caption, tags = process_image(uploaded_file)
                    file_name = f"{int(time.time())}_{uploaded_file.name}"
                    file_data = compress_image(uploaded_file).read()
                    response = supabase.storage.from_("images").upload(
                        file_name, file_data, {"contentType": "image/jpeg"}
                    )
                    if response:
                        image_url = supabase.storage.from_("images").get_public_url(
                            file_name
                        )
                        save_image_to_db(image_url, caption, tags)
                        uploaded_status.append((uploaded_file.name, "success"))
                    else:
                        uploaded_status.append((uploaded_file.name, "failed"))
                except Exception as e:
                    uploaded_status.append((uploaded_file.name, f"error: {e}"))
                progress_bar.progress((idx + 1) / total_files)

        # Clear the uploaded files from session state after processing
        st.session_state["uploaded_files"] = []

        for file_name, status in uploaded_status:
            if status == "success":
                st.toast(f"{file_name} uploaded successfully.")
            elif status == "failed":
                st.toast(f"Error uploading {file_name}.")
            else:
                st.toast(f"{file_name} encountered an error: {status}")


@st.fragment
def header():
    col1, col2 = st.columns([0.1, 0.9])  # Adjust the column widths as needed
    with col1:
        st.image("logos/logo-modified.png", width=100)
    with col2:
        st.markdown(
            """
            <div style="text-align:center;">
                <h1>IRIS Smart Gallery</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

header()  # Displays the header with the refresh button
uploader()  # Displays the uploader
gallery()  # Displays the gallery
