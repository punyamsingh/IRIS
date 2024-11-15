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

# Display the logo and title in the same row
col1, col2 = st.columns([0.1, 0.9])  # Adjust the column widths as needed
with col1:
    st.image("logo-modified2.png", width=100)
with col2:
    st.markdown(
        """
        <style>
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }
        </style>
        <div class="title-container">
            <h1>IRIS Smart Gallery</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("Upload images to view, tag, and search.")

# Caching for models
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

# Image compression function
def compress_image(image_file, max_size=500):
    image = Image.open(image_file).convert("RGB")
    output = BytesIO()
    image.save(output, format="JPEG", quality=75)
    output.seek(0)
    return output

# Function to generate image captions and tags
@st.cache_data
def get_image_tags(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(
        pixel_values, max_length=100, num_beams=5, early_stopping=True
    )[0]
    caption = processor.decode(output_ids, skip_special_tokens=True)
    tags = set(caption.lower().split())
    return caption, list(tags)

@st.cache_data
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

# Sidebar search functionality
st.sidebar.title("Search")
search_query = st.sidebar.text_input("Search by caption or tags")
threshold = st.sidebar.slider(
    "Relevance Threshold", min_value=0.2, max_value=0.5, value=0.5, step=0.01
)

# Image Upload
st.subheader("Upload a New Image")
uploaded_files = st.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        caption, tags = process_image(uploaded_file)
        file_name = f"{int(time.time())}_{uploaded_file.name}"
        file_data = compress_image(uploaded_file).read()
        response = supabase.storage.from_("images").upload(
            file_name, file_data, {"contentType": "image/jpeg"}
        )
        if response:
            image_url = supabase.storage.from_("images").get_public_url(file_name)
            save_image_to_db(image_url, caption, tags)
        else:
            st.error(f"Error uploading image: {uploaded_file.name}")

    # Clear cached data to refresh the gallery
    st.cache_data.clear()
    st.rerun()  # Automatically rerun the script

# Functions to fetch and rank images
@st.cache_data
def fetch_images():
    response = supabase.table("images").select("*").execute()
    if response:
        return response.data
    else:
        return []

@st.cache_data
def get_literal_similarity(query, image_tags):
    all_tags = " ".join(image_tags)
    X = tfidf_vectorizer.fit_transform([query, all_tags])
    return 1 - cosine(X[0].toarray().flatten(), X[1].toarray().flatten())

@st.cache_data
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

# Pagination
images_per_page = 8
all_images = fetch_images()

if search_query:
    all_images = rank_images(all_images, search_query, threshold)

if len(all_images) == 0:
    st.info("No images found in the gallery.")
else:
    total_pages = (len(all_images) - 1) // images_per_page + 1
    current_page = st.session_state.get("current_page", 1)
    if current_page < 1 or current_page > total_pages:
        current_page = 1

    # Pagination Controls
    st.markdown("<style>.pagination { text-align: center; }</style>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="pagination">
            {' '.join(
                f'''<button onclick="window.location.href='#{page}'" style="padding: 5px 10px; margin: 0 5px; {"background: lightblue;" if page == current_page else ""}">{page}</button>'''
                for page in range(1, total_pages + 1)
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )

    start_idx = (current_page - 1) * images_per_page
    end_idx = start_idx + images_per_page
    images = all_images[start_idx:end_idx]

    # Display Gallery
    cols = st.columns(4)
    for i, image in enumerate(images):
        with cols[i % 4]:
            caption_text = f"{image['caption']} | Tags: {', '.join(image['tags'])}"
            st.image(image["image_url"], caption=caption_text, use_container_width=True)
