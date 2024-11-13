import streamlit as st
import os
import time
import numpy as np
from PIL import Image
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
# Display the logo on the main page
# st.image("logo-modified2.png", width=100)
# st.title("IRIS Smart Gallery")

# Display the logo and title in the same row
col1, col2 = st.columns([0.1, 0.9])  # Adjust the column widths as needed

with col1:
    st.image("logo-modified2.png", width=100)

with col2:
    # Using markdown to align the title vertically in the center
    st.markdown(
        """
        <style>
        .title-container {
            display: flex;
            justify-content: center;
            align-items: top;
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


@st.cache_resource
def load_blip_model():
    # Initialize the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    return processor, model


@st.cache_resource
def load_sentence_model():
    # Load the Sentence Transformer model
    return SentenceTransformer("all-mpnet-base-v2")


# Load models
processor, model = load_blip_model()
st_model = load_sentence_model()
tfidf_vectorizer = TfidfVectorizer()


# Function to generate image captions and tags
@st.cache_data
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


@st.cache_data
def save_image_to_db(image_url, caption, tags):
    """Save the image details (URL, caption, tags) to Supabase."""
    response = (
        supabase.table("images")
        .insert({"image_url": image_url, "caption": caption, "tags": tags})
        .execute()
    )
    if response:
        st.toast("Image uploaded and processed successfully!")
        st.rerun()
    else:
        st.error("Error saving image to the database.")


def process_image(image_file):
    image_path = f"temp_{int(time.time())}.jpg"
    with open(image_path, "wb") as f:
        f.write(image_file.getbuffer())
    caption, tags = get_image_tags(image_path)
    os.remove(image_path)
    return caption, tags


# Sidebar for search
st.sidebar.title("Search")
search_query = st.sidebar.text_input("Search by caption or tags")

st.sidebar.markdown("**Select Relevance Threshold**")
threshold = st.sidebar.slider(
    "", min_value=0.2, max_value=0.5, value=0.5, step=0.01, label_visibility="collapsed"
)

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


# Fetch and display images in the gallery
images = fetch_images()
if search_query:
    images = rank_images(images, search_query, threshold)

if images:
    cols = st.columns(4)
    for i, image in enumerate(images):
        with cols[i % 4]:
            caption_text = f"{image['caption']} | Tags: {', '.join(image['tags'])}"
            st.image(image["image_url"], caption=caption_text, use_container_width=True)
else:
    st.warning("No images available matching your search.")
