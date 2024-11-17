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
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import mimetypes

st.set_page_config(
    page_title="IRIS Smart Gallery",
    layout="wide",
    page_icon="logo-modified.png",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Model loading functions with proper caching
@st.cache_resource
def load_blip_model():
    """Load BLIP model with Streamlit caching"""
    try:
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        return processor, model
    except Exception as e:
        logger.error(f"Error loading BLIP model: {e}")
        raise


@st.cache_resource
def load_sentence_model():
    """Load sentence transformer model with Streamlit caching"""
    try:
        return SentenceTransformer("all-mpnet-base-v2")
    except Exception as e:
        logger.error(f"Error loading sentence transformer model: {e}")
        raise


# Type definitions and data classes
@dataclass
class ImageData:
    id: int
    image_url: str
    caption: str
    tags: List[str]
    is_processed: bool
    upload_time: str

@dataclass
class UploadResult:
    filename: str
    status: str
    error: Optional[str] = None


class GalleryConfig:
    SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "webp"}
    MAX_IMAGE_SIZE = 500
    COMPRESSION_QUALITY = 75
    IMAGES_PER_PAGE = 20
    DEFAULT_THRESHOLD = 0.5
    MAX_RETRIES = 3


# Environment and configuration
class Config:
    def __init__(self):
        load_dotenv()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")

        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing required environment variables")


# Database operations
class DatabaseManager:
    def __init__(self, config: Config):
        self.supabase = create_client(config.supabase_url, config.supabase_key)

    def save_image(self, image_data: Dict) -> bool:
        try:
            response = self.supabase.table("images").insert(image_data).execute()
            return bool(response)
        except Exception as e:
            logger.error(f"Database save error: {e}")
            return False

    def delete_image(self, image_id: str) -> bool:
        try:
            response = (
                self.supabase.table("images").delete().eq("id", image_id).execute()
            )
            return bool(response)
        except Exception as e:
            logger.error(f"Database delete error: {e}")
            return False

    def fetch_images(self) -> List[ImageData]:
        try:
            response = self.supabase.table("images").select("*").execute()
            if response.data:
                return [ImageData(**img) for img in response.data]
            return []
        except Exception as e:
            logger.error(f"Error fetching images: {e}")
            return []


# Image processing
class ImageProcessor:
    def __init__(self):
        # Load models using cached functions
        self.blip_processor, self.blip_model = load_blip_model()
        self.sentence_model = load_sentence_model()
        self.tfidf_vectorizer = TfidfVectorizer()

    def compress_image(
        self, image_file, max_size=GalleryConfig.MAX_IMAGE_SIZE
    ) -> BytesIO:
        try:
            image = Image.open(image_file).convert("RGB")
            # Maintain aspect ratio while resizing
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            output = BytesIO()
            image.save(output, format="JPEG", quality=GalleryConfig.COMPRESSION_QUALITY)
            output.seek(0)
            return output
        except Exception as e:
            logger.error(f"Image compression error: {e}")
            raise

    def get_image_tags(self, image_path: str) -> Tuple[str, List[str]]:
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.blip_processor(
                images=image, return_tensors="pt"
            ).pixel_values
            output_ids = self.blip_model.generate(
                pixel_values, max_length=100, num_beams=5, early_stopping=True
            )[0]
            caption = self.blip_processor.decode(output_ids, skip_special_tokens=True)
            tags = list(set(caption.lower().split()))
            return caption, tags
        except Exception as e:
            logger.error(f"Tag generation error: {e}")
            raise

    def calculate_similarity(self, query: str, image_data: ImageData) -> float:
        try:
            # Calculate TF-IDF similarity
            text_to_compare = " ".join(image_data.tags + [image_data.caption.lower()])
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                [query.lower(), text_to_compare]
            )
            tfidf_similarity = 1 - cosine(
                tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1]
            )

            # Calculate semantic similarity
            query_embedding = self.sentence_model.encode(query)
            text_embedding = self.sentence_model.encode(text_to_compare)
            semantic_similarity = 1 - cosine(query_embedding, text_embedding)

            # Return weighted average
            return 0.3 * tfidf_similarity + 0.7 * semantic_similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0


# User Interface Components
class GalleryUI:
    def __init__(self, db_manager: DatabaseManager, image_processor: ImageProcessor):
        self.db_manager = db_manager
        self.image_processor = image_processor

    # def setup_page(self):
    #     st.set_page_config(
    #         page_title="IRIS Smart Gallery",
    #         layout="wide",
    #         page_icon="logo-modified.png",
    #     )

    def render_header(self):
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            st.image("logos/logo-modified.png", width=100)
        with col2:
            st.markdown(
                """
                <div style="text-align:center">
                    <h1>IRIS Smart Gallery</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def render_search_sidebar(self):
        st.sidebar.title("Search")
        search_query = st.sidebar.text_input("Search by caption or tags")
        threshold = st.sidebar.slider(
            "Relevance Threshold",
            min_value=0.2,
            max_value=0.5,
            value=GalleryConfig.DEFAULT_THRESHOLD,
            step=0.01,
        )
        return search_query, threshold

    def render_upload_section(self):
        st.subheader("Upload New Images")
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=list(GalleryConfig.SUPPORTED_FORMATS),
            accept_multiple_files=True,
            key="file_uploader",
        )
        return uploaded_files

    def render_gallery(
        self, images: List[ImageData], search_query: str, threshold: float
    ):
        cols = st.columns([0.93, 0.07])
        with cols[0]:
            st.subheader("Image Gallery")
        with cols[1]:
            if st.button("🔄", key="refresh_gallery"):
                st.rerun()

        if not images:
            st.info("No images found in the gallery.")
            return

        # Filter images if search query exists
        if search_query:
            filtered_images = []
            for image in images:
                similarity = self.image_processor.calculate_similarity(
                    search_query, image
                )
                if similarity >= threshold:
                    filtered_images.append((image, similarity))
            filtered_images.sort(key=lambda x: x[1], reverse=True)
            images = [img for img, _ in filtered_images]

        self._render_gallery_grid(images)

    def _render_gallery_grid(self, images: List[ImageData]):
        # Implement pagination
        items_per_page = GalleryConfig.IMAGES_PER_PAGE
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0

        total_pages = (len(images) - 1) // items_per_page + 1

        # Pagination controls
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("Previous", disabled=st.session_state.page_number == 0):
                st.session_state.page_number -= 1
        with col2:
            st.write(f"Page {st.session_state.page_number + 1} of {total_pages}")
        with col3:
            if st.button(
                "Next", disabled=st.session_state.page_number >= total_pages - 1
            ):
                st.session_state.page_number += 1

        # Display images for current page
        start_idx = st.session_state.page_number * items_per_page
        end_idx = start_idx + items_per_page
        page_images = images[start_idx:end_idx]

        cols = st.columns(3)
        for idx, image in enumerate(page_images):
            with cols[idx % 3]:
                self._render_gallery_item(image)

    def _render_gallery_item(self, image: ImageData):
        col_img, col_del = st.columns([0.9, 0.1])
        with col_img:
            st.image(
                image.image_url,
                caption=f"{image.caption}\nTags: {', '.join(image.tags)}",
                use_container_width=True,
            )
        with col_del:
            if st.button("❌", key=f"delete_{image.id}"):
                self._handle_image_deletion(image)

    def _handle_image_deletion(self, image: ImageData):
        if st.modal("Confirm Deletion", key=f"modal_{image.id}"):
            st.warning(
                f"Are you sure you want to delete this image?\n\n**{image.caption}**"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete", key=f"confirm_delete_{image.id}"):
                    if self.db_manager.delete_image(image.id):
                        st.success("Image deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete image.")
            with col2:
                if st.button("Cancel", key=f"cancel_delete_{image.id}"):
                    st.rerun()


def main():
    try:
        # Initialize components
        config = Config()
        db_manager = DatabaseManager(config)
        image_processor = ImageProcessor()
        ui = GalleryUI(db_manager, image_processor)

        # Setup page and render header
        # ui.setup_page()
        ui.render_header()

        # Render search sidebar
        search_query, threshold = ui.render_search_sidebar()

        # Render upload section
        uploaded_files = ui.render_upload_section()

        # Handle file uploads
        if uploaded_files:
            for file in uploaded_files:
                try:
                    compressed_image = image_processor.compress_image(file)
                    caption, tags = image_processor.get_image_tags(file)

                    # Save to database
                    image_data = {
                        "image_url": f"path/to/{file.name}",  # Update with actual path
                        "caption": caption,
                        "tags": tags,
                        "upload_time": datetime.now().isoformat(),
                    }

                    if db_manager.save_image(image_data):
                        st.success(f"Successfully uploaded {file.name}")
                    else:
                        st.error(f"Failed to upload {file.name}")

                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    logger.error(f"Upload error for {file.name}: {e}")

        # Fetch and render gallery
        images = db_manager.fetch_images()
        ui.render_gallery(images, search_query, threshold)

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
    main()
