# IRIS (Image Retrieval And Intelligent Search) Smart Gallery

https://irisgallery.streamlit.app/

## Overview
IRIS Smart Gallery is an AI-powered image gallery application that allows users to upload and search for images through captions and tags. The application leverages cutting-edge technologies for **image captioning**, **semantic search**, and **image management**. Powered by the **BLIP** image captioning model and the **SentenceTransformer** model, IRIS Smart Gallery enables efficient and intelligent image retrieval, giving users the ability to search through images by content description or tags.

This application is designed for anyone looking to manage, categorize, and search images quickly using state-of-the-art Natural Language Processing (NLP) and Machine Learning models.

---

## Features
- **AI-powered Image Captioning & Tagging**: Each image uploaded is processed to generate descriptive captions and tags.
- **Search by Caption or Tags**: Find images by querying captions or tags with literal and semantic search methods.
- **Image Compression & Upload**: Compresses images before uploading to save bandwidth and optimize storage.
- **Gallery View**: Browse images in a clean and responsive grid layout with pagination.
- **Manual Refresh**: Refresh the image gallery with a button to load new content or updated images.
- **Progressive Upload Feedback**: Displays upload progress for multiple images with status updates.

---

## Tech Stack
- **Streamlit**: Used for building the interactive web application and handling the user interface.
- **BLIP Model**: Used for generating captions from images. (Salesforce/blip-image-captioning-large)
- **Sentence Transformers**: Used for semantic search to calculate the similarity between queries and image tags.
- **Supabase**: Used for backend storage, database management, and image hosting.
- **TF-IDF Vectorizer**: Used for literal keyword matching in captions and tags for search purposes.
- **PIL (Python Imaging Library)**: Used for image manipulation and compression.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/iris-smart-gallery.git
   cd iris-smart-gallery
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For MacOS/Linux
   venv\Scripts\activate  # For Windows
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory and add your Supabase credentials:
     ```
     SUPABASE_URL=your-supabase-url
     SUPABASE_KEY=your-supabase-key
     ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### 1. **Upload Images**
   - Use the **"Upload a New Image"** section to upload one or more images.
   - Each uploaded image is processed to generate a caption and tags, which are saved to the Supabase database.

### 2. **Search Images**
   - On the left sidebar, use the **"Search by caption or tags"** input field to search for images based on captions or tags.
   - Adjust the **"Relevance Threshold"** slider to control how strict the search results should be. The higher the threshold, the more relevant the results must be to the search query.

### 3. **View Image Gallery**
   - Images are displayed in a grid layout with captions and tags.
   - The gallery supports **pagination**, showing 20 images per page.
   - Use the **"🔄" Refresh** button to reload the gallery and see any new updates.

---

## Features in Detail

### Image Captioning & Tagging
Each image uploaded is processed using the **BLIP image captioning model** to generate a caption. The caption is split into individual words that form the image tags. These captions and tags are then saved in the database and used for search.

### Search Functionality
- **Literal Search**: Uses **TF-IDF** vectorization to compare the query with tags and captions, offering keyword-based search results.
- **Semantic Search**: Utilizes the **SentenceTransformer** model to calculate the semantic similarity between the search query and image tags. This allows for more context-aware search results that go beyond simple keyword matching.

### Image Compression & Upload
Images are compressed to a smaller size to save bandwidth while maintaining visual quality. After compression, the images are uploaded to **Supabase storage** and associated with their captions and tags in the database.

### Gallery View & Pagination
The gallery displays images in a clean, responsive grid format. The gallery supports pagination, allowing users to browse through large image collections in an organized manner.

### Manual Refresh
The **"🔄" Refresh** button allows users to manually refresh the gallery. This feature ensures that the gallery always displays the latest images without having to reload the entire page.

---

## Folder Structure

```bash
iris-smart-gallery/
├── app.py             # Main Streamlit app
├── .env               # Environment variables (for Supabase credentials)
├── requirements.txt   # Python dependencies
└── assets/            # Static assets like images (e.g., logo-modified2.png)
```

---

## Dependencies

- streamlit
- transformers
- sentence-transformers
- scikit-learn
- numpy
- supabase-py
- Pillow
- dotenv

You can install all the required packages with:

```bash
pip install -r requirements.txt
```

---

## Contributing

Feel free to open issues, suggest improvements, or contribute to the project. If you’d like to contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-name`).
6. Create a new pull request.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
- **Salesforce BLIP Model**: [BLIP Image Captioning Model](https://huggingface.co/Salesforce/blip-image-captioning-large)
- **Sentence Transformers**: [Sentence-Transformers Library](https://www.sbert.net/)
- **Supabase**: [Supabase](https://supabase.com/) for backend storage and database management.

---

## Final Thoughts
With **IRIS Smart Gallery**, you can seamlessly manage your image collections, easily upload and retrieve images using both literal and semantic search methods, and enjoy a sleek, intuitive user experience. Whether you're building an AI-powered image gallery or just organizing your own collection, IRIS provides the tools you need to get started quickly and efficiently.