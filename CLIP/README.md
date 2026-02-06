# CLIP Image–Text Matching (Contrastive Learning) — Experiment

Big Data Analysis (BIT, 2025–2026) — Lab  
Task: explain contrastive learning, explain how CLIP works, and run an image–text classification experiment using CLIP.

## 1) What is Contrastive Learning?
**Contrastive learning** is a representation learning approach that trains a model to bring **semantically related pairs** (positive pairs) closer in an embedding space and push **unrelated pairs** (negative pairs) farther apart.  
In practice, this is often done with a contrastive loss (e.g., InfoNCE), where the model learns to maximize similarity for matched pairs and minimize similarity for mismatched pairs.

## 2) How CLIP Works
**CLIP (Contrastive Language–Image Pre-training)** learns a shared embedding space for images and text:
- An **image encoder** converts an image into a vector embedding.
- A **text encoder** converts a text prompt into a vector embedding.
- The model is trained contrastively on large (image, caption) pairs so that matching image–text pairs have high cosine similarity.
At inference time, you can classify an image by computing similarities between the image embedding and a set of text embeddings, then selecting the best match (highest similarity).

## 3) Experiment: Image Classification with CLIP
This repo contains a script that:
1. Loads **10+ images** from `images/`
2. Uses CLIP (`ViT-B/32`) to compute **image and text embeddings**
3. Builds a **cosine similarity matrix**
4. Visualizes the matrix as a heatmap
5. Reports:
   - best-matching text for each image
   - accuracy (correct top-1 match)
   - top-3 matches per image

### 3.1 Dataset (Images + Text Prompts)
- Images are stored in: `images/`
- Text prompts are defined in the code as `text_descriptions`
- Image filenames are mapped via `original_filenames` (supports `.jpg/.png/.jpeg`)

> Note: Filenames in this project may contain non-English characters. Make sure your OS and Git handle UTF-8 filenames correctly.
