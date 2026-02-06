import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity

print("All libraries loaded!")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(f"CLIP model loaded on device: {device}")

text_descriptions = [
    "urban bicycle",
    "book and glasses", 
    "red sports car",
    "musical symbol treble clef",
    "pepperoni pizza",
    "snowy mountains",
    "cactus in desert",
    "cup of coffee",
    "glass skyscraper",
    "calico cat"
]

original_filenames = [
    "Городской велосипед",
    "Книга и очки", 
    "Красный спортивный автомобиль",
    "Музыкальный символ скрипичного ключа",
    "Пепперони",
    "заснеженные горы",
    "кактус в пустыне",
    "кофе",
    "небоскреб",
    "трехцветная кошка"
]

print("Loading images...")

images = []
found_image_paths = []

for i, filename in enumerate(original_filenames):
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
        file_path = f"images/{filename}{ext}"
        if os.path.exists(file_path):
            try:
                image = Image.open(file_path)
                images.append(preprocess(image))
                found_image_paths.append(file_path)
                print(f"Found: {file_path} -> '{text_descriptions[i]}'")
                break
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    else:
        print(f"File not found: images/{filename}.jpg (or .png)")

print(f"\nSuccessfully loaded {len(images)} images from {len(original_filenames)}")

if images:
    print("Calculating embeddings...")
    
    image_input = torch.tensor(np.stack(images)).to(device)
    text_tokens = clip.tokenize(text_descriptions[:len(images)]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print("Embeddings calculated!")
    
    similarity_matrix = cosine_similarity(image_features.cpu().numpy(), 
                                        text_features.cpu().numpy())
    
    print("Similarity matrix:")
    print(similarity_matrix.round(3))
    
    plt.figure(figsize=(16, 12))
    im = plt.imshow(similarity_matrix, cmap='YlOrRd', interpolation='nearest')
    cbar = plt.colorbar(im)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    plt.xticks(range(len(text_descriptions[:len(images)])), text_descriptions[:len(images)], rotation=45, ha='right')
    plt.yticks(range(len(found_image_paths)), [os.path.basename(p) for p in found_image_paths])
    
    plt.xlabel('Text Descriptions', fontsize=12)
    plt.ylabel('Image Files', fontsize=12)
    plt.title('CLIP Similarity Matrix: Images vs Text Descriptions', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("MATCHING RESULTS")
    print("="*60)
    
    correct_pairs = 0
    total_pairs = len(images)
    
    for i in range(total_pairs):
        best_text_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i, best_text_idx]
        
        status = "CORRECT" if best_text_idx == i else " ERROR"
        print(f"\nImage {i+1} ('{os.path.basename(found_image_paths[i])}'): {status}")
        print(f"   Expected text:     '{text_descriptions[i]}'")
        print(f"   Predicted text:    '{text_descriptions[best_text_idx]}'")
        print(f"   Similarity: {best_score:.3f}")
        
        if best_text_idx == i:
            correct_pairs += 1
    
    accuracy = correct_pairs / total_pairs
    print("\n" + "="*60)
    print(f"FINAL ACCURACY: {accuracy:.1%} ({correct_pairs}/{total_pairs})")
    print("="*60)
    
    print("\nDETAILED ANALYSIS (TOP-3 for each image):")
    print("="*60)
    
    for i in range(len(images)):
        print(f"\nImage {i+1} ('{os.path.basename(found_image_paths[i])}'):")
        
        scores = similarity_matrix[i]
        top3_indices = np.argsort(scores)[-3:][::-1]
        
        for rank, text_idx in enumerate(top3_indices):
            score = scores[text_idx]
            marker = "🎯" if text_idx == i else "   "
            print(f"   {marker} {rank+1}. '{text_descriptions[text_idx]}' - {score:.3f}")
    
    print("\nEXPERIMENT COMPLETED!")
    
else:
    print("\nNo images found for processing!")