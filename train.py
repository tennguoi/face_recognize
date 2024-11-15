# face_train.py

import torch
import numpy as np
from PIL import Image
import os
import json
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

def load_and_preprocess_image(image_path):
    """Load and preprocess image for face recognition"""
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    return transform(img)

def main():
    print("\n[INFO] Starting face embedding generation...")
    
    # Initialize FaceNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Path settings
    images_path = './images/'
    
    # Dictionary to store embeddings
    embeddings_dict = {}
    
    # Process each image
    for image_file in os.listdir(images_path):
        if not image_file.endswith('.jpg'):
            continue
            
        # Get user ID from filename
        user_id = int(image_file.split('-')[1])
        
        # Load and preprocess image
        image_path = os.path.join(images_path, image_file)
        try:
            img_tensor = load_and_preprocess_image(image_path).unsqueeze(0).to(device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = facenet(img_tensor).cpu().numpy()
            
            # Store or update embedding
            if user_id in embeddings_dict:
                embeddings_dict[user_id] = np.mean([embeddings_dict[user_id], embedding[0]], axis=0)
            else:
                embeddings_dict[user_id] = embedding[0]
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    # Save embeddings
    np.savez('face_embeddings.npz', embeddings=embeddings_dict)
    
    print(f"\n[INFO] Successfully generated embeddings for {len(embeddings_dict)} users")
    print("[INFO] Saved embeddings to 'face_embeddings.npz'")

if __name__ == "__main__":
    main()