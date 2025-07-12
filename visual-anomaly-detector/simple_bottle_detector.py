import os
import torch
from torchvision import models, transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# ✅ Set paths
DATASET_PATH = "./normal_samples/bottle"
TRAIN_DIR = os.path.join(DATASET_PATH, "train", "good")
TEST_DIR = os.path.join(DATASET_PATH, "test")
IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Preprocessing
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

# ✅ Load ResNet50 backbone
model = models.wide_resnet50_2(weights="Wide_ResNet50_2_Weights.DEFAULT")
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
model.eval().to(DEVICE)

def extract_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()
    return embedding

# ✅ Step 1: Build feature embeddings from training set
print("Extracting training embeddings...")
train_embeddings = []
for fname in os.listdir(TRAIN_DIR):
    fpath = os.path.join(TRAIN_DIR, fname)
    if fpath.lower().endswith((".png", ".jpg")):
        emb = extract_embedding(fpath)
        train_embeddings.append(emb)

train_embeddings = np.stack(train_embeddings)

# ✅ Step 2: Compare test images
def calculate_distance(test_emb, train_embs):
    distances = np.linalg.norm(train_embs - test_emb, axis=1)
    return distances.min()

print("\nDetecting anomalies in test images...")
THRESHOLD = 25.0  # 👈 Tune if needed

for defect_type in os.listdir(TEST_DIR):
    defect_path = os.path.join(TEST_DIR, defect_type)
    for fname in os.listdir(defect_path):
        fpath = os.path.join(defect_path, fname)
        if fpath.lower().endswith((".png", ".jpg")):
            test_emb = extract_embedding(fpath)
            score = calculate_distance(test_emb, train_embeddings)
            result = "DEFECTIVE" if score > THRESHOLD else "GOOD"
            print(f"[{result}] Score: {score:.2f} - {defect_type}/{fname}")
