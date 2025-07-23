import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2
from PIL import Image
import numpy as np

# Title
st.title("Bottle Defect Detection (MVTec AD)")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a bottle...", type=["jpg", "jpeg", "png"])

# Define transformation manually (ImageNet mean and std)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

# Load backbone model
@st.cache_resource
def load_model():
    model = wide_resnet50_2(pretrained=True)
    model.eval()
    return model

model = load_model()

# Simple similarity threshold
THRESHOLD = 0.9

def calculate_similarity(feat1, feat2):
    return torch.cosine_similarity(feat1, feat2).item()

# Load a single reference image (good bottle image)
@st.cache_resource
def get_reference_embedding():
    ref_img = Image.open("./visual-anomaly-detector/normal_samples/bottle/good/000.png").convert("RGB")
    ref_tensor = transform(ref_img).unsqueeze(0)
    with torch.no_grad():
        ref_features = model(ref_tensor)
    return ref_features

ref_embedding = get_reference_embedding()

# When an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Feature extraction
    with torch.no_grad():
        features = model(input_tensor)

    # Similarity check
    similarity = calculate_similarity(ref_embedding, features)
    st.write(f"Similarity to good reference: **{similarity:.4f}**")

    # Classification
    if similarity > THRESHOLD:
        st.success("✅ The product is GOOD.")
    else:
        st.error("❌ The product is DEFECTIVE.")
