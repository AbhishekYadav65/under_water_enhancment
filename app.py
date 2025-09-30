# app_underwater.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="AI Underwater Enhancement System", layout="wide")
st.title("🌊 AI Underwater Image Enhancement System")

uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "png", "jpeg"])

def simple_enhancement(image):
    """Apply simple CLAHE + sharpening to mimic AI enhancement"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Sharpening
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

def fake_detection(image):
    """Draw random detection boxes to simulate threat detection"""
    img = np.array(image).copy()
    h, w, _ = img.shape
    for _ in range(random.randint(1,3)):
        x1, y1 = random.randint(0,w//2), random.randint(0,h//2)
        x2, y2 = x1+random.randint(50,150), y1+random.randint(50,150)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(img, "Threat?", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return img

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    enhanced = simple_enhancement(image)
    detected = fake_detection(enhanced)

    col1, col2, col3 = st.columns(3)
    with col1: st.image(image, caption="Original", use_column_width=True)
    with col2: st.image(enhanced, caption="Enhanced", use_column_width=True)
    with col3: st.image(detected, caption="Enhanced + Threat Detection", use_column_width=True)
