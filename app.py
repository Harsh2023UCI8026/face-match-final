import streamlit as st
import os
if not os.path.exists("data"):
    import download_data
import cv2
import pickle
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
import insightface
import random

st.set_page_config(page_title="Bollywood Celebrity Matcher", layout="centered")

@st.cache_resource
def load_arcface():
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320,320))  # faster
    return app


arc_app = load_arcface()


feature_list = pickle.load(open("embedding.pkl","rb"))
filenames = pickle.load(open("filenames.pkl","rb"))


FUN_LINES = [
    "Arre bhai full hero lag rahe ho!",
    "Bollywood calling you",
    "Direct main lead vibes!",
    "Side role? Impossible bro!",
    "Next superstar spotted"
]

def save_image(img_file):
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", img_file.name)

    with open(path, "wb") as f:
        f.write(img_file.getbuffer())

    return path

def extract_features(path):

    img = cv2.imread(path)

    faces = arc_app.get(img)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda x: x.bbox[2]*x.bbox[3])

    feat = face.embedding

    return feat / np.linalg.norm(feat)

def recommend(feat):

    sims = cosine_similarity([feat], feature_list)[0]

    idx = np.argmax(sims)

    return idx, sims[idx]

def create_result_poster(user_img, celeb_img, actor, conf):

    user_img = user_img.resize((400,400))
    celeb_img = celeb_img.resize((400,400))

    canvas = Image.new("RGB", (850, 520), "#111111")

    canvas.paste(user_img, (25, 80))
    canvas.paste(celeb_img, (425, 80))

    draw = ImageDraw.Draw(canvas)

    draw.text((200, 20), "Bollywood Celebrity Match", fill="white")
    draw.text((250, 460), f"You look like {actor}", fill="#00ff99")
    draw.text((330, 490), f"{conf:.2f}% Match", fill="white")

    path = "result.png"
    canvas.save(path)

    return path

st.title("Bollywood Celebrity Face Matcher")


mode = st.radio("Choose input method:", ["Upload Image", "Webcam Capture"], index=0)

img_path = None

if mode == "Upload Image":

    uploaded = st.file_uploader("Upload photo", type=["jpg","jpeg","png"])

    if uploaded:
        img_path = save_image(uploaded)

else:
    cam = st.camera_input("Take photo")

    if cam:
        img_path = save_image(cam)

if img_path:

    user_img = Image.open(img_path)

    feat = extract_features(img_path)

    if feat is None:
        st.error("No face detected ❌")
        st.stop()

    idx, sim = recommend(feat)

    celeb_path = filenames[idx]
    celeb_img = Image.open(celeb_path)

    actor = os.path.basename(os.path.dirname(celeb_path)).replace("_", " ")

    conf = sim * 100

    col1, col2 = st.columns(2)

    with col1:
        st.image(user_img, caption="You")

    with col2:
        st.image(celeb_img, caption=actor)

    st.success(f"✨ You look like {actor}")
    st.write(f"### 🔥 Similarity: {conf:.2f}%")
    st.info(random.choice(FUN_LINES))


    poster_path = create_result_poster(user_img, celeb_img, actor, conf)

    with open(poster_path, "rb") as f:
        st.download_button("⬇ Download Result Image", f, file_name="celebrity_match.png")
