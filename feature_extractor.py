import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import insightface

DATASET_PATH = "data"

print("🔄 Loading ArcFace model...")

app = insightface.app.FaceAnalysis(name="buffalo_l")

app.prepare(ctx_id=0, det_size=(320, 320))

all_paths = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_paths.append(os.path.join(root, file))

print(f"✅ Found {len(all_paths)} images")
print("🚀 Extracting embeddings...\n")

filenames = []
features = []

skipped_no_face = 0
skipped_small = 0
skipped_bad = 0


for path in tqdm(all_paths, desc="Processing", unit="img"):

    img = cv2.imread(path)

    if img is None:
        skipped_bad += 1
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = app.get(img)

    if len(faces) == 0:
        skipped_no_face += 1
        continue

    # choose largest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    # 🔥 relaxed filter (better results)
    w = face.bbox[2] - face.bbox[0]
    h = face.bbox[3] - face.bbox[1]

    if w < 40 or h < 40:
        skipped_small += 1
        continue

    embedding = face.embedding

    features.append(embedding)
    filenames.append(path)

if len(features) == 0:
    print("❌ No embeddings found. Check dataset.")
    exit()

features = np.vstack(features).astype("float32")

features = features / np.linalg.norm(features, axis=1, keepdims=True)

pickle.dump(features, open("embedding.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("\n✅ Done!")
print(f"💾 Saved embeddings : {len(features)}")
print(f"⚠️ No face         : {skipped_no_face}")
print(f"⚠️ Too small       : {skipped_small}")
print(f"⚠️ Bad images      : {skipped_bad}")
