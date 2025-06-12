from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pymongo import MongoClient
import base64
import io
from PIL import Image
import datetime
import os
from scipy.spatial.distance import cosine
from flask_cors import CORS

# --- Flask App ---
app = Flask(__name__)
CORS(app)

# --- MongoDB ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://raflipratama2826:IBwcA4VGjffeAR80@mediface.zjspsth.mongodb.net/?retryWrites=true&w=majority&appName=mediface")
client = MongoClient(MONGO_URI)
db = client["hospital_db"]
collection_face_embeddings = db["face_embeddings"]

# --- Build Model ---
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu', name='face_embedding'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
    ])
    return model

arcface = build_model()

# --- Haar Cascade ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Utility Functions ---
def load_image_from_base64(base64_str):
    try:
        header, encoded = base64_str.split(",", 1) if "," in base64_str else ("", base64_str)
        img_data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return img
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

def detect_and_crop_face(image_pil):
    image_np = np.array(image_pil)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face_img = image_pil.crop((x, y, x + w, y + h)).resize((112, 112))
    return face_img

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return "Face Embedding API is running."

@app.route("/generate-embedding", methods=["POST"])
def generate_embedding():
    data = request.json
    patient_id = data.get("patient_id")
    name = data.get("name")
    photo_list = data.get("photos", [])

    if not photo_list:
        return jsonify({"error": "No photos provided"}), 400

    embeddings = []

    for photo_data in photo_list:
        img = load_image_from_base64(photo_data)
        if img is None:
            continue

        face = detect_and_crop_face(img)
        if face is None:
            continue

        face_array = np.array(face) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        embedding = arcface.predict(face_array)[0]
        embeddings.append(embedding.tolist())

    if not embeddings:
        return jsonify({"error": "No valid embeddings could be created"}), 400

    mean_embedding = np.mean(embeddings, axis=0).tolist()

    result = collection_face_embeddings.update_one(
        {"_id": patient_id},
        {"$set": {
            "name": name,
            "embeddings": mean_embedding,
            "updated_at": datetime.datetime.utcnow()
        }},
        upsert=True
    )

    return jsonify({
        "status": "success",
        "matched_count": result.matched_count,
        "modified_count": result.modified_count
    })

@app.route("/get-embedding-by-name/<name>", methods=["GET"])
def get_embedding(name):
    data = collection_face_embeddings.find_one({"name": name}, {"_id": 0})
    
    if not data:
        return jsonify({"error": f"Embedding not found for name: {name}"}), 404

    return jsonify({
        "status": "success",
        "name": data.get("name"),
        "embeddings": data.get("embeddings")
    })

@app.route("/recognize", methods=["POST"])
def recognize_face():
    data = request.json
    photo_data = data.get("photo")

    if not photo_data:
        return jsonify({"error": "No photo provided"}), 400

    img = load_image_from_base64(photo_data)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    face = detect_and_crop_face(img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400

    face_array = np.array(face) / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    input_embedding = arcface.predict(face_array)[0]

    all_faces = list(collection_face_embeddings.find({}, {"_id": 1, "name": 1, "embeddings": 1}))

    best_match = None
    min_distance = float('inf')
    threshold = 0.7

    for face_data in all_faces:
        db_embedding = np.array(face_data["embeddings"])
        distance = cosine(input_embedding, db_embedding)
        if distance < min_distance:
            min_distance = distance
            best_match = face_data

    if best_match and min_distance < threshold:
        return jsonify({
            "status": "matched",
            "name": best_match["name"],
            "patient_id": str(best_match["_id"]),
            "distance": min_distance
        })
    else:
        return jsonify({
            "status": "unmatched",
            "message": "No matching face found.",
            "distance": min_distance
        })
    
# --- Run App (for local testing) ---
if __name__ == "__main__":
    app.run(debug=True)

