# Echosign-backend
# EchoSign – Real-Time Bidirectional Indian Sign Language (ISL) Translator

EchoSign is a real-time, bidirectional communication platform designed to bridge the communication gap between Deaf/hard-of-hearing individuals and non-signers using Indian Sign Language (ISL).

## Key Features
* **Sign-to-Text & Speech (ISL Recognition):** Real-time webcam gesture recognition using MediaPipe Hands (126 3D landmarks) and a Random Forest Classifier (`94.20%` accuracy across 23 ISL classes).
* **Speech-to-Sign Translation:** Converts spoken or typed English sentences into ISL motion videos and A–Z fingerspelling sequences.
* **Smart Sentence Builder & Quick Phrases:** Builds complete sentences from recognized signs with confidence filtering and one-tap emergency/daily communication phrases.
* **Accessibility First:** High-contrast UI, Web Speech text-to-speech audio output, and adjustable playback speed.

## Research & Methodology
* **Problem Statement:** Conventional sign language translators are often unidirectional, hardware-dependent (sensor gloves), or trained on single-user datasets that fail in real-world multi-user environments.
* **Approach:** EchoSign uses a vision-based, hardware-free pipeline:
  1. **Landmark Extraction:** MediaPipe Hands extracts 21 3D coordinates (x, y, z) per hand (126 features total for two hands), making predictions invariant to background clutter and skin tone.
  2. **Multi-Contributor Dataset Balancing:** Gesture samples were collected across 4 contributors under varying lighting and camera angles, normalized, and class-balanced (`11,146` total samples across `23` classes).
  3. **Classification:** A class-balanced Random Forest Classifier (`n_estimators=400`) was selected for low-latency real-time inference (`<30 ms` per frame) without requiring a dedicated GPU.

## Supported ISL Vocabulary (23 Classes)
`HELLO`, `HI`, `GOOD`, `BAD`, `GOOD MORNING`, `THANK YOU`, `PLEASE`, `SORRY`, `YES`, `NO`, `I`, `WANT`, `NEED`, `HELP`, `WATER`, `FOOD`, `MEDICINE`, `TOILET`, `HURT`, `CASH`, `TIME`, `STOP`, and `IDLE` (no-gesture baseline), plus full **A–Z Fingerspelling**.

## Tech Stack
* **Machine Learning & Computer Vision:** Python 3.11, MediaPipe Hands, Scikit-Learn (Random Forest, `n_estimators=400`), OpenCV, Pandas, NumPy
* **Backend API:** Flask, Flask-CORS
* **Frontend:** HTML5, CSS3, JavaScript (Web Speech API)

## Project Structure
```text
EchoSign/
├── backend/
│   ├── app.py                  # Flask REST API (/health, /predict, /speech-to-sign)
│   ├── train.py                # Model training & confusion matrix generator
│   ├── model.pkl               # Trained Random Forest model (94.20% accuracy)
│   ├── confusion_matrix.png    # Evaluation confusion matrix
│   ├── data/
│   │   └── final_dataset.csv   # Balanced multi-contributor dataset (23 classes)
│   └── static/
│       └── signs/
│           ├── words/          # Compressed ISL word motion videos (.mp4)
│           └── az/             # A–Z fingerspelling images (.jpg) & motion letters (.mp4)
├── frontend/                   # Web UI & camera/speech integration
├── docs/                       # Project Report & Presentation (PPT)
└── README.md
```

## How to Run Locally

### 1. Start the Backend Server
```bash
cd backend
python -m venv venv311
.\venv311\Scripts\activate
pip install -r requirements.txt
python app.py
```
The Flask server will start at `http://127.0.0.1:5000`.

### 2. Run the Frontend
Open `frontend/index.html` using Live Server (or any local HTTP server) in Google Chrome / Microsoft Edge.

## Model Performance
* **Algorithm:** Random Forest Classifier (`n_estimators=400`, `class_weight="balanced"`)
* **Dataset:** Multi-person dataset collected across 4 contributors (`23` ISL classes including `IDLE`)
* **Overall Accuracy:** **94.20%** (see `backend/confusion_matrix.png` for per-class precision, recall, and F1-scores)

## Documentation & Presentation
* **Project Presentation (PPT):** Available in `docs/EchoSign_Presentation.pptx` *(covers architecture, dataset pipeline, live demo flow, and accuracy evaluation)*
* **Project Report:** Available in `docs/EchoSign_Project_Report.pdf`

## Team Members
* **Anmol:** Backend Development, MediaPipe Feature Extraction & Machine Learning Model
* **Anushka:** Frontend Development, UI/UX, Accessibility & System Integration
* **Imama:** Dataset Curation, Multi-User Testing & Documentation
* **Areen:** ISL Video Dataset Collection, Sign Visuals & Research Support
