import os
import csv
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DATA_DIR = "../dataset"
OUTPUT_CSV = "mp4_extracted.csv"

# Build 127-column header: ['label', 'x0', 'y0', 'z0', ..., 'x41', 'y41', 'z41']
header = ["label"]
for i in range(42):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

rows_written = 0

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label in sorted(os.listdir(DATA_DIR)):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        print(f"Processing sign: {label}...")
        for video_file in os.listdir(label_path):
            if not video_file.lower().endswith(".mp4"):
                continue

            cap = cv2.VideoCapture(os.path.join(label_path, video_file))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    features = []
                    for hand_landmarks in results.multi_hand_landmarks[:2]:
                        for lm in hand_landmarks.landmark:
                            features.extend([lm.x, lm.y, lm.z])

                    # Pad with 0.0 if only 1 hand is visible so total features = 126
                    while len(features) < 126:
                        features.extend([0.0] * 63)

                    writer.writerow([label] + features)
                    rows_written += 1
            cap.release()

print(f"Done! Created {OUTPUT_CSV} with {rows_written} rows and 127 columns.")