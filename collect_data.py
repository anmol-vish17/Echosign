import csv
from pathlib import Path

import cv2
import mediapipe as mp

from features import build_features


DATA_FILE = Path(__file__).resolve().parent / "data" / "samples.csv"
WINDOW_NAME = "Echosign data collection"
BUTTON_BOUNDS = (10, 45, 270, 95)


def on_mouse(event, x, y, flags, state):
    left, top, right, bottom = BUTTON_BOUNDS
    if event == cv2.EVENT_LBUTTONDOWN and left <= x <= right and top <= y <= bottom:
        state["recording"] = not state["recording"]


def count_label_rows(label):
    if not DATA_FILE.is_file():
        return 0
    with DATA_FILE.open("r", newline="", encoding="utf-8") as samples_file:
        return sum(
            1
            for row in csv.reader(samples_file)
            if row and row[0].strip() == label
        )


def main():
    label = input("Sign label to record (for example WATER or IDLE): ").strip().upper()
    if not label:
        raise SystemExit("A sign label is required.")

    hands_module = mp.solutions.hands
    detector = hands_module.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        detector.close()
        raise SystemExit("Could not open the webcam. Check camera permissions and try again.")

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    previous_label_count = count_label_rows(label)
    recording_state = {"recording": False}
    sample_count = 0
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, recording_state)

    try:
        with DATA_FILE.open("a", newline="", encoding="utf-8") as samples_file:
            writer = csv.writer(samples_file)

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = result.multi_hand_landmarks or []

                for hand_landmarks in landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hands_module.HAND_CONNECTIONS,
                    )

                if recording_state["recording"] and (landmarks or label == "IDLE"):
                    hands = [
                        [[point.x, point.y, point.z] for point in hand.landmark]
                        for hand in landmarks
                    ]
                    writer.writerow([label, *build_features(hands)])
                    samples_file.flush()
                    sample_count += 1

                status = "RECORDING" if recording_state["recording"] else "PAUSED"
                button_text = "PAUSE RECORDING" if recording_state["recording"] else "START RECORDING"
                cv2.rectangle(frame, (10, 45), (270, 95), (40, 130, 40), -1)
                cv2.putText(
                    frame,
                    button_text,
                    (25, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                message = f"{label} | {status} | {sample_count} samples | SPACE toggle, Q quit"
                cv2.putText(frame, message, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    recording_state["recording"] = not recording_state["recording"]
                elif key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    current_label_count = count_label_rows(label)
    saved_count = current_label_count - previous_label_count
    if saved_count == sample_count:
        print(
            f"Verified: saved {saved_count} new samples for {label}. "
            f"The CSV now contains {current_label_count} {label} samples at {DATA_FILE}."
        )
    else:
        print(
            f"WARNING: captured {sample_count} samples for {label}, but the CSV "
            f"contains only {saved_count} additional rows. Check the file at {DATA_FILE}."
        )


if __name__ == "__main__":
    main()