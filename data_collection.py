"""
Stage 1 — Data Collection (v6)
Flexible hand detection — accepts 1 OR 2 hands for ANY sign.
Just follow the YouTube video and show the correct ISL sign!
The code records whatever you show.

Controls:
    A-Z        → letter
    0-9        → number
    SHIFT+1    → hello
    SHIFT+2    → thanks
    SHIFT+3    → yes
    SHIFT+4    → no
    SHIFT+5    → help
    SHIFT+6    → please
    SHIFT+7    → sorry
    SHIFT+8    → good
    SHIFT+9    → bad
    SHIFT+0    → more
    ESC        → quit
"""

import cv2
import mediapipe as mp
import csv
import os

ALPHABETS   = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
NUMBERS     = [str(n) for n in range(10)]
WORDS       = ['hello','thanks','yes','no','help','please','sorry','good','bad','more']
ALL_CLASSES = ALPHABETS + NUMBERS + WORDS

SAMPLES_PER_CLASS = 400
DATA_DIR  = 'data'
DATA_FILE = os.path.join(DATA_DIR, 'landmarks_v6.csv')
os.makedirs(DATA_DIR, exist_ok=True)

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

def extract_landmarks(hand_landmarks_list):
    """Always returns 126 values (2 hands x 63).
    If only 1 hand, second hand filled with zeros."""
    all_coords = []
    for i in range(2):
        if i < len(hand_landmarks_list):
            hl    = hand_landmarks_list[i]
            wrist = hl.landmark[0]
            for lm in hl.landmark:
                all_coords.extend([lm.x-wrist.x, lm.y-wrist.y, lm.z-wrist.z])
        else:
            all_coords.extend([0.0] * 63)
    return all_coords

def write_row(label, coords):
    with open(DATA_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([label] + coords)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        h1 = [f'h1_{a}{i}' for i in range(21) for a in ['x','y','z']]
        h2 = [f'h2_{a}{i}' for i in range(21) for a in ['x','y','z']]
        csv.writer(f).writerow(['label'] + h1 + h2)

def count_samples():
    counts = {c: 0 for c in ALL_CLASSES}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts

SHIFT_MAP = {
    ord('!'): 'hello',  ord('@'): 'thanks', ord('#'): 'yes',
    ord('$'): 'no',     ord('%'): 'help',   ord('^'): 'please',
    ord('&'): 'sorry',  ord('*'): 'good',   ord('('): 'bad',
    ord(')'): 'more',
}

cap = cv2.VideoCapture(0)
current_class = None
recording     = False
collected     = 0

print("\n=== ISL Data Collection v6 (Flexible Hand Mode) ===")
print(f"Target: {SAMPLES_PER_CLASS} samples per class")
print("Follow the YouTube video — show 1 or 2 hands as the video shows!")
print("Code accepts any number of hands automatically.")
print("ESC = quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmark_data = None
    hand_count    = 0

    if result.multi_hand_landmarks:
        hand_count = len(result.multi_hand_landmarks)
        for hl in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,180,255), thickness=2)
            )
        landmark_data = extract_landmarks(result.multi_hand_landmarks)

    key = cv2.waitKeyEx(1)

    if key == 27:
        break
    elif key in SHIFT_MAP:
        current_class = SHIFT_MAP[key]
        recording = True; collected = 0
        print(f"Recording WORD: {current_class}")
    elif 65 <= key <= 90:
        current_class = chr(key)
        recording = True; collected = 0
        print(f"Recording LETTER: {current_class}")
    elif 97 <= key <= 122:
        current_class = chr(key).upper()
        recording = True; collected = 0
        print(f"Recording LETTER: {current_class}")
    elif 48 <= key <= 57:
        current_class = chr(key)
        recording = True; collected = 0
        print(f"Recording NUMBER: {current_class}")

    # Record whenever at least 1 hand is visible
    if recording and current_class and landmark_data and hand_count >= 1:
        sample_counts = count_samples()
        if sample_counts[current_class] < SAMPLES_PER_CLASS:
            write_row(current_class, landmark_data)
            collected += 1
        else:
            print(f"✓ {current_class} complete ({SAMPLES_PER_CLASS} samples)")
            recording = False

    # ── HUD ──────────────────────────────────────────────────────────────────
    sample_counts = count_samples()
    done_count    = sum(1 for v in sample_counts.values() if v >= SAMPLES_PER_CLASS)

    # Hand status
    if hand_count == 0:
        hand_msg = "No hand detected — show your hand!"; hand_clr = (0,0,255)
    elif hand_count == 1:
        hand_msg = "1 hand detected ✓";                  hand_clr = (0,255,100)
    else:
        hand_msg = "2 hands detected ✓";                 hand_clr = (0,255,255)

    cv2.putText(frame, f"Class: {current_class or '—'} | {'RECORDING' if recording else 'READY'}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Collected: {collected}/{SAMPLES_PER_CLASS}",
                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    cv2.putText(frame, hand_msg,
                (10,92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_clr, 2)
    cv2.putText(frame, f"Classes done: {done_count}/{len(ALL_CLASSES)}  |  Total: {sum(sample_counts.values())}",
                (10,122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, "Follow YouTube video — 1 or 2 hands, whatever the sign needs!",
                (10,152), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,255,180), 1)
    cv2.putText(frame, "SHIFT+1=hello SHIFT+2=thanks SHIFT+3=yes SHIFT+4=no SHIFT+5=help",
                (10,415), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,255), 1)
    cv2.putText(frame, "SHIFT+6=please SHIFT+7=sorry SHIFT+8=good SHIFT+9=bad SHIFT+0=more",
                (10,437), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,255), 1)
    cv2.putText(frame, "ESC = quit",
                (10,459), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,200,255), 1)

    # Progress bar
    if current_class:
        filled = int((collected / SAMPLES_PER_CLASS) * 400)
        cv2.rectangle(frame, (10,470), (410,485), (40,40,40), -1)
        cv2.rectangle(frame, (10,470), (10+filled,485), (0,255,100), -1)
        cv2.putText(frame, f"{collected}/{SAMPLES_PER_CLASS}",
                    (415,483), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # Overall progress bar
    overall_filled = int((done_count / len(ALL_CLASSES)) * 400)
    cv2.putText(frame, "Overall:", (10,500), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.rectangle(frame, (80,490), (480,503), (40,40,40), -1)
    cv2.rectangle(frame, (80,490), (80+overall_filled,503), (255,165,0), -1)
    cv2.putText(frame, f"{done_count}/{len(ALL_CLASSES)}",
                (485,502), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,165,0), 1)

    cv2.imshow('ISL Data Collection v6', frame)

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n=== Collection Summary ===")
final = count_samples()
done  = 0
for cls, cnt in sorted(final.items()):
    if cnt > 0:
        mark = "✓" if cnt >= SAMPLES_PER_CLASS else f"✗ {cnt}/{SAMPLES_PER_CLASS}"
        print(f"  {cls:10s}: {mark}")
        if cnt >= SAMPLES_PER_CLASS:
            done += 1
print(f"\nCompleted : {done}/{len(ALL_CLASSES)} classes")
print(f"Total     : {sum(final.values())} samples")
print("Run train_model.py next!")
