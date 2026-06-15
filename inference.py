import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import collections

print('Loading model...')
model = tf.keras.models.load_model('model/isl_model_best.keras')
classes = np.load('model/label_encoder.npy', allow_pickle=True)
print(f'Classes: {list(classes)}')

CONFIDENCE_THRESHOLD = 0.75
PREDICTION_BUFFER = 15

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)

prediction_buffer = collections.deque(maxlen=PREDICTION_BUFFER)
current_sign = ''
current_word = ''
sentence = []
last_added_time = time.time()
ADD_DELAY = 1.5

def extract_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return coords

def get_stable_prediction(buffer):
    if not buffer:
        return None, 0.0
    counter = collections.Counter(buffer)
    sign, count = counter.most_common(1)[0]
    return sign, count / len(buffer)

def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print('Starting. Press q to quit.')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    predicted_sign = None
    confidence = 0.0

    if result.multi_hand_landmarks:
        all_lm = result.multi_hand_landmarks
        hand1 = extract_landmarks(all_lm[0]) if len(all_lm) > 0 else [0.0]*63
        hand2 = extract_landmarks(all_lm[1]) if len(all_lm) > 1 else [0.0]*63
        for hl in all_lm:
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,180,255), thickness=2))
        inp = np.array(hand1 + hand2, dtype=np.float32).reshape(1, -1)
        pred = model.predict(inp, verbose=0)[0]
        idx = np.argmax(pred)
        conf = pred[idx]
        prediction_buffer.append(classes[idx] if conf >= CONFIDENCE_THRESHOLD else None)
        predicted_sign, confidence = get_stable_prediction(prediction_buffer)
        current_sign = predicted_sign or ''
        if current_sign and time.time() - last_added_time > ADD_DELAY and len(current_sign) == 1:
            current_word += current_sign
            last_added_time = time.time()
    else:
        prediction_buffer.clear()
        current_sign = ''

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        current_word = ''
    elif key == 13:
        if current_word:
            sentence.append(current_word)
            current_word = ''
    elif key == 8:
        current_word = current_word[:-1]
    elif key == ord('w') and current_sign and len(current_sign) > 1:
        sentence.append(current_sign)
        last_added_time = time.time()

    draw_rounded_rect(frame, 0, 0, w, 110, 0, (30,30,30))
    cv2.putText(frame, current_sign if current_sign else '-', (20,75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,100), 4)
    cv2.putText(frame, f'{confidence*100:.0f}%' if current_sign else '', (160,75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200,200,200), 2)
    cv2.putText(frame, 'Predicted sign', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    draw_rounded_rect(frame, 0, h-100, w, 100, 0, (20,20,20))
    cv2.putText(frame, f'Word: {current_word}_' if current_word else 'Word: (start signing)', (15,h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,100), 2)
    cv2.putText(frame, 'Sentence: ' + ' '.join(sentence) if sentence else 'Sentence: -', (15,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 1)
    cv2.putText(frame, 'SPACE=clear  ENTER=confirm  BACKSPACE=delete  w=add word  q=quit', (10,h-110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)
    cv2.imshow('ISL Sign Language Recognition', frame)

cap.release()
cv2.destroyAllWindows()
hands.close()
print('Done. Sentence:', ' '.join(sentence))