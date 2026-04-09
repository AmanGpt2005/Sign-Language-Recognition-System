from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

app = Flask(__name__)

# ── Load model files ───────────────────────────────────────────
model = tf.keras.models.load_model("isl_model.h5")
classes = np.load("label_encoder.npy", allow_pickle=True)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Download MediaPipe task file ───────────────────────────────
if not os.path.exists("hand_landmarker.task"):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "hand_landmarker.task"
    )

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)

# ── Gujarati word mapping ──────────────────────────────────────
gujarati_map = {
    'A': 'આંબો (Mango)',
    'B': 'બળદ (Ox)',
    'C': 'ચંદ્ર (Moon)',
    'D': 'દરવાજો (Door)',
    'E': 'એકતા (Unity)',
    'F': 'ફૂલ (Flower)',
    'G': 'ગાય (Cow)',
    'H': 'હાથ (Hand)',
    'I': 'ઇમારત (Building)',
    'J': 'જળ (Water)',
    'K': 'કમળ (Lotus)',
    'L': 'લીંબુ (Lemon)',
    'M': 'માતા (Mother)',
    'N': 'નદી (River)',
    'O': 'ઓરડો (Room)',
    'P': 'પાણી (Water)',
    'Q': 'કતાર (Queue)',
    'R': 'રસ્તો (Road)',
    'S': 'સૂર્ય (Sun)',
    'T': 'તારો (Star)',
    'U': 'ઉજાસ (Light)',
    'V': 'વૃક્ષ (Tree)',
    'W': 'વાયુ (Wind)',
    'X': 'અજ્ઞાત (Unknown)',
    'Y': 'યુવાન (Youth)',
    'Z': 'ઝાડ (Tree)',
}

label_map = {i: chr(65 + i) for i in range(26)}

# ── Preprocessing — must match training exactly ────────────────
def extract_raw(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def apply_relative_single(coords):
    coords = np.array(coords)
    if coords[0] != -1.0:
        wx, wy, wz = coords[0], coords[1], coords[2]
        for i in range(0, len(coords), 3):
            coords[i]   -= wx
            coords[i+1] -= wy
            coords[i+2] -= wz
    return coords.tolist()

# ── Prediction ─────────────────────────────────────────────────
def predict_from_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return None, None, "Could not decode image"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    left_landmarks  = [-1.0] * 63  # -1.0 matches training missing value
    right_landmarks = [-1.0] * 63
    uses_two_hands  = 0

    if not result.hand_landmarks:
        return None, None, "No hand detected in image"

    uses_two_hands = 1 if len(result.hand_landmarks) == 2 else 0

    for i, hand_lms in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        if handedness == 'Left':
            left_landmarks = apply_relative_single(extract_raw(hand_lms))
        else:
            right_landmarks = apply_relative_single(extract_raw(hand_lms))

    features = np.array([uses_two_hands] + left_landmarks + right_landmarks).reshape(1, -1)
    features = scaler.transform(features)
    pred = model.predict(features, verbose=0)

    letter = label_map[int(classes[np.argmax(pred)])]
    confidence = float(np.max(pred))
    gujarati_word = gujarati_map.get(letter, '')
    return letter, confidence, gujarati_word

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_bytes = request.files['image'].read()
    letter, confidence, gujarati_word = predict_from_image(image_bytes)
    if letter is None:
        return jsonify({'error': gujarati_word}), 400
    return jsonify({
        'letter': letter,
        'confidence': round(confidence * 100, 1),
        'gujarati_word': gujarati_word
    })

@app.route('/learn')
def learn():
    letters = []
    gujarati_letters = {
        'A':'અ','B':'બ','C':'ક','D':'ડ','E':'એ','F':'ફ','G':'ગ',
        'H':'હ','I':'ઇ','J':'જ','K':'ક','L':'લ','M':'મ','N':'ન',
        'O':'ઓ','P':'પ','Q':'ક','R':'ર','S':'સ','T':'ત','U':'ઉ',
        'V':'વ','W':'વ','X':'ક્સ','Y':'વાય','Z':'ઝ'
    }
    for i in range(26):
        letter = chr(65 + i)
        letters.append({
            'english': letter,
            'gujarati': gujarati_letters[letter],
            'gujarati_word': gujarati_map[letter],
            'image': f'signs/{letter}.jpg'
        })
    return render_template('learn.html', letters=letters)

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/check', methods=['POST'])
def check():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    expected = request.form.get('expected', '')
    image_bytes = request.files['image'].read()
    letter, confidence, _ = predict_from_image(image_bytes)
    if letter is None:
        return jsonify({'correct': False, 'predicted': None, 'confidence': 0, 'message': 'No hand detected'})
    correct = letter == expected
    return jsonify({
        'correct': correct,
        'predicted': letter,
        'confidence': round(confidence * 100, 1),
        'message': 'Correct!' if correct else f'Try again — you signed {letter}'
    })

@app.route('/words')
def words():
    gujarati_words = {
        'A': [('આંબો', 'Mango', '🥭'), ('આકાશ', 'Sky', '🌤️'), ('આંખ', 'Eye', '👁️')],
        'B': [('બળદ', 'Ox', '🐂'), ('બારી', 'Window', '🪟'), ('બગીચો', 'Garden', '🌳')],
        'C': [('ચંદ્ર', 'Moon', '🌙'), ('ચોપડી', 'Book', '📚'), ('ચા', 'Tea', '🍵')],
        'D': [('દરવાજો', 'Door', '🚪'), ('દૂધ', 'Milk', '🥛'), ('દીવો', 'Lamp', '🪔')],
        'E': [('એકતા', 'Unity', '🤝'), ('એરણ', 'Anvil', '⚒️'), ('એપ્રિલ', 'April', '📅')],
        'F': [('ફૂલ', 'Flower', '🌸'), ('ફળ', 'Fruit', '🍎'), ('ફાનસ', 'Lantern', '🏮')],
        'G': [('ગાય', 'Cow', '🐄'), ('ગુલાબ', 'Rose', '🌹'), ('ગામ', 'Village', '🏘️')],
        'H': [('હાથ', 'Hand', '✋'), ('હૃદય', 'Heart', '❤️'), ('હાથી', 'Elephant', '🐘')],
        'I': [('ઇમારત', 'Building', '🏢'), ('ઇનામ', 'Prize', '🏆'), ('ઇંડું', 'Egg', '🥚')],
        'J': [('જળ', 'Water', '💧'), ('જંગલ', 'Forest', '🌲'), ('જહાજ', 'Ship', '🚢')],
        'K': [('કમળ', 'Lotus', '🪷'), ('કૂતરો', 'Dog', '🐕'), ('કેળું', 'Banana', '🍌')],
        'L': [('લીંબુ', 'Lemon', '🍋'), ('લાડુ', 'Ladoo', '🍬'), ('લોટ', 'Flour', '🌾')],
        'M': [('માતા', 'Mother', '👩'), ('માછલી', 'Fish', '🐟'), ('મકાન', 'House', '🏠')],
        'N': [('નદી', 'River', '🌊'), ('નળ', 'Tap', '🚰'), ('નાક', 'Nose', '👃')],
        'O': [('ઓરડો', 'Room', '🏠'), ('ઓશીકું', 'Pillow', '🛏️'), ('ઓટો', 'Auto', '🛺')],
        'P': [('પાણી', 'Water', '💧'), ('પક્ષી', 'Bird', '🐦'), ('પુસ્તક', 'Book', '📖')],
        'Q': [('કતાર', 'Queue', '👥'), ('ક્વિઝ', 'Quiz', '❓'), ('કિલ્લો', 'Fort', '🏰')],
        'R': [('રસ્તો', 'Road', '🛣️'), ('રાત', 'Night', '🌙'), ('રમત', 'Game', '🎮')],
        'S': [('સૂર્ય', 'Sun', '☀️'), ('સફરજન', 'Apple', '🍎'), ('સાપ', 'Snake', '🐍')],
        'T': [('તારો', 'Star', '⭐'), ('તરબૂચ', 'Watermelon', '🍉'), ('તળાવ', 'Pond', '🏞️')],
        'U': [('ઉજાસ', 'Light', '💡'), ('ઉંટ', 'Camel', '🐪'), ('ઉદ્યાન', 'Park', '🌳')],
        'V': [('વૃક્ષ', 'Tree', '🌳'), ('વાદળ', 'Cloud', '☁️'), ('વાઘ', 'Tiger', '🐯')],
        'W': [('વાયુ', 'Wind', '💨'), ('વ્હેલ', 'Whale', '🐋'), ('વિશ્વ', 'World', '🌍')],
        'X': [('અજ્ઞાત', 'Unknown', '❓'), ('ક્સાઇલોફોન', 'Xylophone', '🎵'), ('એક્સ-રે', 'X-Ray', '🩻')],
        'Y': [('યુવાન', 'Youth', '👦'), ('યોગ', 'Yoga', '🧘'), ('યાત્રા', 'Journey', '✈️')],
        'Z': [('ઝાડ', 'Tree', '🌳'), ('ઝરણું', 'Waterfall', '💦'), ('ઝેબ્રા', 'Zebra', '🦓')],
    }
    return render_template('words.html', gujarati_words=gujarati_words)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/save_progress', methods=['POST'])
def save_progress():
    import json
    data = request.get_json()
    # Save to a simple JSON file
    with open('progress.json', 'w') as f:
        json.dump(data, f)
    return jsonify({'status': 'saved'})

@app.route('/get_progress')
def get_progress():
    import json, os
    if os.path.exists('progress.json'):
        with open('progress.json') as f:
            return jsonify(json.load(f))
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True)