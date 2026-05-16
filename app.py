from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import traceback
import os
from google import genai
import requests
import time

app = Flask(__name__)

last_sensor = {
    "soil": 0,
    "temp": 0,
    "humidity": 0,
    "timestamp": 0,   # 🔥 NEW
    "source": "none"  # iot / weather
}

# =========================================
# CONFIGURE GEMINI
# =========================================
# =========================================
# CONFIGURE GEMINI
# =========================================

API_KEY = "AIzaSyBEoPvzVku8DkCn-xH7qKGzkoGmpXst3xQ"

client = genai.Client(api_key=API_KEY)

print("✅ Gemini API Connected")

# =========================================
# CHAT MEMORY (NEW)
# =========================================
chat_history = []

# =========================================
# LOAD MODEL
# =========================================
try:
    model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:")
    traceback.print_exc()
    model = None

# =========================================
# LOAD CLASS LABELS
# =========================================
try:
    with open("class_names.json") as f:
        class_indices = json.load(f)

    class_names = {int(v): k for k, v in class_indices.items()}
    print("✅ Class labels loaded")
except Exception as e:
    print("❌ Error loading class names:")
    traceback.print_exc()
    class_names = {}

# =========================================
# IMAGE PREPROCESSING
# =========================================
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image, dtype=np.float32) / 255.0

    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)

    image = np.expand_dims(image, axis=0)
    return image

# =========================================
# GEMINI CHATBOT (UPGRADED)
# =========================================
def ask_gemini(user_message, disease=""):
    try:
        global chat_history

        # Save user message
        chat_history.append(f"User: {user_message}")
        

        prompt = f"""
You are an expert agriculture assistant.

Detected Disease: {disease}

Conversation:
{chr(10).join(chat_history)}

Reply in simple language for farmers.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        reply = response.text if response.text else "No response"

        # Save AI reply
        chat_history.append(f"AI: {reply}")

        return reply

    except Exception as e:
        print("❌ Gemini Error:", e)
        return "AI response failed."
    

def get_best_data(lat, lon):
    global last_sensor

    now = time.time()

    # ⏱ if IoT data is less than 30 sec old → use it
    if last_sensor["timestamp"] and (now - last_sensor["timestamp"] < 30):
        print("✅ Using IoT data")
        return last_sensor["soil"], last_sensor["temp"], last_sensor["humidity"], "iot"

    # 🌍 else fallback to weather
    if lat and lon:
        print("🌍 Using Weather fallback")
        soil, temp, humidity = get_weather(lat, lon)

        # update sensor so UI shows weather too
        last_sensor.update({
            "soil": soil,
            "temp": temp,
            "humidity": humidity,
            "timestamp": now,
            "source": "weather"
        })

        return soil, temp, humidity, "weather"

    # ⚠ fallback if nothing
    print("⚠ Using default values")
    return 50, 25, 50, "default"

# =========================================
# ROUTES
# =========================================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        image = Image.open(file).convert('RGB')
        processed = preprocess_image(image)

        prediction = model.predict(processed)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        disease_name = class_names.get(class_index, "Unknown")
        disease_name = disease_name.replace("___", " ").replace("_", " ")

        return jsonify({
            "disease": disease_name,
            "confidence": f"{round(confidence * 100, 2)}%"
        })

    except Exception as e:
        print("❌ Prediction error:")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500

# ✅ UPDATED CHAT ROUTE (NOW TAKES USER INPUT)
@app.route('/chat', methods=['POST'])
def chat():
    try:
        global chat_history

        data = request.get_json()
        disease = data.get("disease", "")
        user_message = data.get("message", "")

        lat = data.get("lat")
        lon = data.get("lon")

        # ✅ SMART DATA SOURCE SWITCH
        try:
            soil, temp, humidity, source = get_best_data(lat, lon)
        except:
            soil, temp, humidity, source = 0, 25, 50, "fallback"

        # ✅ Save user message
        chat_history.append(f"User: {user_message}")

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        # ✅ STRICT JSON PROMPT
        prompt = f"""
You are a smart agriculture assistant.

Return ONLY valid JSON. No extra text.

Data Source: {source}
Disease: {disease}
Soil: {soil}
Temperature: {temp}
Humidity: {humidity}

Conversation:
{chr(10).join(chat_history)}

User Question: {user_message}

Return format:

{{
  "english": {{
    "problem": "",
    "solution": "",
    "watering": "",
    "prevention": ""
  }},
  "kannada": {{
    "problem": "",
    "solution": "",
    "watering": "",
    "prevention": ""
  }}
}}
"""

        # ✅ HANDLE QUOTA ERROR (VERY IMPORTANT)
        try:
            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt
            )
        except Exception as e:
            if "429" in str(e):
                import time
                time.sleep(20)
                response = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=prompt
                )
            else:
                raise e

        text = response.text.strip()

        # ✅ SAFE JSON PARSE
        try:
            structured = json.loads(text)
        except:
            print("⚠️ JSON parse failed:", text)

            # fallback response
            structured = {
                "english": {
                    "problem": "Unable to analyze properly",
                    "solution": "Please try again",
                    "watering": "Check soil moisture",
                    "prevention": "Maintain good plant care"
                },
                "kannada": {
                    "problem": "ಸರಿಯಾಗಿ ವಿಶ್ಲೇಷಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ",
                    "solution": "ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ",
                    "watering": "ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ",
                    "prevention": "ಸಸ್ಯದ ಸರಿಯಾದ ಆರೈಕೆ ಮಾಡಿ"
                }
            }

        # ✅ Save AI response (IMPORTANT)
        chat_history.append(f"AI: {json.dumps(structured)}")

        return jsonify(structured)

    except Exception as e:
        print("❌ Chat error:", e)
        return jsonify({"error": str(e)}), 500

# =========================================
# IOT SENSOR DATA API
# =========================================
@app.route('/iot-data', methods=['POST'])
def iot_data():
    try:
        data = request.get_json()

        global last_sensor

        last_sensor = {
            "soil": data.get("soil_moisture"),
            "temp": data.get("temperature"),
            "humidity": data.get("humidity"),
            "timestamp": time.time(),   # ✅ NEW
            "source": "iot"             # ✅ NEW
        }

        print("📡 IoT Updated:", last_sensor)

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"error": "IoT failed"}), 500

@app.route('/last-iot')
def last_iot():
    return jsonify(last_sensor)

def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m"

        res = requests.get(url)
        data = res.json()

        temp = data["current_weather"]["temperature"]
        humidity = data["hourly"]["relativehumidity_2m"][0]

        # Fake soil estimation (basic logic)
        soil = 50  # default medium

        return soil, temp, humidity

    except Exception as e:
        print("Weather error:", e)
        return 50, 25, 50  # fallback safe values

# =========================================
# RUN
# =========================================
if __name__ == '__main__':
    app.run(debug=True)