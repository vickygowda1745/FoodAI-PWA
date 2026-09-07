import os
import io
import json
import re

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from google import genai
from google.genai import types


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best_food_model_v2.keras"
CLASS_FILE = "class_names_v2.txt"

IMAGE_SIZE = (224, 224)

GEMINI_MODEL = "gemini-3.6-flash"

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    print("Gemini AI: CONNECTED")

else:
    gemini_client = None

    print("WARNING: GEMINI_API_KEY not found.")
    print("Gemini recognition will not work.")


# ============================================================
# LOAD YOUR CUSTOM FOOD MODEL
# ============================================================

print("Loading custom Food AI V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    CLASS_FILE,
    "r",
    encoding="utf-8"
) as f:

    class_names = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    "Custom food classes loaded:",
    len(class_names)
)


# ============================================================
# RECOMMENDATION DATABASE
# ============================================================

def get_local_recommendation(food_name):

    food = food_name.lower()

    if "biryani" in food:

        return {
            "recommendation":
                "Best served with raita, salad or boiled egg.",

            "health_note":
                "Biryani can be calorie-dense, so moderate portions are recommended.",

            "similar_food":
                "Pulao"
        }

    if "poha" in food:

        return {
            "recommendation":
                "Pair with curd, sprouts or fruit for a balanced breakfast.",

            "health_note":
                "Adding vegetables and protein can improve the nutritional balance.",

            "similar_food":
                "Upma"
        }

    if "dosa" in food:

        return {
            "recommendation":
                "Serve with sambar and coconut chutney.",

            "health_note":
                "Sambar can add protein and vegetables to the meal.",

            "similar_food":
                "Idli"
        }

    if "idli" in food:

        return {
            "recommendation":
                "Serve with sambar and chutney.",

            "health_note":
                "Idli is steamed and generally lighter than fried breakfast foods.",

            "similar_food":
                "Dosa"
        }

    if "burger" in food:

        return {
            "recommendation":
                "Pair with salad, vegetables or a low-sugar drink.",

            "health_note":
                "Sauces, cheese and fried sides can significantly increase calories.",

            "similar_food":
                "Sandwich"
        }

    if "pizza" in food:

        return {
            "recommendation":
                "Pair with salad or vegetables.",

            "health_note":
                "Calories and sodium depend strongly on portion size and toppings.",

            "similar_food":
                "Pasta"
        }

    if "samosa" in food:

        return {
            "recommendation":
                "Pair with mint chutney and a lighter main meal.",

            "health_note":
                "Samosa is fried, so moderate portions are recommended.",

            "similar_food":
                "Kachori"
        }

    return {
        "recommendation":
            f"{food_name.title()} can be enjoyed as part of a balanced meal.",

        "health_note":
            "Nutrition depends on ingredients, preparation method and portion size.",

        "similar_food":
            "No suggestion available"
    }


# ============================================================
# GET IMAGE FROM REQUEST
# ============================================================

def get_image_bytes():

    if "image" in request.files:

        return request.files[
            "image"
        ].read()

    if "file" in request.files:

        return request.files[
            "file"
        ].read()

    return request.get_data()


# ============================================================
# GEMINI OPEN-ENDED FOOD RECOGNITION
# ============================================================

def recognize_with_gemini(image_bytes):

    if gemini_client is None:

        raise RuntimeError(
            "Gemini API is not configured."
        )

    prompt = """
You are the recognition engine for an
AI Food Recognition & Recommendation Agent.

Carefully inspect this image.

Your task:

1. Determine whether food or a drink is visible.
2. Identify the most specific food/dish name you can reasonably recognize.
3. Do NOT restrict yourself to a predefined list.
4. Food may be Indian or from any cuisine in the world.
5. If multiple foods are visible, identify the main dish and mention important visible side dishes.
6. Do not invent certainty when the image is unclear.
7. Give practical meal recommendations, not medical advice.

Return ONLY valid JSON in exactly this format:

{
  "is_food": true,
  "food": "food name",
  "confidence": 0,
  "cuisine": "cuisine or Unknown",
  "description": "short description",
  "visible_items": ["item 1", "item 2"],
  "recommendation": "short serving or pairing suggestion",
  "health_note": "short general nutrition note",
  "similar_food": "similar dish"
}

confidence must be an integer from 0 to 100.

If this is clearly not food, return:

{
  "is_food": false,
  "food": "Not food",
  "confidence": 0,
  "cuisine": "N/A",
  "description": "No food confidently detected.",
  "visible_items": [],
  "recommendation": "",
  "health_note": "",
  "similar_food": ""
}
"""

    response = gemini_client.models.generate_content(

        model=GEMINI_MODEL,

        contents=[
            prompt,

            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]
    )

    text = response.text.strip()

    # Remove Markdown JSON fences if Gemini adds them
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return json.loads(text)


# ============================================================
# LOCAL V2 MODEL FALLBACK
# ============================================================

def recognize_with_local_model(
    image_bytes
):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    best_index = int(
        top_indices[0]
    )

    food_name = class_names[
        best_index
    ]

    confidence = float(
        predictions[
            best_index
        ]
    ) * 100

    top3 = []

    for index in top_indices:

        top3.append({

            "food":
                class_names[
                    int(index)
                ],

            "confidence":
                round(
                    float(
                        predictions[
                            index
                        ]
                    ) * 100,
                    2
                )
        })

    recommendation = (
        get_local_recommendation(
            food_name
        )
    )

    return {

        "success": True,

        "recognizer":
            "Food AI V2 fallback",

        "food":
            food_name,

        "confidence":
            round(
                confidence,
                2
            ),

        "cuisine":
            "Unknown",

        "description":
            "Recognized using the custom 239-class Food AI V2 model.",

        "visible_items":
            [food_name],

        "top_predictions":
            top3,

        "recommendation":
            recommendation[
                "recommendation"
            ],

        "health_note":
            recommendation[
                "health_note"
            ],

        "similar_food":
            recommendation[
                "similar_food"
            ]
    }


# ============================================================
# MAIN FOOD PREDICTION
# ============================================================

def predict_food():

    try:

        image_bytes = get_image_bytes()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400


        # Validate image before sending
        try:

            test_image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

            test_image.verify()

        except Exception:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400


        # ====================================================
        # PRIMARY: GEMINI GENERAL FOOD RECOGNITION
        # ====================================================

        try:

            result = recognize_with_gemini(
                image_bytes
            )

            if not result.get(
                "is_food",
                True
            ):

                return jsonify({

                    "success":
                        False,

                    "recognizer":
                        "Gemini Vision",

                    "food":
                        "Not food",

                    "confidence":
                        result.get(
                            "confidence",
                            0
                        ),

                    "message":
                        "No food confidently detected."
                })


            return jsonify({

                "success":
                    True,

                "recognizer":
                    "Gemini Vision",

                "food":
                    result.get(
                        "food",
                        "Unknown food"
                    ),

                "confidence":
                    result.get(
                        "confidence",
                        0
                    ),

                "cuisine":
                    result.get(
                        "cuisine",
                        "Unknown"
                    ),

                "description":
                    result.get(
                        "description",
                        ""
                    ),

                "visible_items":
                    result.get(
                        "visible_items",
                        []
                    ),

                "recommendation":
                    result.get(
                        "recommendation",
                        ""
                    ),

                "health_note":
                    result.get(
                        "health_note",
                        ""
                    ),

                "similar_food":
                    result.get(
                        "similar_food",
                        ""
                    )
            })


        # ====================================================
        # FALLBACK TO YOUR TRAINED MODEL
        # ====================================================

        except Exception as gemini_error:

            print(
                "Gemini error:",
                gemini_error
            )

            print(
                "Using local Food AI V2 fallback..."
            )

            return jsonify(
                recognize_with_local_model(
                    image_bytes
                )
            )


    except UnidentifiedImageError:

        return jsonify({
            "success": False,
            "error": "Invalid image"
        }), 400


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# PWA FILES
# ============================================================

@app.route("/manifest.json")
def manifest():

    return send_from_directory(
        ".",
        "manifest.json"
    )


@app.route("/service-worker.js")
def service_worker():

    response = send_from_directory(
        ".",
        "service-worker.js"
    )

    response.headers[
        "Cache-Control"
    ] = "no-cache"

    return response


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "online",

        "project":
            "AI Food Recognition & Recommendation Agent",

        "general_recognizer":
            "Gemini Vision"
            if gemini_client
            else "Unavailable",

        "fallback_model":
            "Food AI V2",

        "fallback_classes":
            len(class_names)

    })


# ============================================================
# API ROUTES
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    return predict_food()


@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    return predict_food()


@app.route(
    "/scan",
    methods=["POST"]
)
def scan():

    return predict_food()


@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    return predict_food()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "========================================"
    )

    print(
        "AI FOOD RECOGNITION"
    )

    print(
        "& RECOMMENDATION AGENT"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Primary recognition: Gemini Vision"
    )

    print(
        "Fallback recognition: Food AI V2"
    )

    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )