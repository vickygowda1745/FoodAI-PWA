import os
import io
import json
import re
import time

from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from google import genai
from google.genai import types


# ============================================================
# SETTINGS
# ============================================================

PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ============================================================
# GEMINI CLIENT
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini Vision connected")


# ============================================================
# IMAGE INPUT
# ============================================================

def get_image_bytes():

    if "image" in request.files:
        return request.files["image"].read()

    if "file" in request.files:
        return request.files["file"].read()

    return request.get_data()


# ============================================================
# GEMINI REQUEST
# ============================================================

def call_gemini(model_name, prompt, image_bytes):

    return gemini_client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]
    )


# ============================================================
# OPEN-ENDED FOOD RECOGNITION
# ============================================================

def recognize_food(image_bytes):

    prompt = """
You are an AI Food Recognition & Recommendation Agent.

Analyze this image carefully.

Identify the food as specifically as reasonably possible.

Do not restrict yourself to any predefined food list.

The food can be from any cuisine in the world.

If multiple foods are visible, identify the main dish and list other visible items.

Return ONLY valid JSON in this exact format:

{
  "is_food": true,
  "food": "specific food name",
  "confidence": 0,
  "cuisine": "cuisine or Unknown",
  "description": "short description",
  "visible_items": ["item 1", "item 2"],
  "recommendation": "best pairing or serving suggestion",
  "health_note": "short general nutrition note",
  "similar_food": "similar dish"
}

confidence must be an integer between 0 and 100.

If the image does not contain food, return:

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

    last_error = None

    # --------------------------------------------------------
    # TRY PRIMARY MODEL
    # --------------------------------------------------------

    for attempt in range(MAX_RETRIES):

        try:

            response = call_gemini(
                PRIMARY_MODEL,
                prompt,
                image_bytes
            )

            return parse_gemini_response(
                response.text
            )

        except Exception as e:

            last_error = e

            error_text = str(e)

            print(
                f"Primary Gemini attempt "
                f"{attempt + 1} failed:",
                error_text
            )

            if (
                "503" in error_text
                or
                "UNAVAILABLE" in error_text
                or
                "high demand" in error_text.lower()
            ):
                time.sleep(
                    RETRY_DELAY_SECONDS
                )
                continue

            break


    # --------------------------------------------------------
    # TRY FALLBACK MODEL
    # --------------------------------------------------------

    try:

        print(
            "Trying fallback Gemini model..."
        )

        response = call_gemini(
            FALLBACK_MODEL,
            prompt,
            image_bytes
        )

        return parse_gemini_response(
            response.text
        )

    except Exception as fallback_error:

        print(
            "Fallback Gemini error:",
            fallback_error
        )

        raise RuntimeError(
            f"Gemini temporarily unavailable. "
            f"Primary error: {last_error}. "
            f"Fallback error: {fallback_error}"
        )


# ============================================================
# PARSE GEMINI JSON
# ============================================================

def parse_gemini_response(text):

    text = text.strip()

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
# MAIN PREDICTION
# ============================================================

def predict_food():

    try:

        image_bytes = get_image_bytes()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400


        # Validate image

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            image.verify()

        except Exception:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400


        result = recognize_food(
            image_bytes
        )


        if not result.get(
            "is_food",
            True
        ):

            return jsonify({
                "success": False,
                "food": "Not food",
                "confidence": 0,
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


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({

            "success":
                False,

            "error":
                "Recognition service is temporarily busy. "
                "Please try scanning again in a few seconds.",

            "details":
                str(e)

        }), 503


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

        "recognizer":
            "Gemini Vision",

        "mode":
            "Open-ended food recognition",

        "primary_model":
            PRIMARY_MODEL,

        "fallback_model":
            FALLBACK_MODEL
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
        "================================"
    )
    print(
        "AI FOOD RECOGNITION"
    )
    print(
        "& RECOMMENDATION AGENT"
    )
    print(
        "================================"
    )
    print()

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),
        debug=False
    )