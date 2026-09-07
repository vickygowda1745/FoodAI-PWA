import os
import io
import json
import re
import time

from PIL import Image
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    make_response
)
from flask_cors import CORS

from google import genai
from google.genai import types


# ============================================================
# SETTINGS
# ============================================================

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite"
]

MAX_RETRIES_PER_MODEL = 1
RETRY_DELAY_SECONDS = 1

GITHUB_ORIGIN = "https://vickygowda1745.github.io"


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                GITHUB_ORIGIN
            ],
            "methods": [
                "GET",
                "POST",
                "OPTIONS"
            ],
            "allow_headers": [
                "Content-Type"
            ]
        }
    }
)


@app.after_request
def add_cors_headers(response):

    origin = request.headers.get("Origin")

    if origin == GITHUB_ORIGIN:

        response.headers[
            "Access-Control-Allow-Origin"
        ] = GITHUB_ORIGIN

        response.headers[
            "Access-Control-Allow-Methods"
        ] = "GET, POST, OPTIONS"

        response.headers[
            "Access-Control-Allow-Headers"
        ] = "Content-Type"

    return response


# ============================================================
# GEMINI CLIENT
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is not set"
    )


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini Vision connected")


# ============================================================
# IMAGE INPUT
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
# PARSE JSON
# ============================================================

def parse_json_response(text):

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
# GEMINI CALL
# ============================================================

def call_model(
    model_name,
    prompt,
    image_bytes
):

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

Carefully analyze this image.

Your job is to identify the food as specifically as reasonably possible.

IMPORTANT:

- Do NOT restrict yourself to a fixed list.
- Food may come from any country or cuisine.
- It may be homemade, restaurant food, street food, packaged food,
  dessert, drink, snack, breakfast, lunch or dinner.
- If several foods are visible, identify the MAIN food and list
  other visible foods separately.
- If the exact dish is uncertain, give the most likely dish name,
  but lower the confidence.
- Never pretend to have 100 percent certainty unless visually obvious.

Return ONLY valid JSON:

{
  "is_food": true,
  "food": "specific food name",
  "confidence": 0,
  "cuisine": "cuisine or Unknown",
  "description": "short description",
  "visible_items": ["item 1", "item 2"],
  "recommendation": "best serving or pairing recommendation",
  "health_note": "short general nutrition note",
  "similar_food": "similar dish"
}

confidence must be an integer between 0 and 100.

If this is not food:

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

    errors = []

    # Try different Gemini models one by one
    for model_name in MODELS:

        for attempt in range(
            MAX_RETRIES_PER_MODEL + 1
        ):

            try:

                print(
                    "Trying model:",
                    model_name,
                    "attempt:",
                    attempt + 1
                )

                response = call_model(
                    model_name,
                    prompt,
                    image_bytes
                )

                result = parse_json_response(
                    response.text
                )

                result["model_used"] = (
                    model_name
                )

                return result


            except Exception as e:

                error_text = str(e)

                errors.append(
                    f"{model_name}: {error_text}"
                )

                print(
                    "Model failed:",
                    model_name,
                    error_text
                )

                temporary_error = (
                    "503" in error_text
                    or
                    "UNAVAILABLE" in error_text
                    or
                    "high demand"
                    in error_text.lower()
                    or
                    "429" in error_text
                )

                if temporary_error:

                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

                    continue

                break


    raise RuntimeError(
        "All recognition models failed. "
        + " | ".join(errors)
    )


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
                io.BytesIO(
                    image_bytes
                )
            )

            image.verify()

        except Exception:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400


        # Recognition

        result = recognize_food(
            image_bytes
        )


        # Not food

        if not result.get(
            "is_food",
            True
        ):

            return jsonify({

                "success":
                    False,

                "food":
                    "Not food",

                "confidence":
                    0,

                "message":
                    "No food confidently detected."

            }), 200


        # Success

        return jsonify({

            "success":
                True,

            "recognizer":
                "Gemini Vision",

            "model_used":
                result.get(
                    "model_used",
                    ""
                ),

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

        }), 200


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({

            "success":
                False,

            "error":
                "Recognition is temporarily unavailable. "
                "Please try again shortly.",

            "details":
                str(e)

        }), 503


# ============================================================
# OPTIONS
# ============================================================

@app.route(
    "/predict",
    methods=["OPTIONS"]
)
@app.route(
    "/scan",
    methods=["OPTIONS"]
)
@app.route(
    "/analyze",
    methods=["OPTIONS"]
)
@app.route(
    "/api/predict",
    methods=["OPTIONS"]
)
def handle_options():

    response = make_response(
        "",
        204
    )

    response.headers[
        "Access-Control-Allow-Origin"
    ] = GITHUB_ORIGIN

    response.headers[
        "Access-Control-Allow-Methods"
    ] = "POST, OPTIONS"

    response.headers[
        "Access-Control-Allow-Headers"
    ] = "Content-Type"

    return response


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({

        "project":
            "AI Food Recognition & Recommendation Agent",

        "status":
            "online",

        "recognition":
            "Open-ended Gemini Vision",

        "models":
            MODELS,

        "frontend":
            "https://vickygowda1745.github.io/FoodAI-PWA/"

    })


# ============================================================
# HEALTH
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

        "models":
            MODELS,

        "cors_origin":
            GITHUB_ORIGIN

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
    "/scan",
    methods=["POST"]
)
def scan():

    return predict_food()


@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    return predict_food()


@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    return predict_food()


# ============================================================
# STATIC FILES
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
# START
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

    print(
        "Open-ended recognition enabled"
    )

    print(
        "Models:",
        MODELS
    )

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