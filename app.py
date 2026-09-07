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

PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

GITHUB_ORIGIN = "https://vickygowda1745.github.io"


# ============================================================
# FLASK APP
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

    origin = request.headers.get(
        "Origin"
    )

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

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

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
# GEMINI CALL
# ============================================================

def call_gemini(
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
# PARSE GEMINI RESPONSE
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
# OPEN-ENDED FOOD RECOGNITION
# ============================================================

def recognize_food(
    image_bytes
):

    prompt = """
You are an AI Food Recognition & Recommendation Agent.

Carefully analyze the food image.

Your task:

1. Identify the food as specifically as reasonably possible.
2. Do NOT restrict yourself to a predefined food list.
3. The dish can belong to any cuisine in the world.
4. If several foods are visible, identify the main dish and list important visible side items.
5. If uncertain, do not pretend to be certain.
6. Provide practical food recommendations.
7. Do not provide medical advice.

Return ONLY valid JSON:

{
  "is_food": true,
  "food": "specific food name",
  "confidence": 0,
  "cuisine": "cuisine or Unknown",
  "description": "short description",
  "visible_items": ["item 1", "item 2"],
  "recommendation": "pairing or serving recommendation",
  "health_note": "short general nutrition note",
  "similar_food": "similar food"
}

confidence must be an integer from 0 to 100.

If the image does not contain recognizable food, return:

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


    # ========================================================
    # PRIMARY MODEL RETRIES
    # ========================================================

    for attempt in range(
        MAX_RETRIES
    ):

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
                "Primary Gemini attempt",
                attempt + 1,
                "failed:",
                error_text
            )


            temporary_error = (
                "503" in error_text
                or
                "UNAVAILABLE" in error_text
                or
                "high demand"
                in error_text.lower()
            )


            if temporary_error:

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

                continue


            break


    # ========================================================
    # FALLBACK MODEL
    # ========================================================

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
            "Gemini recognition temporarily unavailable. "
            f"Primary error: {last_error}. "
            f"Fallback error: {fallback_error}"
        )


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


        # ====================================================
        # VALIDATE IMAGE
        # ====================================================

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


        # ====================================================
        # GEMINI RECOGNITION
        # ====================================================

        result = recognize_food(
            image_bytes
        )


        # ====================================================
        # NOT FOOD
        # ====================================================

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

            }), 200


        # ====================================================
        # SUCCESS RESPONSE
        # ====================================================

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
                "Recognition service is temporarily busy. "
                "Please try again in a few seconds.",

            "details":
                str(e)

        }), 503


# ============================================================
# OPTIONS / PREFLIGHT
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

        "scanner":
            "Gemini Vision",

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

        "primary_model":
            PRIMARY_MODEL,

        "fallback_model":
            FALLBACK_MODEL,

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
# OPTIONAL STATIC PWA FILES
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
        "Recognition: Gemini Vision"
    )
    print(
        "Mode: Open-ended food recognition"
    )
    print(
        "GitHub Pages CORS: ENABLED"
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