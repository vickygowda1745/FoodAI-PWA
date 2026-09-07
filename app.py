import os
import io
import json
import re

from PIL import Image
from flask import (
    Flask,
    request,
    jsonify,
    make_response
)
from flask_cors import CORS

from google import genai
from google.genai import types


# ============================================================
# SETTINGS
# ============================================================

FAST_MODEL = "gemini-3.5-flash-lite"
DETAIL_MODEL = "gemini-3.8-flash"

GITHUB_ORIGIN = "https://vickygowda1745.github.io"


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024


# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": [GITHUB_ORIGIN],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
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


client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini connected")


# ============================================================
# HELPERS
# ============================================================

def get_image_bytes():

    if "image" in request.files:
        return request.files["image"].read()

    if "file" in request.files:
        return request.files["file"].read()

    return request.get_data()


def validate_image(image_bytes):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.verify()

        return True

    except Exception:

        return False


def clean_json(text):

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
# NAME NORMALIZATION
# ============================================================

def normalize_food_name(food_name):

    if not food_name:
        return "Unknown Food"

    name = food_name.strip()

    replacements = {
        "Shrimp Biryani": "Prawn Biryani",
        "Shrimp Fried Rice": "Prawn Fried Rice",
        "Shrimp Curry": "Prawn Curry",
        "Chicken Biriyani": "Chicken Biryani",
        "Veg Biriyani": "Vegetable Biryani",
        "Vegetable Biriyani": "Vegetable Biryani"
    }

    return replacements.get(
        name,
        name
    )


# ============================================================
# FAST FOOD IDENTIFICATION
# ============================================================

def identify_food_fast(image_bytes):

    prompt = """
Identify the main food in this image.

Return ONLY valid JSON:

{
  "food": "specific food name",
  "confidence": 0
}

Rules:
- Do not give explanation.
- Do not give recommendations.
- Do not return markdown.
- Use the most specific food name you can.
- Prefer Indian naming when appropriate.
- Use "Prawn" instead of "Shrimp" for Indian dishes.
- confidence must be an integer from 0 to 100.
"""

    response = client.models.generate_content(
        model=FAST_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]
    )

    result = clean_json(
        response.text
    )

    result["food"] = normalize_food_name(
        result.get(
            "food",
            "Unknown Food"
        )
    )

    return result


# ============================================================
# DETAILED RECOMMENDATION
# ============================================================

def get_food_details(
    image_bytes,
    detected_food
):

    prompt = f"""
The food has already been identified as:

{detected_food}

Analyze the image and return useful recommendation details.

Return ONLY valid JSON:

{{
  "cuisine": "cuisine or Unknown",
  "description": "short description",
  "visible_items": ["item 1", "item 2"],
  "recommendation": "best serving or pairing suggestion",
  "health_note": "short general nutrition note",
  "similar_food": "similar dish"
}}

Rules:
- Keep answers short and practical.
- Do not provide medical advice.
- Do not return markdown.
"""

    response = client.models.generate_content(
        model=DETAIL_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]
    )

    return clean_json(
        response.text
    )


# ============================================================
# FAST IDENTIFY ENDPOINT
# ============================================================

@app.route(
    "/identify",
    methods=["POST"]
)
def identify():

    try:

        image_bytes = get_image_bytes()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400


        if not validate_image(
            image_bytes
        ):

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400


        result = identify_food_fast(
            image_bytes
        )


        return jsonify({

            "success": True,

            "food":
                result.get(
                    "food",
                    "Unknown Food"
                ),

            "confidence":
                result.get(
                    "confidence",
                    0
                )

        }), 200


    except Exception as e:

        print(
            "Fast identify error:",
            e
        )

        return jsonify({

            "success": False,

            "error":
                "Fast recognition temporarily unavailable.",

            "details":
                str(e)

        }), 503


# ============================================================
# DETAILS ENDPOINT
# ============================================================

@app.route(
    "/details",
    methods=["POST"]
)
def details():

    try:

        image_bytes = get_image_bytes()

        food_name = request.form.get(
            "food",
            ""
        )


        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400


        if not food_name:

            return jsonify({
                "success": False,
                "error": "Food name missing"
            }), 400


        if not validate_image(
            image_bytes
        ):

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400


        result = get_food_details(
            image_bytes,
            food_name
        )


        return jsonify({

            "success": True,

            "food":
                normalize_food_name(
                    food_name
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
            "Details error:",
            e
        )

        return jsonify({

            "success": False,

            "error":
                "Recommendation details temporarily unavailable.",

            "details":
                str(e)

        }), 503


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

        "fast_model":
            FAST_MODEL,

        "detail_model":
            DETAIL_MODEL,

        "mode":
            "Fast identify + background recommendations"

    })


# ============================================================
# OPTIONS
# ============================================================

@app.route(
    "/identify",
    methods=["OPTIONS"]
)
@app.route(
    "/details",
    methods=["OPTIONS"]
)
def options():

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

        "frontend":
            "https://vickygowda1745.github.io/FoodAI-PWA/"

    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "AI Food Recognition & Recommendation Agent"
    )

    print(
        "Fast model:",
        FAST_MODEL
    )

    print(
        "Detail model:",
        DETAIL_MODEL
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