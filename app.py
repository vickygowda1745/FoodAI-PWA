import os
import io
import json
import re
import base64

from PIL import Image
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

from openai import OpenAI
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

OPENAI_MODEL = "gpt-5.6-luna"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"

GITHUB_ORIGIN = "https://vickygowda1745.github.io"

MAX_IMAGE_SIZE = 6 * 1024 * 1024


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE


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
def cors_headers(response):

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
# API CLIENTS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY missing")


if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY missing")


openai_client = None

gemini_client = None


if OPENAI_API_KEY:

    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=12.0,
        max_retries=0
    )


if GEMINI_API_KEY:

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# IMAGE
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


def validate_image(image_bytes):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.verify()

        return True

    except Exception:

        return False


# ============================================================
# JSON CLEANER
# ============================================================

def parse_json(text):

    if not text:

        raise ValueError(
            "Empty AI response"
        )


    text = text.strip()


    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"^```\s*",
        "",
        text
    )


    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    return json.loads(
        text.strip()
    )


# ============================================================
# FOOD NAME NORMALIZATION
# ============================================================

def normalize_food_name(name):

    if not name:

        return "Unknown Food"


    clean = name.strip()


    replacements = {

        "shrimp biryani":
            "Prawn Biryani",

        "shrimp biriyani":
            "Prawn Biryani",

        "shrimp curry":
            "Prawn Curry",

        "shrimp fried rice":
            "Prawn Fried Rice",

        "chicken biriyani":
            "Chicken Biryani",

        "veg biriyani":
            "Vegetable Biryani",

        "vegetable biriyani":
            "Vegetable Biryani",

        "mutton biriyani":
            "Mutton Biryani",

        "egg biriyani":
            "Egg Biryani"

    }


    return replacements.get(
        clean.lower(),
        clean
    )


# ============================================================
# PROMPT
# ============================================================

FAST_PROMPT = """
You are the visual recognition engine for an AI Food Recognition
and Recommendation Agent.

Analyze the supplied photograph.

Your FIRST task is precise food identification.

The image may contain ANY food from ANY cuisine in the world.
You are NOT restricted to a predefined list.

Pay attention to:
- ingredients
- rice type
- meat or seafood type
- gravy
- bread type
- cooking style
- garnish
- texture
- regional presentation
- side dishes

For Indian dishes use common Indian naming conventions.

Examples:
shrimp biryani -> Prawn Biryani
chicken biriyani -> Chicken Biryani

Do NOT invent an exact dish when the visual evidence is insufficient.

If several dishes are visible, identify the dominant/main dish.

Return ONLY valid JSON.

{
  "is_food": true,
  "food": "specific dish name",
  "confidence": 0
}

confidence must be an integer from 0 to 100.

If this is clearly not food return:

{
  "is_food": false,
  "food": "Not Food",
  "confidence": 0
}
"""


DETAIL_PROMPT_TEMPLATE = """
You are the recommendation component of an AI Food Recognition
and Recommendation Agent.

The image has already been identified as:

FOOD_NAME

Analyze the image and provide useful concise information.

Return ONLY valid JSON:

{
  "cuisine": "cuisine",
  "description": "short description",
  "visible_items": ["item 1", "item 2"],
  "recommendation": "serving or pairing recommendation",
  "health_note": "short general nutritional observation",
  "similar_food": "one similar dish"
}

Do not provide medical advice.
Do not return Markdown.
"""


# ============================================================
# OPENAI IMAGE FORMAT
# ============================================================

def image_data_url(image_bytes):

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# OPENAI FAST RECOGNITION
# ============================================================

def identify_with_openai(image_bytes):

    if not openai_client:

        raise RuntimeError(
            "OpenAI client unavailable"
        )


    response = openai_client.responses.create(

        model=OPENAI_MODEL,

        reasoning={
            "effort": "none"
        },

        max_output_tokens=100,

        input=[
            {
                "role": "user",

                "content": [
                    {
                        "type": "input_text",
                        "text": FAST_PROMPT
                    },

                    {
                        "type": "input_image",
                        "image_url":
                            image_data_url(
                                image_bytes
                            ),
                        "detail": "low"
                    }
                ]
            }
        ]
    )


    result = parse_json(
        response.output_text
    )


    result["food"] = normalize_food_name(
        result.get(
            "food",
            "Unknown Food"
        )
    )


    result["engine"] = (
        "OpenAI GPT-5.6 Luna"
    )


    return result


# ============================================================
# GEMINI FALLBACK
# ============================================================

def identify_with_gemini(image_bytes):

    if not gemini_client:

        raise RuntimeError(
            "Gemini fallback unavailable"
        )


    response = (
        gemini_client.models.generate_content(

            model=GEMINI_FALLBACK_MODEL,

            contents=[
                FAST_PROMPT,

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            ]
        )
    )


    result = parse_json(
        response.text
    )


    result["food"] = normalize_food_name(
        result.get(
            "food",
            "Unknown Food"
        )
    )


    result["engine"] = (
        "Gemini fallback"
    )


    return result


# ============================================================
# HYBRID IDENTIFICATION
# ============================================================

def identify_food(image_bytes):

    openai_error = None


    # PRIMARY
    try:

        print(
            "Trying OpenAI:",
            OPENAI_MODEL
        )


        result = identify_with_openai(
            image_bytes
        )


        return result


    except Exception as error:

        openai_error = str(error)

        print(
            "OpenAI recognition failed:",
            openai_error
        )


    # FALLBACK
    try:

        print(
            "Trying Gemini fallback:",
            GEMINI_FALLBACK_MODEL
        )


        result = identify_with_gemini(
            image_bytes
        )


        return result


    except Exception as error:

        print(
            "Gemini fallback failed:",
            error
        )


        raise RuntimeError(
            "Both recognition engines failed. "
            "OpenAI: "
            + str(openai_error)
            + " | Gemini: "
            + str(error)
        )


# ============================================================
# OPENAI DETAILS
# ============================================================

def details_with_openai(
    image_bytes,
    food_name
):

    if not openai_client:

        raise RuntimeError(
            "OpenAI client unavailable"
        )


    prompt = (
        DETAIL_PROMPT_TEMPLATE
        .replace(
            "FOOD_NAME",
            food_name
        )
    )


    response = openai_client.responses.create(

        model=OPENAI_MODEL,

        reasoning={
            "effort": "none"
        },

        max_output_tokens=350,

        input=[
            {
                "role": "user",

                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },

                    {
                        "type": "input_image",
                        "image_url":
                            image_data_url(
                                image_bytes
                            ),
                        "detail": "low"
                    }
                ]
            }
        ]
    )


    return parse_json(
        response.output_text
    )


# ============================================================
# GEMINI DETAILS FALLBACK
# ============================================================

def details_with_gemini(
    image_bytes,
    food_name
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini unavailable"
        )


    prompt = (
        DETAIL_PROMPT_TEMPLATE
        .replace(
            "FOOD_NAME",
            food_name
        )
    )


    response = (
        gemini_client.models.generate_content(

            model=GEMINI_FALLBACK_MODEL,

            contents=[
                prompt,

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            ]
        )
    )


    return parse_json(
        response.text
    )


# ============================================================
# /identify
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


        result = identify_food(
            image_bytes
        )


        if not result.get(
            "is_food",
            True
        ):

            return jsonify({

                "success": False,

                "food":
                    "Not Food",

                "confidence":
                    0,

                "message":
                    "No food detected."

            }), 200


        return jsonify({

            "success":
                True,

            "food":
                normalize_food_name(
                    result.get(
                        "food",
                        "Unknown Food"
                    )
                ),

            "confidence":
                result.get(
                    "confidence",
                    0
                ),

            "engine":
                result.get(
                    "engine",
                    "AI Vision"
                )

        }), 200


    except Exception as error:

        print(
            "IDENTIFY ERROR:",
            error
        )


        return jsonify({

            "success":
                False,

            "error":
                "Food recognition failed.",

            "details":
                str(error)

        }), 503


# ============================================================
# /details
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


        try:

            result = details_with_openai(
                image_bytes,
                food_name
            )


            engine = (
                "OpenAI GPT-5.6 Luna"
            )


        except Exception as error:

            print(
                "OpenAI details failed:",
                error
            )


            result = details_with_gemini(
                image_bytes,
                food_name
            )


            engine = (
                "Gemini fallback"
            )


        return jsonify({

            "success":
                True,

            "food":
                normalize_food_name(
                    food_name
                ),

            "engine":
                engine,

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


    except Exception as error:

        print(
            "DETAIL ERROR:",
            error
        )


        return jsonify({

            "success":
                False,

            "error":
                "Recommendation details unavailable.",

            "details":
                str(error)

        }), 503


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "online",

        "project":
            "AI Food Recognition & Recommendation Agent",

        "recognition":
            "Open-ended",

        "primary_engine":
            "OpenAI",

        "primary_model":
            OPENAI_MODEL,

        "fallback_engine":
            "Gemini",

        "fallback_model":
            GEMINI_FALLBACK_MODEL,

        "openai_configured":
            bool(
                OPENAI_API_KEY
            ),

        "gemini_configured":
            bool(
                GEMINI_API_KEY
            )

    })


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "project":
            "AI Food Recognition & Recommendation Agent",

        "status":
            "online",

        "recognition":
            "Open-ended hybrid vision"

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
# START
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "======================================"
    )
    print(
        "AI FOOD RECOGNITION"
    )
    print(
        "& RECOMMENDATION AGENT"
    )
    print(
        "======================================"
    )

    print(
        "Primary:",
        OPENAI_MODEL
    )

    print(
        "Fallback:",
        GEMINI_FALLBACK_MODEL
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