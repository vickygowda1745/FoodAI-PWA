import os
import io
import json
import re
from PIL import Image

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

from google import genai
from google.genai import types


# ============================================================
# FOODAI CONFIGURATION
# ============================================================

# Free Gemini recognition only.
# OpenAI/Astra is NOT used in this version.

PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

GITHUB_ORIGIN = "https://vickygowda1745.github.io"

MAX_IMAGE_SIZE = 6 * 1024 * 1024


# ============================================================
# FLASK APP
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

gemini_client = None


if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini Vision connected")

    except Exception as error:

        print(
            "Gemini client error:",
            error
        )

else:

    print(
        "WARNING: GEMINI_API_KEY missing"
    )


# ============================================================
# IMAGE HELPERS
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


def get_image_mime_type(image_bytes):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image_format = (
            image.format or "JPEG"
        ).upper()


        mime_types = {

            "JPEG":
                "image/jpeg",

            "JPG":
                "image/jpeg",

            "PNG":
                "image/png",

            "WEBP":
                "image/webp",

            "GIF":
                "image/gif"

        }


        return mime_types.get(
            image_format,
            "image/jpeg"
        )

    except Exception:

        return "image/jpeg"


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):

    if not text:

        raise ValueError(
            "Empty response from vision model"
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


    text = text.strip()


    try:

        return json.loads(text)

    except Exception:

        # Try extracting JSON object
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL
        )

        if match:

            return json.loads(
                match.group(0)
            )


        raise ValueError(
            "Vision model did not return valid JSON"
        )


# ============================================================
# FOOD NAME NORMALIZATION
# ============================================================

def normalize_food_name(name):

    if not name:

        return "Unknown Food"


    clean = str(name).strip()


    replacements = {

        "shrimp biryani":
            "Prawn Biryani",

        "shrimp biriyani":
            "Prawn Biryani",

        "shrimp curry":
            "Prawn Curry",

        "shrimp masala":
            "Prawn Masala",

        "shrimp fried rice":
            "Prawn Fried Rice",

        "shrimp noodles":
            "Prawn Noodles",

        "chicken biriyani":
            "Chicken Biryani",

        "mutton biriyani":
            "Mutton Biryani",

        "egg biriyani":
            "Egg Biryani",

        "vegetable biriyani":
            "Vegetable Biryani",

        "veg biriyani":
            "Vegetable Biryani",

        "veg biryani":
            "Vegetable Biryani",

        "paneer butter masala curry":
            "Paneer Butter Masala",

        "butter chicken curry":
            "Butter Chicken",

        "masala dosa":
            "Masala Dosa",

        "idly":
            "Idli",

        "idly sambar":
            "Idli Sambar"

    }


    normalized = replacements.get(
        clean.lower()
    )


    if normalized:

        return normalized


    return clean


# ============================================================
# FOOD RECOGNITION PROMPT
# ============================================================

FOOD_RECOGNITION_PROMPT = """
You are FoodAI, an expert visual food recognition system.

Analyze the supplied image carefully and identify the MAIN FOOD
or MAIN DISH visible.

IMPORTANT:

1. The food may come from ANY cuisine in the world.
2. You are NOT restricted to a predefined food list.
3. Identify the most specific dish supported by the image.
4. Do not simply call everything "rice", "curry", "bread",
   "dessert", etc. when a more specific dish is visually clear.
5. Carefully inspect visible ingredients.
6. Pay special attention to differences between visually similar
   foods.
7. For Indian dishes, use commonly understood Indian names.
8. For Indian seafood dishes use "Prawn" rather than "Shrimp".
9. Distinguish varieties of biryani whenever visible evidence
   supports it:
   - Chicken Biryani
   - Mutton Biryani
   - Prawn Biryani
   - Egg Biryani
   - Vegetable Biryani
10. Distinguish biryani from pulao/fried rice when visual evidence
    supports that distinction.
11. Consider:
    - meat
    - seafood
    - vegetables
    - rice
    - bread
    - gravy
    - sauce
    - garnish
    - texture
    - preparation
    - plating
    - regional appearance
12. Do not invent ingredients that are not reasonably visible.
13. If multiple foods are visible, choose the dominant/main dish.
14. If the image clearly does not contain food, mark is_food false.

Return ONLY valid JSON.

Required format:

{
  "is_food": true,
  "food": "Specific Food Name",
  "confidence": 95
}

confidence must be an integer between 0 and 100.

If there is clearly no food:

{
  "is_food": false,
  "food": "Not Food",
  "confidence": 0
}
"""


# ============================================================
# DETAILS PROMPT
# ============================================================

def create_details_prompt(food_name):

    return f"""
You are FoodAI's food information and recommendation engine.

The image has already been identified as:

{food_name}

Analyze the same food image.

Return ONLY valid JSON in this exact structure:

{{
  "cuisine": "Cuisine or regional origin",
  "description": "Short useful description of the food",
  "visible_items": [
    "visible ingredient or item 1",
    "visible ingredient or item 2"
  ],
  "recommendation": "Short serving or pairing recommendation",
  "health_note": "Short general nutrition observation",
  "similar_food": "One similar food"
}}

Rules:

- Keep answers concise.
- Do not provide medical diagnosis.
- Do not claim ingredients that cannot reasonably be inferred.
- Use Indian terminology for Indian foods.
- Do not return Markdown.
- Return JSON only.
"""


# ============================================================
# GEMINI MODEL CALL
# ============================================================

def call_gemini(
    model_name,
    prompt,
    image_bytes
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini Vision is not configured"
        )


    mime_type = get_image_mime_type(
        image_bytes
    )


    response = (
        gemini_client.models.generate_content(

            model=model_name,

            contents=[

                prompt,

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )

            ]

        )
    )


    if not response:

        raise RuntimeError(
            "Empty Gemini response"
        )


    if not response.text:

        raise RuntimeError(
            "Gemini returned no text"
        )


    return response.text


# ============================================================
# OPEN-ENDED FOOD IDENTIFICATION
# ============================================================

def identify_food(image_bytes):

    errors = []


    models = [

        PRIMARY_MODEL,

        FALLBACK_MODEL

    ]


    for model_name in models:

        try:

            print(
                "FoodAI recognition model:",
                model_name
            )


            response_text = call_gemini(

                model_name,

                FOOD_RECOGNITION_PROMPT,

                image_bytes

            )


            result = parse_json(
                response_text
            )


            is_food = result.get(
                "is_food",
                True
            )


            food_name = normalize_food_name(

                result.get(
                    "food",
                    "Unknown Food"
                )

            )


            try:

                confidence = int(
                    result.get(
                        "confidence",
                        0
                    )
                )

            except Exception:

                confidence = 0


            confidence = max(
                0,
                min(
                    100,
                    confidence
                )
            )


            return {

                "is_food":
                    bool(is_food),

                "food":
                    food_name,

                "confidence":
                    confidence,

                "engine":
                    model_name

            }


        except Exception as error:

            print(
                "Recognition failed:",
                model_name,
                error
            )


            errors.append(
                f"{model_name}: {str(error)}"
            )


    raise RuntimeError(
        "Recognition models unavailable. "
        + " | ".join(errors)
    )


# ============================================================
# FOOD DETAILS
# ============================================================

def get_food_details(
    image_bytes,
    food_name
):

    errors = []


    prompt = create_details_prompt(
        food_name
    )


    models = [

        PRIMARY_MODEL,

        FALLBACK_MODEL

    ]


    for model_name in models:

        try:

            response_text = call_gemini(

                model_name,

                prompt,

                image_bytes

            )


            result = parse_json(
                response_text
            )


            result["engine"] = (
                model_name
            )


            return result


        except Exception as error:

            print(
                "Details failed:",
                model_name,
                error
            )


            errors.append(
                f"{model_name}: {str(error)}"
            )


    raise RuntimeError(
        "Food details unavailable. "
        + " | ".join(errors)
    )


# ============================================================
# IDENTIFY ENDPOINT
# ============================================================

@app.route(
    "/identify",
    methods=[
        "POST",
        "OPTIONS"
    ]
)
def identify():

    if request.method == "OPTIONS":

        return make_response(
            "",
            204
        )


    try:

        image_bytes = get_image_bytes()


        if not image_bytes:

            return jsonify({

                "success":
                    False,

                "error":
                    "No image received"

            }), 400


        if not validate_image(
            image_bytes
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Invalid image"

            }), 400


        result = identify_food(
            image_bytes
        )


        if not result[
            "is_food"
        ]:

            return jsonify({

                "success":
                    False,

                "is_food":
                    False,

                "food":
                    "Not Food",

                "confidence":
                    0,

                "message":
                    "No food detected in this image."

            }), 200


        return jsonify({

            "success":
                True,

            "is_food":
                True,

            "food":
                result[
                    "food"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "engine":
                result[
                    "engine"
                ]

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
                "Food recognition temporarily unavailable.",

            "details":
                str(error)

        }), 503


# ============================================================
# LEGACY /predict ENDPOINT
# ============================================================

# Keep this because the existing frontend may still call /predict.

@app.route(
    "/predict",
    methods=[
        "POST",
        "OPTIONS"
    ]
)
def predict():

    if request.method == "OPTIONS":

        return make_response(
            "",
            204
        )


    try:

        image_bytes = get_image_bytes()


        if not image_bytes:

            return jsonify({

                "success":
                    False,

                "error":
                    "No image received"

            }), 400


        if not validate_image(
            image_bytes
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Invalid image"

            }), 400


        result = identify_food(
            image_bytes
        )


        if not result[
            "is_food"
        ]:

            return jsonify({

                "success":
                    False,

                "food":
                    "Not Food",

                "confidence":
                    0

            }), 200


        # Get details after recognition.
        # If details fail, recognition still succeeds.

        details = {}


        try:

            details = get_food_details(

                image_bytes,

                result[
                    "food"
                ]

            )

        except Exception as details_error:

            print(
                "Optional details failed:",
                details_error
            )


        return jsonify({

            "success":
                True,

            "food":
                result[
                    "food"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "engine":
                result[
                    "engine"
                ],

            "cuisine":
                details.get(
                    "cuisine",
                    ""
                ),

            "description":
                details.get(
                    "description",
                    ""
                ),

            "visible_items":
                details.get(
                    "visible_items",
                    []
                ),

            "recommendation":
                details.get(
                    "recommendation",
                    ""
                ),

            "health_note":
                details.get(
                    "health_note",
                    ""
                ),

            "similar_food":
                details.get(
                    "similar_food",
                    ""
                )

        }), 200


    except Exception as error:

        print(
            "PREDICT ERROR:",
            error
        )


        return jsonify({

            "success":
                False,

            "error":
                "Food recognition temporarily unavailable.",

            "details":
                str(error)

        }), 503


# ============================================================
# DETAILS ENDPOINT
# ============================================================

@app.route(
    "/details",
    methods=[
        "POST",
        "OPTIONS"
    ]
)
def details():

    if request.method == "OPTIONS":

        return make_response(
            "",
            204
        )


    try:

        image_bytes = get_image_bytes()


        food_name = request.form.get(
            "food",
            ""
        ).strip()


        if not image_bytes:

            return jsonify({

                "success":
                    False,

                "error":
                    "No image received"

            }), 400


        if not food_name:

            return jsonify({

                "success":
                    False,

                "error":
                    "Food name missing"

            }), 400


        if not validate_image(
            image_bytes
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Invalid image"

            }), 400


        food_name = normalize_food_name(
            food_name
        )


        result = get_food_details(

            image_bytes,

            food_name

        )


        return jsonify({

            "success":
                True,

            "food":
                food_name,

            "engine":
                result.get(
                    "engine",
                    "Gemini Vision"
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


    except Exception as error:

        print(
            "DETAIL ERROR:",
            error
        )


        return jsonify({

            "success":
                False,

            "error":
                "Food details temporarily unavailable.",

            "details":
                str(error)

        }), 503


# ============================================================
# HEALTH CHECK
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

        "recognition_mode":
            "open-ended",

        "recognizer":
            "Gemini Vision",

        "primary_model":
            PRIMARY_MODEL,

        "fallback_model":
            FALLBACK_MODEL,

        "gemini_configured":
            bool(
                GEMINI_API_KEY
            ),

        "openai_used":
            False

    }), 200


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
            "Open-ended food recognition",

        "recognizer":
            "Gemini Vision",

        "openai_used":
            False

    }), 200


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "============================================"
    )

    print(
        " FOODAI"
    )

    print(
        " AI Food Recognition & Recommendation Agent"
    )

    print(
        "============================================"
    )

    print(
        "Primary:",
        PRIMARY_MODEL
    )

    print(
        "Fallback:",
        FALLBACK_MODEL
    )

    print(
        "OpenAI:",
        "DISABLED"
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