import os
import io
import json
import re
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone

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
# FOODAI SETTINGS
# ============================================================

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
            "origins": [
                GITHUB_ORIGIN
            ],
            "methods": [
                "GET",
                "POST",
                "OPTIONS"
            ],
            "allow_headers": [
                "Content-Type",
                "X-FoodAI-Session"
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
        ] = (
            "Content-Type, X-FoodAI-Session"
        )

    return response


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

gemini_client = None


if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "Gemini Vision connected"
        )

    except Exception as error:

        print(
            "Gemini connection error:",
            error
        )

else:

    print(
        "WARNING: GEMINI_API_KEY missing"
    )


# ============================================================
# TELEGRAM SETTINGS
# ============================================================

# DO NOT put the actual token or chat ID in this file.
# They will be stored in Render Environment Variables.

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)


def telegram_configured():

    return bool(
        TELEGRAM_BOT_TOKEN
        and
        TELEGRAM_CHAT_ID
    )


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def send_telegram_message(text):

    if not telegram_configured():

        print(
            "Telegram not configured"
        )

        return


    try:

        url = (
            "https://api.telegram.org/bot"
            + TELEGRAM_BOT_TOKEN
            + "/sendMessage"
        )


        payload = urllib.parse.urlencode(
            {
                "chat_id":
                    TELEGRAM_CHAT_ID,

                "text":
                    text
            }
        ).encode(
            "utf-8"
        )


        request_object = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )


        urllib.request.urlopen(
            request_object,
            timeout=8
        ).read()


        print(
            "Telegram notification sent"
        )


    except Exception as error:

        # Telegram failure must NEVER break
        # food recognition.

        print(
            "Telegram error:",
            error
        )


# ============================================================
# DEVICE / SESSION INFORMATION
# ============================================================

def get_scan_context():

    user_agent = request.headers.get(
        "User-Agent",
        "Unknown device"
    )


    session_id = request.headers.get(
        "X-FoodAI-Session",
        "Not supplied"
    )


    # Keep only a short browser/device description.
    # We are NOT collecting IMEI, phone number,
    # MAC address or permanent device identifiers.

    if len(user_agent) > 160:

        user_agent = (
            user_agent[:160]
            + "..."
        )


    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


    return {

        "user_agent":
            user_agent,

        "session_id":
            session_id,

        "time":
            timestamp

    }


# ============================================================
# ASYNC SCAN NOTIFICATION
# ============================================================

def notify_scan(
    food_name,
    confidence,
    engine,
    context
):

    message = (
        "🍽️ FOODAI — NEW SCAN\n\n"
        f"Food: {food_name}\n"
        f"Confidence: {confidence}%\n"
        f"Recognition: {engine}\n"
        f"Time: {context['time']}\n"
        f"Session: {context['session_id']}\n"
        f"Device/Browser: {context['user_agent']}"
    )


    thread = threading.Thread(
        target=send_telegram_message,
        args=(message,),
        daemon=True
    )


    thread.start()


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


def validate_image(
    image_bytes
):

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        image.verify()

        return True


    except Exception:

        return False


def get_image_mime_type(
    image_bytes
):

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )


        image_format = (
            image.format
            or
            "JPEG"
        ).upper()


        mime_map = {

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


        return mime_map.get(
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


    try:

        return json.loads(
            text
        )


    except Exception:

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
            "AI did not return valid JSON"
        )


# ============================================================
# FOOD NAME NORMALIZATION
# ============================================================

def normalize_food_name(name):

    if not name:

        return "Unknown Food"


    clean = str(
        name
    ).strip()


    replacements = {

        "shrimp biryani":
            "Prawn Biryani",

        "shrimp biriyani":
            "Prawn Biryani",

        "shrimp curry":
            "Prawn Curry",

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

        "veg biriyani":
            "Vegetable Biryani",

        "veg biryani":
            "Vegetable Biryani",

        "vegetable biriyani":
            "Vegetable Biryani",

        "idly":
            "Idli",

        "idly sambar":
            "Idli Sambar",

        "butter chicken curry":
            "Butter Chicken",

        "paneer butter masala curry":
            "Paneer Butter Masala"

    }


    return replacements.get(
        clean.lower(),
        clean
    )


# ============================================================
# FOOD RECOGNITION PROMPT
# ============================================================

RECOGNITION_PROMPT = """
You are FoodAI, an expert visual food recognition system.

Analyze this image and identify the MAIN food or dish.

The image may contain ANY food from ANY cuisine in the world.
You are NOT restricted to a fixed food list.

Identify the MOST SPECIFIC dish reasonably supported by
visible evidence.

Look carefully at:

- meat or seafood type
- rice variety
- bread type
- gravy
- vegetables
- sauce
- garnish
- texture
- cooking method
- plating
- regional appearance

If it is biryani, distinguish whenever visually possible:

Chicken Biryani
Mutton Biryani
Prawn Biryani
Egg Biryani
Vegetable Biryani

For Indian foods use common Indian terminology.

Use Prawn instead of Shrimp for Indian dishes.

Do not answer only "Biryani" when the protein or variety
is clearly identifiable.

Return ONLY valid JSON:

{
  "is_food": true,
  "food": "Specific Food Name",
  "confidence": 95
}

confidence must be an integer from 0 to 100.

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

def create_details_prompt(
    food_name
):

    return f"""
You are the FoodAI recommendation engine.

Food:

{food_name}

Give concise useful information about this food.

Return ONLY valid JSON:

{{
  "cuisine": "Cuisine or region",
  "description": "Short food description",
  "visible_items": [
    "item 1",
    "item 2"
  ],
  "recommendation": "Best serving or pairing recommendation",
  "health_note": "Short general nutritional observation",
  "similar_food": "One similar dish"
}}

Do not give medical advice.

Use Indian terminology for Indian foods.

Return JSON only.
"""


# ============================================================
# SEARCH PROMPT
# ============================================================

def create_search_prompt(
    query
):

    return f"""
You are FoodAI's food search engine.

The user searched for:

{query}

Interpret the intended food or dish.

Return ONLY valid JSON:

{{
  "food": "Canonical food name",
  "cuisine": "Cuisine or region",
  "description": "Short description",
  "recommendation": "Serving or pairing recommendation",
  "health_note": "Short general nutritional observation",
  "similar_food": "One similar dish"
}}

If the query contains spelling mistakes, infer the most likely
food name.

Use Indian terminology for Indian dishes.

Return JSON only.
"""


# ============================================================
# GEMINI CALL
# ============================================================

def call_gemini_with_image(
    model,
    prompt,
    image_bytes
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini is not configured"
        )


    mime_type = get_image_mime_type(
        image_bytes
    )


    response = (
        gemini_client.models.generate_content(

            model=model,

            contents=[

                prompt,

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )

            ]

        )
    )


    if not response.text:

        raise RuntimeError(
            "Empty AI response"
        )


    return response.text


def call_gemini_text(
    model,
    prompt
):

    if not gemini_client:

        raise RuntimeError(
            "Gemini is not configured"
        )


    response = (
        gemini_client.models.generate_content(

            model=model,

            contents=prompt

        )
    )


    if not response.text:

        raise RuntimeError(
            "Empty AI response"
        )


    return response.text


# ============================================================
# FOOD IDENTIFICATION
# ============================================================

def identify_food(
    image_bytes
):

    errors = []


    models = [

        PRIMARY_MODEL,

        FALLBACK_MODEL

    ]


    for model in models:

        try:

            print(
                "Trying recognition:",
                model
            )


            response_text = (
                call_gemini_with_image(

                    model,

                    RECOGNITION_PROMPT,

                    image_bytes

                )
            )


            data = parse_json(
                response_text
            )


            food = normalize_food_name(
                data.get(
                    "food",
                    "Unknown Food"
                )
            )


            try:

                confidence = int(
                    data.get(
                        "confidence",
                        0
                    )
                )

            except Exception:

                confidence = 0


            confidence = max(
                0,
                min(
                    confidence,
                    100
                )
            )


            return {

                "is_food":
                    bool(
                        data.get(
                            "is_food",
                            True
                        )
                    ),

                "food":
                    food,

                "confidence":
                    confidence,

                "engine":
                    model

            }


        except Exception as error:

            print(
                "Recognition model failed:",
                model,
                error
            )


            errors.append(
                f"{model}: {error}"
            )


    raise RuntimeError(
        "Recognition unavailable: "
        + " | ".join(errors)
    )


# ============================================================
# FOOD DETAILS
# ============================================================

def generate_details(
    image_bytes,
    food_name
):

    prompt = create_details_prompt(
        food_name
    )


    errors = []


    for model in [

        PRIMARY_MODEL,

        FALLBACK_MODEL

    ]:

        try:

            response_text = (
                call_gemini_with_image(

                    model,

                    prompt,

                    image_bytes

                )
            )


            result = parse_json(
                response_text
            )


            result[
                "engine"
            ] = model


            return result


        except Exception as error:

            errors.append(
                f"{model}: {error}"
            )


    raise RuntimeError(
        "Food details unavailable: "
        + " | ".join(errors)
    )


# ============================================================
# SEARCH FOOD
# ============================================================

def search_food_information(
    query
):

    prompt = create_search_prompt(
        query
    )


    errors = []


    for model in [

        PRIMARY_MODEL,

        FALLBACK_MODEL

    ]:

        try:

            response_text = (
                call_gemini_text(

                    model,

                    prompt

                )
            )


            result = parse_json(
                response_text
            )


            result[
                "food"
            ] = normalize_food_name(

                result.get(
                    "food",
                    query
                )

            )


            result[
                "engine"
            ] = model


            return result


        except Exception as error:

            errors.append(
                f"{model}: {error}"
            )


    raise RuntimeError(
        "Food search unavailable: "
        + " | ".join(errors)
    )


# ============================================================
# IDENTIFY ROUTE
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

        scan_context = (
            get_scan_context()
        )


        image_bytes = (
            get_image_bytes()
        )


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
                    0,

                "message":
                    "No food detected."

            }), 200


        # Telegram notification.
        # Runs in background.

        notify_scan(

            result[
                "food"
            ],

            result[
                "confidence"
            ],

            result[
                "engine"
            ],

            scan_context

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
# LEGACY PREDICT ROUTE
# ============================================================

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

        scan_context = (
            get_scan_context()
        )


        image_bytes = (
            get_image_bytes()
        )


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


        notify_scan(

            result[
                "food"
            ],

            result[
                "confidence"
            ],

            result[
                "engine"
            ],

            scan_context

        )


        details = {}


        try:

            details = (
                generate_details(

                    image_bytes,

                    result[
                        "food"
                    ]

                )
            )


        except Exception as detail_error:

            print(
                "Detail generation failed:",
                detail_error
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
# DETAILS ROUTE
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

        image_bytes = (
            get_image_bytes()
        )


        food_name = (
            request.form.get(
                "food",
                ""
            ).strip()
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


        food_name = (
            normalize_food_name(
                food_name
            )
        )


        result = (
            generate_details(

                image_bytes,

                food_name

            )
        )


        return jsonify({

            "success":
                True,

            "food":
                food_name,

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
# SEARCH ROUTE
# ============================================================

@app.route(
    "/search",
    methods=[
        "POST",
        "OPTIONS"
    ]
)
def search():

    if request.method == "OPTIONS":

        return make_response(
            "",
            204
        )


    try:

        data = request.get_json(
            silent=True
        ) or {}


        query = str(
            data.get(
                "query",
                ""
            )
        ).strip()


        if not query:

            return jsonify({

                "success":
                    False,

                "error":
                    "Search query is required"

            }), 400


        if len(query) > 100:

            return jsonify({

                "success":
                    False,

                "error":
                    "Search query is too long"

            }), 400


        result = (
            search_food_information(
                query
            )
        )


        return jsonify({

            "success":
                True,

            "query":
                query,

            "food":
                result.get(
                    "food",
                    query
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
                ),

            "engine":
                result.get(
                    "engine",
                    "FoodAI"
                )

        }), 200


    except Exception as error:

        print(
            "SEARCH ERROR:",
            error
        )


        return jsonify({

            "success":
                False,

            "error":
                "Food search temporarily unavailable.",

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

        "recognition_mode":
            "open-ended",

        "primary_model":
            PRIMARY_MODEL,

        "fallback_model":
            FALLBACK_MODEL,

        "gemini_configured":
            bool(
                GEMINI_API_KEY
            ),

        "telegram_configured":
            telegram_configured(),

        "search_enabled":
            True,

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

        "features": [

            "Camera food recognition",

            "Food search",

            "Food recommendations",

            "Telegram scan notifications"

        ]

    }), 200


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "======================================"
    )

    print(
        "              FOODAI"
    )

    print(
        "AI Food Recognition"
    )

    print(
        "& Recommendation Agent"
    )

    print(
        "======================================"
    )

    print()

    print(
        "Primary:",
        PRIMARY_MODEL
    )

    print(
        "Fallback:",
        FALLBACK_MODEL
    )

    print(
        "Search:",
        "ENABLED"
    )

    print(
        "Telegram:",
        (
            "CONFIGURED"
            if telegram_configured()
            else
            "NOT CONFIGURED"
        )
    )

    print()


    app.run(

        host=
            "0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),

        debug=False

    )