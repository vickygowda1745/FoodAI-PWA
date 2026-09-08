import os
import io
import json
import re
import base64
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from PIL import Image

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from groq import Groq


# ============================================================
# FOODAI CONFIGURATION
# ============================================================

PROJECT_NAME = "AI Food Recognition & Recommendation Agent"

# Qwen Vision hosted by Groq.
# No Gemini. No OpenAI.
PRIMARY_MODEL = "qwen/qwen3.6-27b"

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
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = None


if GROQ_API_KEY:

    try:

        groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=15.0
        )

        print("Groq connected")

    except Exception as error:

        print(
            "Groq connection error:",
            error
        )

else:

    print(
        "WARNING: GROQ_API_KEY missing"
    )


def groq_configured():

    return bool(
        GROQ_API_KEY
        and
        groq_client
    )


# ============================================================
# TELEGRAM
# ============================================================

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


def send_telegram_message(text):

    if not telegram_configured():

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
        ).encode("utf-8")


        req = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )


        urllib.request.urlopen(
            req,
            timeout=5
        ).read()


        print(
            "Telegram notification sent"
        )


    except Exception as error:

        # Telegram must never break scanning.

        print(
            "Telegram error:",
            error
        )


# ============================================================
# SCAN CONTEXT
# ============================================================

def get_scan_context():

    user_agent = request.headers.get(
        "User-Agent",
        "Unknown device"
    )


    if len(user_agent) > 160:

        user_agent = (
            user_agent[:160]
            + "..."
        )


    session_id = request.headers.get(
        "X-FoodAI-Session",
        "Not supplied"
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


    threading.Thread(
        target=send_telegram_message,
        args=(message,),
        daemon=True
    ).start()


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

    if not image_bytes:

        return False


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


def image_to_data_url(
    image_bytes
):

    mime_type = get_image_mime_type(
        image_bytes
    )


    encoded = base64.b64encode(
        image_bytes
    ).decode(
        "utf-8"
    )


    return (
        f"data:{mime_type};base64,{encoded}"
    )


# ============================================================
# JSON HELPERS
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
# PROMPTS
# ============================================================

RECOGNITION_PROMPT = """
You are FoodAI, an expert visual food recognition system.

Analyze the image carefully.

Identify the MAIN food or dish shown.

The image can contain ANY food from ANY cuisine in the world.
Do not use a fixed list.

Identify the MOST SPECIFIC dish supported by visible evidence.

Pay special attention to:

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

For biryani, distinguish whenever visually possible:

Chicken Biryani
Mutton Biryani
Prawn Biryani
Egg Biryani
Vegetable Biryani

For Indian food, use common Indian terminology.

Use Prawn instead of Shrimp for Indian dishes.

Do NOT answer only "Biryani" if the visible protein
or variety can reasonably be identified.

Do not invent ingredients that are not visible.

Return ONLY this JSON object:

{
  "is_food": true,
  "food": "Specific Food Name",
  "confidence": 95
}

confidence must be an integer from 0 to 100.

If the image clearly does not contain food:

{
  "is_food": false,
  "food": "Not Food",
  "confidence": 0
}
"""


def create_details_prompt(
    food_name
):

    return f"""
You are FoodAI's food recommendation engine.

Food:
{food_name}

Provide useful, concise information.

Return ONLY JSON:

{{
  "cuisine": "Cuisine or region",
  "description": "Short description of the food",
  "visible_items": [
    "item 1",
    "item 2"
  ],
  "recommendation": "Best serving or pairing recommendation",
  "health_note": "Short general nutritional observation",
  "similar_food": "One similar dish"
}}

Do not provide medical advice.

Use Indian terminology for Indian foods.

Do not claim an ingredient is visible unless it is actually
known from the supplied dish/image context.

Return JSON only.
"""


def create_search_prompt(
    query
):

    return f"""
You are FoodAI's food search engine.

The user searched for:

{query}

Interpret the intended dish even if there are minor
spelling mistakes.

Return ONLY JSON:

{{
  "food": "Canonical food name",
  "cuisine": "Cuisine or region",
  "description": "Short description",
  "recommendation": "Best serving or pairing recommendation",
  "health_note": "Short general nutritional observation",
  "similar_food": "One similar dish"
}}

Use common Indian terminology for Indian dishes.

Return JSON only.
"""


# ============================================================
# GROQ IMAGE CALL
# ============================================================

def call_groq_image(
    prompt,
    image_bytes,
    max_tokens=180
):

    if not groq_configured():

        raise RuntimeError(
            "Groq is not configured"
        )


    data_url = image_to_data_url(
        image_bytes
    )


    completion = (
        groq_client.chat.completions.create(

            model=
                PRIMARY_MODEL,

            messages=[
                {
                    "role":
                        "user",

                    "content": [
                        {
                            "type":
                                "text",

                            "text":
                                prompt
                        },

                        {
                            "type":
                                "image_url",

                            "image_url": {
                                "url":
                                    data_url
                            }
                        }
                    ]
                }
            ],

            temperature=0.2,

            max_completion_tokens=
                max_tokens,

            response_format={
                "type":
                    "json_object"
            },

            reasoning_effort=
                "none",

            stream=False
        )
    )


    content = (
        completion
        .choices[0]
        .message
        .content
    )


    if not content:

        raise RuntimeError(
            "Groq returned an empty response"
        )


    return content


# ============================================================
# GROQ TEXT CALL
# ============================================================

def call_groq_text(
    prompt,
    max_tokens=350
):

    if not groq_configured():

        raise RuntimeError(
            "Groq is not configured"
        )


    completion = (
        groq_client.chat.completions.create(

            model=
                PRIMARY_MODEL,

            messages=[
                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ],

            temperature=0.2,

            max_completion_tokens=
                max_tokens,

            response_format={
                "type":
                    "json_object"
            },

            reasoning_effort=
                "none",

            stream=False
        )
    )


    content = (
        completion
        .choices[0]
        .message
        .content
    )


    if not content:

        raise RuntimeError(
            "Groq returned an empty response"
        )


    return content


# ============================================================
# RECOGNITION
# ============================================================

def identify_food(
    image_bytes
):

    response_text = (
        call_groq_image(

            RECOGNITION_PROMPT,

            image_bytes,

            max_tokens=120

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
            PRIMARY_MODEL
    }


# ============================================================
# DETAILS
# ============================================================

def generate_details(
    image_bytes,
    food_name
):

    prompt = create_details_prompt(
        food_name
    )


    response_text = (
        call_groq_image(

            prompt,

            image_bytes,

            max_tokens=350

        )
    )


    result = parse_json(
        response_text
    )


    result[
        "engine"
    ] = PRIMARY_MODEL


    return result


# ============================================================
# SEARCH
# ============================================================

def search_food_information(
    query
):

    prompt = create_search_prompt(
        query
    )


    response_text = (
        call_groq_text(

            prompt,

            max_tokens=350

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
    ] = PRIMARY_MODEL


    return result


# ============================================================
# IDENTIFY
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
            repr(error)
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
# PREDICT
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

            details = generate_details(

                image_bytes,

                result[
                    "food"
                ]

            )

        except Exception as detail_error:

            print(
                "DETAIL GENERATION ERROR:",
                repr(detail_error)
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
            repr(error)
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
# DETAILS
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


        result = generate_details(

            image_bytes,

            food_name

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
                ),

            "engine":
                result.get(
                    "engine",
                    PRIMARY_MODEL
                )

        }), 200


    except Exception as error:

        print(
            "DETAIL ERROR:",
            repr(error)
        )


        return jsonify({

            "success":
                False,

            "error":
                "Food recommendations failed.",

            "details":
                str(error)

        }), 503


# ============================================================
# SEARCH
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


        result = search_food_information(
            query
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
                    PRIMARY_MODEL
                )

        }), 200


    except Exception as error:

        print(
            "SEARCH ERROR:",
            repr(error)
        )


        return jsonify({

            "success":
                False,

            "error":
                "Food search failed.",

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
            PROJECT_NAME,

        "recognition_mode":
            "open-ended",

        "provider":
            "Groq",

        "primary_model":
            PRIMARY_MODEL,

        "groq_configured":
            groq_configured(),

        "telegram_configured":
            telegram_configured(),

        "search_enabled":
            True,

        "gemini_used":
            False,

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
            PROJECT_NAME,

        "status":
            "online",

        "provider":
            "Groq",

        "vision_model":
            PRIMARY_MODEL,

        "features": [

            "Camera food recognition",

            "Food search",

            "Food recommendations",

            "Telegram scan notifications",

            "Location and nearby restaurants coming next"

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
        "Provider: Groq"
    )

    print(
        "Vision model:",
        PRIMARY_MODEL
    )

    print(
        "Groq:",
        (
            "CONFIGURED"
            if groq_configured()
            else
            "NOT CONFIGURED"
        )
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

    print(
        "Gemini: REMOVED"
    )

    print(
        "OpenAI: REMOVED"
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