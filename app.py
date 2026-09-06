import os
import json

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    make_response,
)

from PIL import Image, UnidentifiedImageError

from google import genai
from google.genai import types


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# SETTINGS
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set")

client = None

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


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
# PWA MANIFEST
# ============================================================

@app.route("/manifest.json")
def manifest():

    response = make_response(
        send_from_directory(
            ".",
            "manifest.json"
        )
    )

    response.headers[
        "Content-Type"
    ] = "application/manifest+json"

    return response


# ============================================================
# SERVICE WORKER
# ============================================================

@app.route("/service-worker.js")
def service_worker():

    response = make_response(
        send_from_directory(
            ".",
            "service-worker.js"
        )
    )

    response.headers[
        "Content-Type"
    ] = "application/javascript"

    response.headers[
        "Cache-Control"
    ] = "no-cache"

    return response


# ============================================================
# ANALYZE FOOD
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error":
            "No image received"
        }), 400

    image_file = request.files["image"]

    if not image_file.filename:

        return jsonify({
            "error":
            "No image selected"
        }), 400

    if not allowed_file(
        image_file.filename
    ):

        return jsonify({
            "error":
            "Unsupported image type"
        }), 400


    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = image_file.read()

        # Verify Pillow can open image
        from io import BytesIO

        verify_image = Image.open(
            BytesIO(image_bytes)
        )

        verify_image.verify()

        if not client:

            return jsonify({
                "error":
                "GEMINI_API_KEY is not configured."
            }), 500


        # ----------------------------------------------------
        # GEMINI FOOD ANALYSIS
        # ----------------------------------------------------

        prompt = """
Analyze the food shown in this image.

Identify the main food dish as accurately as possible.

Then provide one short useful recommendation and approximate
nutrition information.

Return valid JSON only in exactly this format:

{
  "food": "food name",
  "recommendation": "short recommendation",
  "calories": "approximate kcal range",
  "protein": "approximate grams",
  "carbohydrates": "approximate grams",
  "fat": "approximate grams",
  "health_note": "short practical health note"
}

Rules:

1. Do not return markdown.
2. Do not return code fences.
3. Do not add headings.
4. Do not add text outside the JSON.
5. If uncertain, choose the most likely food but mention the
   uncertainty briefly in the recommendation.
6. Nutrition values must be clearly approximate.
"""


        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=(
                image_file.mimetype
                or
                "image/jpeg"
            )
        )


        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=[
                prompt,
                image_part
            ]
        )


        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        text = response.text.strip()

        if text.startswith("```"):

            text = (
                text
                .replace(
                    "```json",
                    ""
                )
                .replace(
                    "```",
                    ""
                )
                .strip()
            )


        result = json.loads(text)


        # ----------------------------------------------------
        # RETURN ONLY EXPECTED DATA
        # ----------------------------------------------------

        safe_result = {

            "food":
                str(
                    result.get(
                        "food",
                        "Unknown"
                    )
                ),

            "recommendation":
                str(
                    result.get(
                        "recommendation",
                        "No recommendation available."
                    )
                ),

            "calories":
                str(
                    result.get(
                        "calories",
                        "Unavailable"
                    )
                ),

            "protein":
                str(
                    result.get(
                        "protein",
                        "Unavailable"
                    )
                ),

            "carbohydrates":
                str(
                    result.get(
                        "carbohydrates",
                        "Unavailable"
                    )
                ),

            "fat":
                str(
                    result.get(
                        "fat",
                        "Unavailable"
                    )
                ),

            "health_note":
                str(
                    result.get(
                        "health_note",
                        "Nutrition values are approximate."
                    )
                )
        }


        return jsonify(
            safe_result
        )


    # ========================================================
    # INVALID IMAGE
    # ========================================================

    except UnidentifiedImageError:

        return jsonify({
            "error":
            "The uploaded file is not a valid image."
        }), 400


    # ========================================================
    # INVALID AI RESPONSE
    # ========================================================

    except json.JSONDecodeError:

        return jsonify({
            "error":
            "AI returned an invalid response. Please scan again."
        }), 500


    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as e:

        print(
            "Server error:",
            type(e).__name__,
            str(e)
        )

        return jsonify({
            "error":
            "Food analysis failed. Please try again."
        }), 500


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error":
        "Image is too large. Maximum size is 5 MB."
    }), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )