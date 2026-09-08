import os
import io
import json
import re
import base64
import html
import threading
import urllib.parse
import urllib.request
import requests

from datetime import datetime, timezone

from PIL import Image

from flask import (
    Flask,
    request,
    jsonify,
    make_response
)

from flask_cors import CORS
from groq import Groq


# ============================================================
# FOODAI CONFIG
# ============================================================

PROJECT_NAME = (
    "AI Food Recognition & Recommendation Agent"
)

PRIMARY_MODEL = "qwen/qwen3.6-27b"

GITHUB_ORIGIN = (
    "https://vickygowda1745.github.io"
)

MAX_IMAGE_SIZE = (
    6 * 1024 * 1024
)

WIKIMEDIA_API = (
    "https://commons.wikimedia.org/w/api.php"
)

WIKIMEDIA_USER_AGENT = (
    "FoodAI-College-Project/1.0 "
    "(food recognition and recommendation app)"
)


# Small in-memory image cache.
# Avoids asking Wikimedia repeatedly
# for the same food during one server session.

IMAGE_CACHE = {}


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_IMAGE_SIZE


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
                "X-FoodAI-Session",
                "X-FoodAI-Latitude",
                "X-FoodAI-Longitude"
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
        ] = (
            "Content-Type, "
            "X-FoodAI-Session, "
            "X-FoodAI-Latitude, "
            "X-FoodAI-Longitude"
        )

    return response


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

groq_client = None


if GROQ_API_KEY:

    try:

        groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=15.0
        )

        print(
            "Groq connected"
        )

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

        payload = (
            urllib.parse.urlencode(
                {
                    "chat_id":
                        TELEGRAM_CHAT_ID,

                    "text":
                        text
                }
            )
            .encode(
                "utf-8"
            )
        )

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

        print(
            "Telegram error:",
            error
        )


# ============================================================
# DEVICE / SESSION
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
        f"{location_message_lines()}"
        f"Time: {context['time']}\n"
        f"Session: {context['session_id']}\n"
        f"Device/Browser: "
        f"{context['user_agent']}"
    )

    threading.Thread(
        target=
            send_telegram_message,

        args=(
            message,
        ),

        daemon=True
    ).start()


def notify_search(
    query,
    food_name,
    cuisine,
    engine,
    context
):

    message = (
        "🔎 FOODAI — NEW SEARCH\n\n"
        f"Search: {query}\n"
        f"Food: {food_name}\n"
        f"Cuisine: {cuisine}\n"
        f"Engine: {engine}\n"
        f"{location_message_lines()}"
        f"Time: {context['time']}\n"
        f"Session: {context['session_id']}\n"
        f"Device/Browser: "
        f"{context['user_agent']}"
    )

    threading.Thread(
        target=send_telegram_message,
        args=(message,),
        daemon=True
    ).start()


# ============================================================
# LOCATION + NEARBY RESTAURANTS
# ============================================================

OVERPASS_API = (
    "https://overpass-api.de/api/interpreter"
)

NOMINATIM_REVERSE_API = (
    "https://nominatim.openstreetmap.org/reverse"
)


def safe_coordinate(
    value,
    minimum,
    maximum
):

    try:

        number = float(value)

        if (
            number < minimum
            or
            number > maximum
        ):
            return None

        return number

    except (
        TypeError,
        ValueError
    ):
        return None


def get_request_location():

    latitude = safe_coordinate(
        request.headers.get(
            "X-FoodAI-Latitude"
        ),
        -90,
        90
    )

    longitude = safe_coordinate(
        request.headers.get(
            "X-FoodAI-Longitude"
        ),
        -180,
        180
    )

    if (
        latitude is None
        or
        longitude is None
    ):

        return None

    return {
        "latitude":
            latitude,

        "longitude":
            longitude
    }


def reverse_location(
    latitude,
    longitude
):

    try:

        response = requests.get(

            NOMINATIM_REVERSE_API,

            params={
                "format":
                    "jsonv2",

                "lat":
                    latitude,

                "lon":
                    longitude,

                "zoom":
                    16,

                "addressdetails":
                    1
            },

            headers={
                "User-Agent":
                    (
                        "FoodAI/1.0 "
                        "(Food Recognition "
                        "Student Project)"
                    )
            },

            timeout=8
        )

        response.raise_for_status()

        data = response.json()

        address = (
            data.get(
                "address"
            )
            or
            {}
        )

        parts = []

        for key in [
            "amenity",
            "building",
            "road",
            "suburb",
            "neighbourhood",
            "village",
            "town",
            "city",
            "county",
            "state"
        ]:

            value = address.get(
                key
            )

            if (
                value
                and
                value not in parts
            ):
                parts.append(
                    str(value)
                )

        if parts:

            return ", ".join(
                parts[:6]
            )

        return (
            data.get(
                "display_name"
            )
            or
            "Location shared"
        )

    except Exception as error:

        print(
            "REVERSE LOCATION ERROR:",
            error
        )

        return "Location shared"


def location_message_lines():

    location = (
        get_request_location()
    )

    if not location:

        return (
            "\n📍 Location: "
            "Not shared\n"
        )

    latitude = (
        location[
            "latitude"
        ]
    )

    longitude = (
        location[
            "longitude"
        ]
    )

    place = reverse_location(
        latitude,
        longitude
    )

    return (
        "\n📍 Location: "
        f"{place}\n"
        f"Latitude: "
        f"{latitude:.6f}\n"
        f"Longitude: "
        f"{longitude:.6f}\n"
    )


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    from math import (
        radians,
        sin,
        cos,
        sqrt,
        atan2
    )

    earth_radius = 6371.0

    dlat = radians(
        lat2 - lat1
    )

    dlon = radians(
        lon2 - lon1
    )

    a = (
        sin(
            dlat / 2
        ) ** 2
        +
        cos(
            radians(lat1)
        )
        *
        cos(
            radians(lat2)
        )
        *
        sin(
            dlon / 2
        ) ** 2
    )

    c = (
        2
        *
        atan2(
            sqrt(a),
            sqrt(1 - a)
        )
    )

    return (
        earth_radius * c
    )


def build_restaurant_address(
    tags
):

    parts = []

    house_number = (
        tags.get(
            "addr:housenumber"
        )
        or
        ""
    )

    street = (
        tags.get(
            "addr:street"
        )
        or
        ""
    )

    if (
        house_number
        or
        street
    ):

        street_line = (
            house_number +
            " " +
            street
        ).strip()

        if street_line:
            parts.append(
                street_line
            )

    for key in [
        "addr:suburb",
        "addr:city",
        "addr:district"
    ]:

        value = tags.get(
            key
        )

        if (
            value
            and
            value not in parts
        ):
            parts.append(
                value
            )

    if not parts:

        return (
            tags.get(
                "addr:full"
            )
            or
            "Address unavailable"
        )

    return ", ".join(
        parts
    )


def get_nearby_restaurants(
    latitude,
    longitude,
    radius=5000
):

    query = f"""
[out:json][timeout:20];
(
  node["amenity"="restaurant"]
    (around:{radius},{latitude},{longitude});
  way["amenity"="restaurant"]
    (around:{radius},{latitude},{longitude});
  relation["amenity"="restaurant"]
    (around:{radius},{latitude},{longitude});

  node["amenity"="fast_food"]
    (around:{radius},{latitude},{longitude});
  way["amenity"="fast_food"]
    (around:{radius},{latitude},{longitude});
  relation["amenity"="fast_food"]
    (around:{radius},{latitude},{longitude});
);
out center tags;
"""

    response = requests.post(

        OVERPASS_API,

        data={
            "data":
                query
        },

        headers={
            "User-Agent":
                (
                    "FoodAI/1.0 "
                    "(Food Recognition "
                    "Student Project)"
                )
        },

        timeout=25
    )

    response.raise_for_status()

    data = response.json()

    restaurants = []

    seen = set()

    for element in (
        data.get(
            "elements",
            []
        )
    ):

        tags = (
            element.get(
                "tags"
            )
            or
            {}
        )

        name = (
            tags.get(
                "name"
            )
            or
            tags.get(
                "brand"
            )
        )

        if not name:
            continue

        lat = element.get(
            "lat"
        )

        lon = element.get(
            "lon"
        )

        if (
            lat is None
            or
            lon is None
        ):

            center = (
                element.get(
                    "center"
                )
                or
                {}
            )

            lat = center.get(
                "lat"
            )

            lon = center.get(
                "lon"
            )

        if (
            lat is None
            or
            lon is None
        ):
            continue

        try:

            lat = float(lat)
            lon = float(lon)

        except (
            TypeError,
            ValueError
        ):
            continue

        unique_key = (
            name.lower(),
            round(lat, 5),
            round(lon, 5)
        )

        if unique_key in seen:
            continue

        seen.add(
            unique_key
        )

        distance = haversine_km(
            latitude,
            longitude,
            lat,
            lon
        )

        restaurants.append({
            "name":
                name,

            "lat":
                lat,

            "lon":
                lon,

            "distance_km":
                round(
                    distance,
                    2
                ),

            "address":
                build_restaurant_address(
                    tags
                ),

            "cuisine":
                tags.get(
                    "cuisine"
                )
                or
                "",

            "source":
                "OpenStreetMap"
        })

    restaurants.sort(
        key=lambda item:
            item[
                "distance_km"
            ]
    )

    return restaurants[:20]


@app.route(
    "/nearby-restaurants",
    methods=["GET"]
)
def nearby_restaurants():

    try:

        latitude = safe_coordinate(
            request.args.get(
                "lat"
            ),
            -90,
            90
        )

        longitude = safe_coordinate(
            request.args.get(
                "lon"
            ),
            -180,
            180
        )

        if (
            latitude is None
            or
            longitude is None
        ):

            return jsonify({
                "success":
                    False,

                "error":
                    (
                        "Valid latitude "
                        "and longitude "
                        "are required."
                    )
            }), 400

        food = (
            request.args.get(
                "food",
                ""
            )
            or
            ""
        ).strip()

        restaurants = (
            get_nearby_restaurants(
                latitude,
                longitude
            )
        )

        return jsonify({
            "success":
                True,

            "food":
                food,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "count":
                len(
                    restaurants
                ),

            "restaurants":
                restaurants,

            "source":
                "OpenStreetMap"
        })

    except Exception as error:

        print(
            "NEARBY RESTAURANTS ERROR:",
            error
        )

        return jsonify({
            "success":
                False,

            "error":
                (
                    "Nearby restaurant "
                    "search is temporarily "
                    "unavailable."
                )
        }), 500



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

    mime_type = (
        get_image_mime_type(
            image_bytes
        )
    )

    encoded = (
        base64.b64encode(
            image_bytes
        )
        .decode(
            "utf-8"
        )
    )

    return (
        f"data:{mime_type};"
        f"base64,{encoded}"
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
# TEXT CLEANER
# ============================================================

def clean_html_text(value):

    if not value:
        return ""

    value = html.unescape(
        str(value)
    )

    value = re.sub(
        r"<[^>]+>",
        "",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


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
# WIKIMEDIA FOOD IMAGE
# ============================================================

def get_wikimedia_food_image(
    food_name
):

    cache_key = (
        food_name
        .strip()
        .lower()
    )

    if cache_key in IMAGE_CACHE:

        return IMAGE_CACHE[
            cache_key
        ]

    result = {
        "image_url":
            "",

        "image_page_url":
            "",

        "image_source":
            "Wikimedia Commons",

        "image_author":
            "",

        "image_license":
            "",

        "image_license_url":
            ""
    }

    try:

        params = {
            "action":
                "query",

            "format":
                "json",

            "generator":
                "search",

            "gsrsearch":
                f"{food_name} food",

            "gsrnamespace":
                "6",

            "gsrlimit":
                "8",

            "prop":
                "imageinfo",

            "iiprop":
                (
                    "url|"
                    "extmetadata"
                ),

            "iiurlwidth":
                "900"
        }

        url = (
            WIKIMEDIA_API
            + "?"
            + urllib.parse.urlencode(
                params
            )
        )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                    WIKIMEDIA_USER_AGENT,

                "Accept":
                    "application/json"
            }
        )

        response = (
            urllib.request.urlopen(
                req,
                timeout=7
            )
            .read()
        )

        data = json.loads(
            response.decode(
                "utf-8"
            )
        )

        pages = (
            data
            .get(
                "query",
                {}
            )
            .get(
                "pages",
                {}
            )
        )

        candidates = []

        for page in pages.values():

            title = str(
                page.get(
                    "title",
                    ""
                )
            )

            imageinfo = (
                page.get(
                    "imageinfo",
                    []
                )
            )

            if not imageinfo:
                continue

            info = imageinfo[0]

            image_url = (
                info.get(
                    "thumburl"
                )
                or
                info.get(
                    "url"
                )
                or
                ""
            )

            if not image_url:
                continue

            lower_url = (
                image_url.lower()
            )

            # Avoid SVG/PDF/etc.
            # We want normal photos.

            if not any(
                extension
                in lower_url

                for extension in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                ]
            ):

                continue

            metadata = (
                info.get(
                    "extmetadata",
                    {}
                )
            )

            description = (
                metadata
                .get(
                    "ImageDescription",
                    {}
                )
                .get(
                    "value",
                    ""
                )
            )

            description = (
                clean_html_text(
                    description
                )
            )

            score_text = (
                title
                + " "
                + description
            ).lower()

            food_words = [
                word
                for word
                in food_name
                .lower()
                .split()
                if len(word) >= 3
            ]

            score = sum(
                1
                for word
                in food_words
                if word in score_text
            )

            candidates.append(
                (
                    score,
                    page,
                    info,
                    metadata
                )
            )

        if not candidates:

            IMAGE_CACHE[
                cache_key
            ] = result

            return result

        candidates.sort(
            key=lambda item:
                item[0],
            reverse=True
        )

        (
            _,
            selected_page,
            selected_info,
            metadata
        ) = candidates[0]

        author = (
            metadata
            .get(
                "Artist",
                {}
            )
            .get(
                "value",
                ""
            )
        )

        author = clean_html_text(
            author
        )

        license_name = (
            metadata
            .get(
                "LicenseShortName",
                {}
            )
            .get(
                "value",
                ""
            )
        )

        license_name = (
            clean_html_text(
                license_name
            )
        )

        license_url = (
            metadata
            .get(
                "LicenseUrl",
                {}
            )
            .get(
                "value",
                ""
            )
        )

        page_title = (
            selected_page.get(
                "title",
                ""
            )
        )

        page_url = (
            "https://commons.wikimedia.org/wiki/"
            + urllib.parse.quote(
                page_title.replace(
                    " ",
                    "_"
                )
            )
        )

        result = {

            "image_url":
                (
                    selected_info.get(
                        "thumburl"
                    )
                    or
                    selected_info.get(
                        "url",
                        ""
                    )
                ),

            "image_page_url":
                page_url,

            "image_source":
                "Wikimedia Commons",

            "image_author":
                author,

            "image_license":
                license_name,

            "image_license_url":
                license_url
        }

        IMAGE_CACHE[
            cache_key
        ] = result

        return result

    except Exception as error:

        print(
            "Wikimedia image error:",
            error
        )

        IMAGE_CACHE[
            cache_key
        ] = result

        return result


# ============================================================
# PROMPTS
# ============================================================

RECOGNITION_PROMPT = """
You are FoodAI, an expert visual food recognition system.

Analyze the image carefully.

Identify the MAIN food or dish shown.

The image can contain ANY food from ANY cuisine in the world.
Do not use a fixed food list.

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

For Indian foods, use common Indian terminology.

Use Prawn instead of Shrimp for Indian dishes.

Do NOT answer only "Biryani" if the visible protein
or variety can reasonably be identified.

Return ONLY JSON:

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

Provide useful and concise information.

Return ONLY JSON:

{{
  "cuisine": "Cuisine or region",
  "description": "Short description",
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

Return JSON only.
"""


def create_search_prompt(
    query
):

    return f"""
You are FoodAI's food search engine.

The user searched for:

{query}

Interpret the intended dish even if the spelling is imperfect.

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
        groq_client
        .chat
        .completions
        .create(

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
        groq_client
        .chat
        .completions
        .create(

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

    food = (
        normalize_food_name(
            data.get(
                "food",
                "Unknown Food"
            )
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

    prompt = (
        create_details_prompt(
            food_name
        )
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

    prompt = (
        create_search_prompt(
            query
        )
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

    result["food"] = (
        normalize_food_name(
            result.get(
                "food",
                query
            )
        )
    )

    result[
        "engine"
    ] = PRIMARY_MODEL

    return result


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
# PREDICT ROUTE
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

            details = (
                generate_details(
                    image_bytes,
                    result[
                        "food"
                    ]
                )
            )

        except Exception as error:

            print(
                "DETAIL GENERATION ERROR:",
                repr(error)
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
            )
            .strip()
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

        search_context = (
            get_scan_context()
        )

        data = (
            request.get_json(
                silent=True
            )
            or
            {}
        )

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

        food_name = (
            result.get(
                "food",
                query
            )
        )

        # Find a matching Commons image.
        # Failure here does NOT break food search.

        image_data = (
            get_wikimedia_food_image(
                food_name
            )
        )

        notify_search(
            query,
            food_name,
            result.get(
                "cuisine",
                "Unknown"
            ),
            result.get(
                "engine",
                PRIMARY_MODEL
            ),
            search_context
        )

        return jsonify({

            "success":
                True,

            "query":
                query,

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
                ),

            "image_url":
                image_data.get(
                    "image_url",
                    ""
                ),

            "image_page_url":
                image_data.get(
                    "image_page_url",
                    ""
                ),

            "image_source":
                image_data.get(
                    "image_source",
                    ""
                ),

            "image_author":
                image_data.get(
                    "image_author",
                    ""
                ),

            "image_license":
                image_data.get(
                    "image_license",
                    ""
                ),

            "image_license_url":
                image_data.get(
                    "image_license_url",
                    ""
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
    methods=[
        "GET"
    ]
)
def health():

    return jsonify({

        "status":
            "online",

        "project":
            PROJECT_NAME,

        "provider":
            "Groq",

        "primary_model":
            PRIMARY_MODEL,

        "recognition_mode":
            "open-ended",

        "groq_configured":
            groq_configured(),

        "telegram_configured":
            telegram_configured(),

        "search_enabled":
            True,

        "wikimedia_images":
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
    methods=[
        "GET"
    ]
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

            "Wikimedia food images",

            "Food recommendations",

            "Telegram scan and search notifications",

            "Location and nearby restaurants coming next"

        ]

    }), 200


# ============================================================
# START SERVER
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
        "Wikimedia Images: ENABLED"
    )

    print(
        "Search: ENABLED"
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