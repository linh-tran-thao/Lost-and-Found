# =======================
# LOST & FOUND INTAKE SYSTEM
# =======================

import os
import json
import re
from datetime import datetime, timezone

import pandas as pd
import sqlite3
from PIL import Image

import streamlit as st
from google import genai
from google.genai import types

# LangChain vector store + embeddings
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings


# ==============================================================
# 1. SECRETS + CONNECTION HELPERS
# ==============================================================

@st.cache_resource
def get_gemini_client():
    """Create a Gemini client from secrets."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not set in secrets.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None


@st.cache_resource
def get_pg_sql_connection():
    """
    Streamlit SQL connection using [connections.postgresql] from secrets.toml.
    This is the recommended way for Neon + Streamlit.
    """
    try:
        conn = st.connection("postgresql", type="sql")
        return conn
    except Exception as e:
        st.error(f"Error creating PostgreSQL connection (st.connection): {e}")
        return None


@st.cache_resource
def get_vector_store():
    """
    Build a PGVector vector store using the same Streamlit SQL connection.
    Requires OPENAI_API_KEY in secrets and pgvector enabled in the DB.
    """
    openai_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_key:
        st.warning("OPENAI_API_KEY not found in secrets. Vector search disabled.")
        return None

    # For langchain_openai
    os.environ["OPENAI_API_KEY"] = openai_key

    sql_conn = get_pg_sql_connection()
    if sql_conn is None:
        st.warning("PostgreSQL connection not available. Vector search disabled.")
        return None

    # Streamlit sql connection uses SQLAlchemy under the hood.
    # We can reuse its connection URL for PGVector.
    connection_url = str(sql_conn._instance.engine.url)

    embeddings = OpenAIEmbeddings()

    try:
        store = PGVector(
            collection_name="lostandfound",     # use your existing collection name
            connection_string=connection_url,
            embedding_function=embeddings,
        )
        return store
    except Exception as e:
        st.error(f"Error initializing PGVector: {e}")
        return None


gemini_client = get_gemini_client()
vector_store = get_vector_store()  # may be None, app will degrade gracefully


# ==============================================================
# 2. PROMPTS
# ==============================================================

GENERATOR_SYSTEM_PROMPT = """
Role:
You are a Lost & Found intake operator for a public transit system. Your job is to gather accurate factual information
about a found item, refine the description interactively with the user, and output a single final structured record.

Behavior Rules:
1. Input Handling
The user may provide either an image or a short text description.
If an image is provided, describe visible traits such as color, material, type, size, markings, and notable features.
If text is provided, restate and cleanly summarize it in factual language.
Do not wait for confirmation before giving the first description.

2. Clarification
Ask targeted concise follow up questions to collect identifying details such as brand, condition,
writing, contents, location (station), and time found.
If the user provides a station name (for example “Times Sq”, “Queensboro Plaza”), try to identify the corresponding subway line or lines.
If multiple lines serve the station, you can mention all of them. If the station name has four or more lines, record only the station name.
If the station is unclear or unknown, set Subway Location to null.
Stop asking questions once the description is clear and specific enough.
Do not include questions or notes in the final output.

3. Finalization
When you have enough detail, output only this structured record:

Subway Location: <station or null>
Color: <dominant or user provided colors or null>
Item Category: <free text category such as Bags and Accessories, Electronics, Clothing or null>
Item Type: <free text item type such as Backpack, Phone, Jacket or null>
Description: <concise free text summary combining all verified details>
"""

USER_SIDE_GENERATOR_PROMPT = """
You are a helpful assistant for riders reporting lost items on a subway system.

Input:
The user may provide an image or a short text description of the lost item.
If an image is provided, describe what you see, including color, material, size, shape, and any markings.
If text is provided, restate the description in clean factual language.

Clarification:
Then ask two to four short follow up questions to collect details such as:
color if unclear, brand or logo, contents if it is a bag, any writing, where it was lost,
and approximate time.

When you have enough information, output only this structured record:

Subway Location: <station name or null>
Color: <color or colors or null>
Item Category: <category or null>
Item Type: <type or null>
Description: <concise factual summary>

Do not include your questions or reasoning in the final structured record.
"""

STANDARDIZER_PROMPT = """
You are the Lost and Found Data Standardizer for a public transit system.
You receive structured text from another model describing an item.
Your task is to map free text fields to standardized tag values and produce a clean JSON record.

Tag Source:
All valid standardized values are in the provided Tags Excel reference summary.
Use only those lists to choose values.

Subway Location: compare only with Subway Location tags.
Color: compare only with Color tags.
Item Category: compare only with Item Category tags.
Item Type: compare only with Item Type tags.

If no good match exists return "null" (or an empty list for list fields).

Input format:

Subway Location: <value or null>
Color: <value or null>
Item Category: <value or null>
Item Type: <value or null>
Description: <free text description>

Output (JSON only):

{
  "subway_location": ["<line or station>", "<line or station>"],
  "color": ["<color1>", "<color2>"],
  "item_category": "<standardized category or null>",
  "item_type": ["<type1>", "<type2>"],
  "description": "<clean description>",
  "time": "<ISO 8601 UTC timestamp>"
}
"""


# ==============================================================
# 3. TAG LOADING + STANDARDIZATION HELPERS
# ==============================================================

@st.cache_data
def load_tag_data():
    """Read Tags.xlsx and prepare tag lists."""
    try:
        df = pd.read_excel("Tags.xlsx")
        return {
            "df": df,
            "locations": sorted(set(df["Subway Location"].dropna().astype(str))),
            "colors": sorted(set(df["Color"].dropna().astype(str))),
            "categories": sorted(set(df["Item Category"].dropna().astype(str))),
            "item_types": sorted(set(df["Item Type"].dropna().astype(str))),
        }
    except Exception as e:
        st.error(f"Error loading tag data (Tags.xlsx): {e}")
        return None


def extract_field(text: str, field: str) -> str:
    """Extract 'Field: value' from structured block."""
    match = re.search(rf"{field}:\s*(.*)", text)
    return match.group(1).strip() if match else "null"


def is_structured_record(message: str) -> bool:
    """Detect if message looks like final 'Subway Location: ...' record."""
    return message.strip().startswith("Subway Location:")


def standardize_description(text: str, tags: dict) -> dict:
    """Send structured text to Gemini to map to tag lists and return JSON."""
    if gemini_client is None:
        st.error("Gemini client not available.")
        return {}

    tags_summary = (
        "\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )

    full_prompt = f"{STANDARDIZER_PROMPT}\n\nStructured input:\n{text}\n{tags_summary}"

    try:
        model = gemini_client.models.get("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
    except Exception as e:
        st.error(f"Error calling Gemini for standardization: {e}")
        return {}

    try:
        cleaned = response.text.strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        json_text = cleaned[json_start:json_end]
        data = json.loads(json_text)

        if "time" not in data or not data["time"]:
            data["time"] = datetime.now(timezone.utc).isoformat()

        # Normalize list fields
        for key in ["subway_location", "color", "item_type"]:
            if key in data and isinstance(data[key], str):
                data[key] = [data[key]]
            elif key not in data:
                data[key] = []

        if "item_category" not in data:
            data["item_category"] = "null"

        if "description" not in data:
            data["description"] = extract_field(text, "Description")

        return data
    except Exception:
        st.error("Could not parse standardizer output as JSON. Raw output:")
        st.text(response.text)
        return {}


# ==============================================================
# 4. LOCAL SQLITE FOR SIMPLE LOGGING (optional)
# ==============================================================

def init_sqlite():
    with sqlite3.connect("lost_and_found.db") as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS found_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caption TEXT,
                location TEXT,
                contact TEXT,
                image_path TEXT,
                json_data TEXT
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS lost_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,
                contact TEXT,
                email TEXT,
                json_data TEXT
            )
        """
        )
        conn.commit()


def add_found_item(caption, location, contact, image_path, json_data_string):
    with sqlite3.connect("lost_and_found.db") as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO found_items (caption, location, contact, image_path, json_data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (caption, location, contact, image_path, json_data_string),
        )
        conn.commit()


def add_lost_item(description, contact, email, json_data_string):
    with sqlite3.connect("lost_and_found.db") as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO lost_items (description, contact, email, json_data)
            VALUES (?, ?, ?, ?)
            """,
            (description, contact, email, json_data_string),
        )
        conn.commit()


def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# ==============================================================
# 5. STREAMLIT PAGE SETUP
# ==============================================================

st.set_page_config(page_title="Lost & Found Intake", page_icon="🧳", layout="wide")

init_sqlite()
tag_data = load_tag_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Found Item (Operator)", "Report Lost Item (User)"],
)


# ==============================================================
# 6. OPERATOR SIDE
# ==============================================================

if page == "Upload Found Item (Operator)":
    st.title("Operator View: Upload Found Item")

    if not tag_data:
        st.stop()

    if gemini_client is None:
        st.stop()

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model="gemini-1.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT
            ),
        )
        st.session_state.operator_msgs = []

    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.operator_msgs:
        st.markdown("Start by uploading an image or giving a short description.")

        c1, c2 = st.columns(2)
        with c1:
            uploaded_image = st.file_uploader(
                "Image of the found item (optional)",
                type=["jpg", "jpeg", "png"],
                key="operator_image",
            )
        with c2:
            initial_text = st.text_input(
                "Short description",
                placeholder="For example black backpack with a NASA patch",
                key="operator_text",
            )

        if st.button("Start Intake"):
            if not uploaded_image and not initial_text:
                st.error("Please upload an image or enter a short description.")
            else:
                parts = []
                content = ""
                if uploaded_image:
                    img = Image.open(uploaded_image).convert("RGB")
                    st.image(img, width=200)
                    parts.append(types.Part.from_image(img))
                    content += "Image uploaded.\n"
                if initial_text:
                    parts.append(types.Part.from_text(initial_text))
                    content += initial_text

                st.session_state.operator_msgs.append(
                    {"role": "user", "content": content}
                )
                with st.spinner("Analyzing item"):
                    response = st.session_state.operator_chat.send_message(parts)
                st.session_state.operator_msgs.append(
                    {"role": "model", "content": response.text}
                )
                st.experimental_rerun()

    op_input = st.chat_input("Add more details or say done when ready")
    if op_input:
        st.session_state.operator_msgs.append({"role": "user", "content": op_input})
        with st.spinner("Processing"):
            response = st.session_state.operator_chat.send_message(op_input)
        st.session_state.operator_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.experimental_rerun()

    # Final record detection
    if st.session_state.operator_msgs and is_structured_record(
        st.session_state.operator_msgs[-1]["content"]
    ):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        st.subheader("Final structured description")
        st.code(structured_text)

        final_json = standardize_description(structured_text, tag_data)
        if final_json:
            st.success("Standardized JSON")
            st.json(final_json)

            contact = st.text_input("Operator contact / badge ID")
            if st.button("Save Found Item"):
                loc_value = (
                    final_json["subway_location"][0]
                    if final_json.get("subway_location")
                    else ""
                )
                add_found_item(
                    final_json.get("description", ""),
                    loc_value,
                    contact,
                    "",
                    json.dumps(final_json),
                )
                st.success("Found item saved to SQLite log.")


# ==============================================================
# 7. USER SIDE + VECTOR MATCHING
# ==============================================================

if page == "Report Lost Item (User)":
    st.title("User View: Report a Lost Item")

    if not tag_data:
        st.stop()

    if gemini_client is None:
        st.stop()

    st.markdown("You can give quick info with dropdowns, then refine with chat.")

    with st.expander("Optional quick info"):
        c1, c2, c3 = st.columns(3)
        with c1:
            location_choice = st.selectbox(
                "Subway station (optional)", [""] + tag_data["locations"]
            )
        with c2:
            category_choice = st.selectbox(
                "Item category (optional)", [""] + tag_data["categories"]
            )
        with c3:
            type_choice = st.selectbox(
                "Item type (optional)", [""] + tag_data["item_types"]
            )

    st.subheader("Describe or show your lost item")
    ci, ct = st.columns(2)
    with ci:
        uploaded_image = st.file_uploader(
            "Image of lost item (optional)",
            type=["jpg", "jpeg", "png"],
            key="user_image",
        )
    with ct:
        initial_text = st.text_input(
            "Short description",
            placeholder="For example blue iPhone with cracked screen",
            key="user_text",
        )

    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.chats.create(
            model="gemini-1.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=USER_SIDE_GENERATOR_PROMPT
            ),
        )
        st.session_state.user_msgs = []

    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.user_msgs and st.button("Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
        else:
            parts = []
            content = ""
            if uploaded_image:
                img = Image.open(uploaded_image).convert("RGB")
                st.image(img, width=250)
                parts.append(types.Part.from_image(img))
                content += "Here is an image of my lost item.\n"
            if initial_text:
                parts.append(types.Part.from_text(initial_text))
                content += initial_text

            st.session_state.user_msgs.append({"role": "user", "content": content})
            with st.spinner("Analyzing"):
                response = st.session_state.user_chat.send_message(parts)
            st.session_state.user_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.experimental_rerun()

    user_input = st.chat_input("Add more details or say done when ready")
    if user_input:
        st.session_state.user_msgs.append({"role": "user", "content": user_input})
        with st.spinner("Thinking"):
            response = st.session_state.user_chat.send_message(user_input)
        st.session_state.user_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.experimental_rerun()

    if st.session_state.user_msgs and is_structured_record(
        st.session_state.user_msgs[-1]["content"]
    ):
        structured_text = st.session_state.user_msgs[-1]["content"]

        merged_text = f"""
Subway Location: {location_choice or extract_field(structured_text, 'Subway Location')}
Color: {extract_field(structured_text, 'Color')}
Item Category: {category_choice or extract_field(structured_text, 'Item Category')}
Item Type: {type_choice or extract_field(structured_text, 'Item Type')}
Description: {extract_field(structured_text, 'Description')}
        """

        st.subheader("Final merged record before standardization")
        st.code(merged_text)

        final_json = standardize_description(merged_text, tag_data)
        if final_json:
            st.success("Standardized record")
            st.json(final_json)

            # === Vector search against found items in Postgres (PGVector) ===
            if vector_store is not None:
                st.subheader("Possible matches in Lost & Found database")
                query_text = final_json.get("description", "")
                try:
                    results = vector_store.similarity_search(query_text, k=3)
                    for i, doc in enumerate(results, start=1):
                        st.markdown(f"**Match {i}** (score not shown, check server side)")
                        st.write(doc.page_content)
                        st.write(doc.metadata)
                except Exception as e:
                    st.error(f"Error during vector search: {e}")
            else:
                st.info("Vector search is disabled (missing Postgres or OpenAI config).")

            st.subheader("Contact information")
            contact = st.text_input("Phone number, ten digits")
            email = st.text_input("Email address")

            if st.button("Submit Lost Item Report"):
                if not validate_phone(contact):
                    st.error("Please enter a 10 digit phone number without spaces.")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    add_lost_item(
                        final_json.get("description", ""),
                        contact,
                        email,
                        json.dumps(final_json),
                    )
                    st.success("Lost item report submitted.")






