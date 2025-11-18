# =======================
# LOST & FOUND INTAKE SYSTEM
# =======================

import streamlit as st
import sqlite3, json, re
from datetime import datetime, timezone
from PIL import Image
import pandas as pd
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from typing import List, Tuple, Dict

# Required imports for matching model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Single place to set model name
MODEL_NAME = "gemini-2.0-flash"

# --- Initialize Gemini client ---
try:
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Error initializing Gemini Client. Check st.secrets['GEMINI_API_KEY'].")
    st.stop()


# =======================
# SAFE GEMINI HELPERS
# =======================

def safe_send(chat, contents, context: str = ""):
    """
    Wrapper for chat.send_message that shows the real Gemini ClientError.
    'contents' can be a string or a list (text + images).
    """
    try:
        return chat.send_message(contents)
    except genai_errors.ClientError as e:
        label = context or "chat"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


def safe_generate(full_prompt: str, context: str = ""):
    """
    Wrapper for gemini_client.models.generate_content with clear error reporting.
    """
    try:
        return gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
        )
    except genai_errors.ClientError as e:
        label = context or "standardization"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


# =======================
# PROMPTS
# =======================

GENERATOR_SYSTEM_PROMPT = """
Role:
You are a Lost & Found intake operator for a public-transit system. Your job is to gather accurate, factual information
about a lost item, refine the description interactively with the user, and automatically pass the finalized record to a second GPT
for tag standardization.

Behavior Rules:
1. Input Handling
The user may provide either an image or a short text description.
If an image is provided, immediately describe visible traits (color, material, type, size, markings, notable features).
If text is provided, restate and cleanly summarize it in factual language. Do not wait for confirmation before generating the first description.
2. Clarification
After your initial description, ask targeted, concise follow-up questions to collect identifying details such as brand, condition,
writing, contents, location (station), and time lost.
If the user provides a station name (e.g., “Times Sq”, “Queensboro Plaza”), automatically identify the corresponding MTA subway line(s)
and record them in the Subway Location field.
Example: “Times Sq” → Lines 1, 2, 3, 7, N, Q, R, W, S.
If multiple lines serve the station, include all.
However, if the station name has 4 or more subway lines, record only the station name in the Subway Location field.
If the station name is unclear or not found, set Subway Location to null.
As the user answers, update and refine your internal description dynamically.
Stop asking questions once enough information has been gathered for a clear and specific description.
Do not include the questions or intermediate notes in the final output.
3. Finalization
When the user says they are finished, or you have enough detail, generate only the final structured record in this format:
Subway Location: <observed or user-provided line, or null>
Color: <dominant or user-provided color(s), or null>
Item Category: <free-text category, e.g., Bags & Accessories, Electronics, Clothing, or null>
Item Type: <free-text item type, e.g., Backpack, Phone, Jacket, or null>
Description: <concise free-text summary combining all verified details>
"""

USER_SIDE_GENERATOR_PROMPT = """
You are a helpful assistant helping subway riders report lost items.

Input:
The user may provide either an image or a short text description of their item.
If an image is provided, begin by describing what you see — include visible traits such as color, material, size, shape, and any markings or distinctive details.
If text is provided, restate their message cleanly in factual language.

Clarification:
Then, ask 2–4 concise follow-up questions to collect identifying details such as:
- color (if not already clear from the image),
- brand or logo,
- contents (if a bag or container),
- any markings or writing,
- where it was lost (station name),
- and approximate time.

When you have enough details, output ONLY the structured record in this format:

Subway Location: <station name or null>
Color: <color(s) or null>
Item Category: <category or null>
Item Type: <type or null>
Description: <concise factual summary combining all verified details>

Guidelines:
- Keep your tone concise and factual.
- Do not include your reasoning or notes.
- Do not output questions or conversation history in the final structured record.
"""

STANDARDIZER_PROMPT = """
You are the Lost & Found Data Standardizer for a public-transit system. You receive structured text from another model describing a lost item. Your job is to map free-text fields to standardized tag values and produce a clean JSON record ready for database storage.

Data Source:
All valid standardized values are stored in the Tags Excel reference file uploaded.
This file is the only source of truth for all mappings.
The Tags Excel contains separate tabs or columns for the following standardized lists:

Subway Location → All valid subway lines and station names
Item Category → All valid item category names
Item Type → All valid item type names
Color → All valid color names

When standardizing input text:

Always match each field only against its corresponding tag list:
subway_location → compare only with values in Subway Location
color → compare only with values in Color
item_category → compare only with values in Item Category
item_type → compare only with values in Item Type
Never mix across tag types.
Use exact or closest textual matches from the relevant tag column only.
If no valid match is found, return "null".
Output the standardized value exactly as it appears in the Excel file — no prefixes, suffixes, or formatting changes.

Behavior Rules:
1. Input Format:
You will receive input in this structure:
Subway Location: <value or null>
Color: <value or null>
Item Category: <value or null>
Item Type: <value or null>
Description: <free-text description>

2. Standardization:
Use the provided Tags Excel reference to ensure consistent value mapping.
Subway Location: Match to valid MTA lines or stations. If none or unclear, output "null".
Color: Match to standardized color names. If multiple colors appear, include all as an array.
Item Category: Map to a consistent category.
Item Type: Map to consistent type(s). If multiple types appear, include all as an array.
Description: Leave as free text but clean it up.
Time: Record the current system time (ISO 8601 UTC).

3. Output Format:
Produce only a JSON object (no explanations):
{
  "subway_location": ["<line1>", "<line2>"],
  "color": ["<color1>", "<color2>"],
  "item_category": "<standardized category or null>",
  "item_type": ["<standardized type>", "<standardized type>"],
  "description": "<clean final description>",
  "time": "<ISO 8601 UTC timestamp>"
}

4. Behavior Guidelines:
- Do not guess missing details.
- If uncertain, leave field as null.
- Ensure valid JSON output.
- Only output the JSON object, nothing else.
"""


# =======================
# DATA HELPERS
# =======================

@st.cache_data
def load_tag_data():
    """Loads standardized tag data from Tags.xlsx."""
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
        st.error(f"Error loading tag data: {e}. Ensure 'Tags.xlsx' is present.")
        return None


def extract_field(text: str, field: str) -> str:
    """Extracts a field value from the structured text output."""
    match = re.search(rf"{field}:\s*(.*)", text)
    return match.group(1).strip() if match else "null"


def standardize_description(text: str, tags: dict) -> dict:
    """Send structured text + tag data to Gemini for JSON standardization."""
    tags_summary = (
        f"\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )

    full_prompt = f"{STANDARDIZER_PROMPT}\n\nHere is the structured input to standardize:\n{text}\n{tags_summary}"

    response = safe_generate(full_prompt, context="standardize_description")

    try:
        cleaned = response.text.strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        json_text = cleaned[json_start:json_end]
        data = json.loads(json_text)

        # Ensure time is present
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
        st.error("Model output could not be parsed as JSON. Displaying raw output:")
        st.text(response.text)
        return {}


# =======================
# MATCHING MODEL
# =======================

@st.cache_data
def get_top_matches(
    item_list: List[Tuple[int, str, str]],
    lost_item_data: Dict[str, any],
) -> List[str]:
    """
    Computes cosine similarity between the lost item's 'description'
    and all pre-filtered found item descriptions, returning the top 3 image paths.

    item_list format: [(unique_id, image_path, description), ...]
    """

    query_description = lost_item_data.get("description", "")

    image_paths = [item[1] for item in item_list]
    item_descriptions = [item[2] for item in item_list]

    all_descriptions = [query_description] + item_descriptions

    vectorizer = TfidfVectorizer().fit(all_descriptions)
    tfidf_matrix = vectorizer.transform(all_descriptions)

    query_vector = tfidf_matrix[0:1]
    item_vectors = tfidf_matrix[1:]

    cosine_scores = cosine_similarity(query_vector, item_vectors).flatten()

    scored_matches = []
    for i, score in enumerate(cosine_scores):
        scored_matches.append((score, image_paths[i]))

    scored_matches.sort(key=lambda x: x[0], reverse=True)

    top_3_paths = [path for score, path in scored_matches[:3]]
    return top_3_paths


# =======================
# DATABASE HELPERS
# =======================

def init_db():
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subway_location TEXT,
            color TEXT,
            item_category TEXT,
            item_type TEXT,
            description TEXT,
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
    conn.close()


def add_found_item(json_data: dict, contact: str, image_path: str = ""):
    """Adds a found item record, extracting key fields from the JSON."""
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO found_items (
            subway_location, color, item_category, item_type,
            description, contact, image_path, json_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ", ".join(json_data.get("subway_location", [])),
            ", ".join(json_data.get("color", [])),
            json_data.get("item_category", ""),
            ", ".join(json_data.get("item_type", [])),
            json_data.get("description", ""),
            contact,
            image_path,
            json.dumps(json_data),
        ),
    )
    conn.commit()
    conn.close()


def add_lost_item(description, contact, email, json_data_string):
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO lost_items (description, contact, email, json_data) VALUES (?, ?, ?, ?)",
        (description, contact, email, json_data_string),
    )
    conn.commit()
    conn.close()


# =======================
# STREAMLIT UI
# =======================

st.set_page_config(page_title="Lost & Found Intake", page_icon="🧳", layout="wide")
init_db()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Upload Found Item (Operator)", "Report Lost Item (User)"],
)


# ===============================================================
# OPERATOR SIDE (CHAT INTAKE)
# ===============================================================

if page == "Upload Found Item (Operator)":
    st.title("Operator: Describe Found Item")

    tag_data = load_tag_data()
    if not tag_data:
        st.stop()

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
            ),
        )
        st.session_state.operator_msgs = []

    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial input
    if not st.session_state.operator_msgs:
        uploaded_image = st.file_uploader(
            "Upload an image of the found item (optional)",
            type=["jpg", "jpeg", "png"],
        )
        initial_text = st.text_input("Or briefly describe the item")

        if st.button("Start Intake"):
            if not uploaded_image and not initial_text:
                st.error("Please upload or describe the item.")
            else:
                parts = []
                message_content = ""

                # For now, we still send the image object to Gemini.
                # If your client complains, we could switch to text-only.
                if uploaded_image:
                    img = Image.open(uploaded_image).convert("RGB")
                    st.image(img, width=200)
                    parts.append(img)
                    message_content += "Image uploaded.\n"

                if initial_text:
                    parts.append(initial_text)
                    message_content += initial_text

                st.session_state.operator_msgs.append(
                    {"role": "user", "content": message_content}
                )
                with st.spinner("Analyzing..."):
                    response = safe_send(
                        st.session_state.operator_chat,
                        parts,
                        context="operator intake",
                    )
                st.session_state.operator_msgs.append(
                    {"role": "model", "content": response.text}
                )
                st.rerun()

    if operator_prompt := st.chat_input("Add more details..."):
        st.session_state.operator_msgs.append(
            {"role": "user", "content": operator_prompt}
        )
        with st.spinner("Processing..."):
            response = safe_send(
                st.session_state.operator_chat,
                operator_prompt,
                context="operator follow-up",
            )
        st.session_state.operator_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.rerun()

    # Detect final structured text
    final_json = {}
    if (
        st.session_state.operator_msgs
        and st.session_state.operator_msgs[-1]["content"].startswith(
            "Subway Location:"
        )
    ):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        final_json = standardize_description(structured_text, tag_data)
        st.success("Structured description generated:")
        st.json(final_json)

        image_path = ""  # TODO: if you save images, store the path here

        contact = st.text_input("Operator Contact/Badge ID")
        if st.button("Save Found Item"):
            add_found_item(final_json, contact, image_path)
            st.success("Saved successfully!")


# ===============================================================
# USER SIDE (HYBRID DROPDOWN + CHAT)
# ===============================================================

if page == "Report Lost Item (User)":
    st.title("Report Your Lost Item")

    tag_data = load_tag_data()
    if not tag_data:
        st.stop()

    # Quick info
    with st.expander("Quick Info (Optional)"):
        location = st.selectbox("Subway Station", [""] + tag_data["locations"])
        category = st.selectbox("Item Category", [""] + tag_data["categories"])
        item_type = st.selectbox("Item Type", [""] + tag_data["item_types"])

    st.subheader("Describe or Show Your Lost Item")
    uploaded_image = st.file_uploader(
        "Upload an image of your lost item (optional)",
        type=["jpg", "jpeg", "png"],
    )
    initial_text = st.text_input(
        "Or describe it briefly (e.g., 'a red leather backpack with gold zippers')"
    )

    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=USER_SIDE_GENERATOR_PROMPT,
            ),
        )
        st.session_state.user_msgs = []

    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Start the chat
    if not st.session_state.user_msgs and st.button("Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or provide a description.")
        else:
            parts = []
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=250)
                parts.append(image)
                message_text += "Here is an image of my lost item.\n"
            if initial_text:
                parts.append(initial_text)
                message_text += initial_text

            st.session_state.user_msgs.append(
                {"role": "user", "content": message_text}
            )
            with st.spinner("Analyzing..."):
                response = safe_send(
                    st.session_state.user_chat,
                    parts,
                    context="user initial report",
                )
            st.session_state.user_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.rerun()

    # Continue conversation
    if user_input := st.chat_input("Add more details..."):
        st.session_state.user_msgs.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Thinking..."):
            response = safe_send(
                st.session_state.user_chat,
                user_input,
                context="user follow-up",
            )
        st.session_state.user_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.rerun()

    # Final structured record detection
    final_json = {}
    if (
        st.session_state.user_msgs
        and st.session_state.user_msgs[-1]["content"].startswith(
            "Subway Location:"
        )
    ):
        structured_text = st.session_state.user_msgs[-1]["content"]

        merged_text = f"""
Subway Location: {location or extract_field(structured_text, 'Subway Location')}
Color: {extract_field(structured_text, 'Color')}
Item Category: {category or extract_field(structured_text, 'Item Category')}
Item Type: {item_type or extract_field(structured_text, 'Item Type')}
Description: {extract_field(structured_text, 'Description')}
        """

        final_json = standardize_description(merged_text, tag_data)
        st.success("Standardized Record:")
        st.json(final_json)

        contact = st.text_input("Contact Number (10 digits)")
        email = st.text_input("Email Address")

        if st.button("Submit Lost Item Report & Find Matches"):
            if not contact or not email:
                st.error("Please provide contact info.")
            else:
                st.info("Searching for potential matches...")

                colors = final_json["color"]
                category_std = final_json["item_category"]
                item_types_std = final_json["item_type"]
                subway_locations = final_json["subway_location"]

                conn = sqlite3.connect("lost_and_found.db")
                cursor = conn.cursor()

                like_clauses = []
                params = []

                if category_std:
                    like_clauses.append("item_category = ?")
                    params.append(category_std)

                if colors:
                    color_likes = [f"color LIKE ?" for _ in colors]
                    like_clauses.append(f"({' OR '.join(color_likes)})")
                    params.extend([f"%{c}%" for c in colors])

                if item_types_std:
                    type_likes = [f"item_type LIKE ?" for _ in item_types_std]
                    like_clauses.append(f"({' OR '.join(type_likes)})")
                    params.extend([f"%{t}%" for t in item_types_std])

                if subway_locations:
                    location_likes = [f"subway_location LIKE ?" for _ in subway_locations]
                    like_clauses.append(f"({' OR '.join(location_likes)})")
                    params.extend([f"%{l}%" for l in subway_locations])

                if like_clauses:
                    where_clause = " AND ".join(like_clauses)
                else:
                    where_clause = "1=1"  # no filters

                sql_query = f"""
                SELECT id, image_path, json_data
                FROM found_items
                WHERE {where_clause};
                """

                try:
                    cursor.execute(sql_query, tuple(params))
                    raw_results = cursor.fetchall()
                except sqlite3.OperationalError as e:
                    st.error(f"SQL Error during filtering: {e}")
                    raw_results = []

                results_for_matching = []
                for id_, image_path, json_data_str in raw_results:
                    try:
                        found_item_json = json.loads(json_data_str)
                        results_for_matching.append(
                            (id_, image_path, found_item_json.get("description", ""))
                        )
                    except json.JSONDecodeError:
                        continue

                conn.close()

                add_lost_item(
                    final_json["description"],
                    contact,
                    email,
                    json.dumps(final_json),
                )
                st.success("Lost item report submitted successfully!")

                if results_for_matching:
                    st.subheader(
                        "Potential Matches Found (Ranked by Description Similarity) 👇"
                    )
                    top_paths = get_top_matches(results_for_matching, final_json)

                    st.json(top_paths)
                    st.write("---")
                    st.markdown("**Top 3 Matches:**")
                    for path in top_paths:
                        st.markdown(f"* `{path}`")
                else:
                    st.info(
                        "No initial matches found in the database based on basic tags. "
                        "Your report has still been saved."
                    )
