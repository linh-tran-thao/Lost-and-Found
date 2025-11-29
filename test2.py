# =======================
# LOST & FOUND INTAKE SYSTEM (Chroma Vector DB)
# =======================

import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------
# BASIC CONFIG + LIGHT THEME TWEAKS
# -----------------------

MODEL_NAME = "gemini-2.0-flash"

st.set_page_config(
    page_title="Lost & Found Intake",
    page_icon="🧳",
    layout="wide",
)

# Small CSS tweaks for nicer UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #555555;
        margin-bottom: 1.0rem;
    }
    .stat-card {
        padding: 0.75rem 0.9rem;
        border-radius: 0.6rem;
        border: 1px solid #e0e0e0;
        background-color: #fafafa;
        font-size: 0.9rem;
    }
    .match-card {
        padding: 0.9rem 1.0rem;
        border-radius: 0.7rem;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        margin-bottom: 0.8rem;
    }
    .match-title {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid #d0d0d0;
        margin-right: 0.3rem;
        margin-bottom: 0.2rem;
        background-color: #f7f7f7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------
# SECRETS / CLIENTS / VECTOR STORE
# -----------------------

@st.cache_resource
def get_secrets() -> Dict[str, Any]:
    """Load API keys and config from Streamlit secrets."""
    try:
        s = st.secrets.to_dict()
    except Exception:
        s = {}
    return {
        "raw": s,
        "gemini_key": s.get("GEMINI_API_KEY"),
        "openai_key": s.get("OPENAI_API_KEY"),
    }


secrets = get_secrets()


@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client."""
    if not secrets["gemini_key"]:
        st.error("GEMINI_API_KEY is not set in Streamlit secrets.")
        return None
    try:
        return genai.Client(api_key=secrets["gemini_key"])
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None


gemini_client = get_gemini_client()


@st.cache_resource
def get_vector_store():
    """
    Create or load a Chroma vector DB using OpenAI embeddings.
    This stores all *found* items with metadata.
    """
    if not secrets["openai_key"]:
        st.warning("OPENAI_API_KEY is not set; semantic matching will be disabled.")
        return None

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=secrets["openai_key"])
        vs = Chroma(
            collection_name="lost_and_found_items",
            embedding_function=embeddings,
            persist_directory="chroma_db",  # local folder for persistence
        )
        return vs
    except Exception as e:
        st.error(f"Error creating Chroma vector store: {e}")
        return None


vector_store = get_vector_store()


# -----------------------
# SAFE GEMINI HELPERS
# -----------------------

def safe_send(chat, message_content: str, context: str = ""):
    """Wrapper around chat.send_message with clear error reporting."""
    try:
        return chat.send_message(message_content)
    except genai_errors.ClientError as e:
        label = context or "chat"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


def safe_generate(full_prompt: str, context: str = ""):
    """Wrapper around generate_content with clear error reporting."""
    if gemini_client is None:
        st.error("Gemini client is not available.")
        st.stop()
    try:
        return gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
        )
    except genai_errors.ClientError as e:
        label = context or "generation"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


# -----------------------
# PROMPTS
# -----------------------

GENERATOR_SYSTEM_PROMPT = """
Role:
You are a Lost & Found intake operator for a public-transit system. Your job is to gather accurate factual information
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

Field rules:

Subway Location:
Compare only with the Subway Location tag list.
Color:
Compare only with the Color tag list.
Item Category:
Compare only with the Item Category tag list.
Item Type:
Compare only with the Item Type tag list.

Use exact or closest textual matches from the correct list only.
If no good match exists return "null" for that field.

Input format:

Subway Location: <value or null>
Color: <value or null>
Item Category: <value or null>
Item Type: <value or null>
Description: <free text description>

Output:

Return only a JSON object of this form:

{
  "subway_location": ["<line or station>", "<line or station>"],
  "color": ["<color1>", "<color2>"],
  "item_category": "<standardized category or null>",
  "item_type": ["<type1>", "<type2>"],
  "description": "<clean description>",
  "time": "<ISO 8601 UTC timestamp>"
}

If a field has a single value it is still an array where the specification says array.
If you cannot confidently match a value, use "null" or an empty array as appropriate.

Do not output any explanation. Only output the JSON object.
"""


# -----------------------
# TAG / DATA HELPERS
# -----------------------

@st.cache_data
def load_tag_data():
    """Load Tags.xlsx and prepare tag lists."""
    try:
        df = pd.read_excel("Tags.xlsx")  # requires openpyxl
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


def extract_field(text_block: str, field: str) -> str:
    """Extract 'Field: value' from a structured block."""
    match = re.search(rf"{field}:\s*(.*)", text_block)
    return match.group(1).strip() if match else "null"


def is_structured_record(message: str) -> bool:
    """Detect if a message is the final structured record."""
    return message.strip().startswith("Subway Location:")


def standardize_description(text_block: str, tags: Dict) -> Dict:
    """Send structured text + tag summary to Gemini and parse JSON."""
    tags_summary = (
        "\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )

    full_prompt = f"{STANDARDIZER_PROMPT}\n\nHere is the structured input to standardize:\n{text_block}\n{tags_summary}"

    response = safe_generate(full_prompt, context="standardize_description")

    try:
        cleaned = response.text.strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        json_text = cleaned[json_start:json_end]
        data = json.loads(json_text)

        # Fill missing time
        if "time" not in data or not data["time"]:
            data["time"] = datetime.now(timezone.utc).isoformat()

        # Normalize lists
        for key in ["subway_location", "color", "item_type"]:
            if key in data and isinstance(data[key], str):
                data[key] = [data[key]]
            elif key not in data:
                data[key] = []

        if "item_category" not in data:
            data["item_category"] = "null"

        if "description" not in data:
            data["description"] = extract_field(text_block, "Description")

        return data

    except Exception:
        st.error("Model output could not be parsed as JSON. Raw output below:")
        st.text(response.text)
        return {}


# -----------------------
# SIMPLE VALIDATION HELPERS
# -----------------------

def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# -----------------------
# VECTOR STORE HELPERS (CHROMA)
# -----------------------

def get_next_found_id() -> int:
    """Generate a simple incremental ID for found items (for display/admin)."""
    if "next_found_id" not in st.session_state:
        st.session_state.next_found_id = 1
    nid = st.session_state.next_found_id
    st.session_state.next_found_id += 1
    return nid


def save_found_item_to_vectorstore(json_data: Dict, contact: str) -> int:
    """
    Add a found item into Chroma vector DB with metadata.
    Returns a local numeric ID.
    """
    if vector_store is None:
        st.error("Vector store is not available; cannot save found item.")
        return -1

    description = json_data.get("description", "")
    if not description:
        st.error("Found item description is empty; cannot embed.")
        return -1

    found_id = get_next_found_id()

    metadata = {
        "record_type": "found",
        "found_id": found_id,
        "subway_location": json_data.get("subway_location", []),
        "color": json_data.get("color", []),
        "item_category": json_data.get("item_category", ""),
        "item_type": json_data.get("item_type", []),
        "description": description,
        "contact": contact,
        "time": json_data.get("time"),
    }

    try:
        vector_store.add_texts(
            texts=[description],
            metadatas=[metadata],
            ids=[str(found_id)],
        )
        vector_store.persist()
        return found_id
    except Exception as e:
        st.error(f"Error saving found item to vector store: {e}")
        return -1


def compute_match_explanation(
    doc_meta: Dict[str, Any], query_json: Dict[str, Any], distance: float
) -> Tuple[float, int, List[str]]:
    """
    Combine vector distance + tag overlap into a simple heuristic.
    Returns: (combined_score_0_to_1, tag_score, list_of_text_explanations).
    """
    explanations: List[str] = []
    tag_score = 0
    max_tag_score = 3  # color, type, location

    # Color overlap
    q_colors = set(query_json.get("color", []) or [])
    f_colors = set(doc_meta.get("color", []) or [])
    if q_colors and f_colors and q_colors.intersection(f_colors):
        tag_score += 1
        explanations.append("Same or similar color")

    # Item type overlap
    q_types = set(query_json.get("item_type", []) or [])
    f_types = set(doc_meta.get("item_type", []) or [])
    if q_types and f_types and q_types.intersection(f_types):
        tag_score += 1
        explanations.append("Same or similar item type")

    # Location overlap
    q_locs = set(query_json.get("subway_location", []) or [])
    f_locs = set(doc_meta.get("subway_location", []) or [])
    if q_locs and f_locs and q_locs.intersection(f_locs):
        tag_score += 1
        explanations.append("Same or similar station")

    # Normalize distance into similarity (0..1, rough)
    similarity = max(0.0, 1.0 - distance)
    # Combine (tunable weights)
    combined = 0.7 * similarity + 0.3 * (tag_score / max_tag_score if max_tag_score else 0.0)
    return combined, tag_score, explanations


def search_matches_for_lost_item(
    final_json: Dict, top_k: int, max_distance: float
):
    """
    Use vector DB (Chroma) to search for similar found items.
    Returns:
        - all_candidates: [(doc, distance), ...]
        - filtered: only items with distance <= max_distance
    """
    if vector_store is None:
        return [], []

    query_text = final_json.get("description", "")
    if not query_text:
        return [], []

    # Optional filter by category
    filter_dict: Dict[str, Any] = {"record_type": "found"}
    if final_json.get("item_category") and final_json["item_category"] != "null":
        filter_dict["item_category"] = final_json["item_category"]

    try:
        docs_scores = vector_store.similarity_search_with_score(
            query_text,
            k=top_k,
            filter=filter_dict,
        )
    except Exception as e:
        st.error(f"Error during vector search: {e}")
        docs_scores = []

    filtered = [(doc, score) for doc, score in docs_scores if score <= max_distance]
    return docs_scores, filtered


def get_all_found_items_as_df() -> pd.DataFrame:
    """Pull all 'found' items from Chroma for admin view."""
    if vector_store is None:
        return pd.DataFrame()

    try:
        coll = vector_store._collection  # underlying chroma collection
        data = coll.get()  # ids, documents, metadatas
    except Exception as e:
        st.error(f"Error reading from vector store: {e}")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    for id_, doc, meta in zip(ids, docs, metas):
        if not meta:
            continue
        if meta.get("record_type") != "found":
            continue

        rows.append(
            {
                "found_id": meta.get("found_id", id_),
                "description": meta.get("description", doc),
                "subway_location": ", ".join(meta.get("subway_location", [])),
                "color": ", ".join(meta.get("color", [])),
                "item_category": meta.get("item_category", ""),
                "item_type": ", ".join(meta.get("item_type", [])),
                "contact": meta.get("contact", ""),
                "time": meta.get("time", ""),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# -----------------------
# APP INIT
# -----------------------

tag_data = load_tag_data()
if not tag_data:
    st.stop()

if gemini_client is None:
    st.stop()

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🧳 Operator: Upload Found Item",
        "🔍 Rider: Report Lost Item",
        "📦 Admin: Inventory Overview",
    ],
)


# ===============================================================
# PAGE 1 — OPERATOR SIDE
# ===============================================================

if page.startswith("🧳"):
    st.markdown('<div class="main-title">Operator View: Upload Found Item</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        "Staff can quickly capture details about a found item and store it in the Lost & Found inventory."
        "</div>",
        unsafe_allow_html=True,
    )

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
            ),
        )
        st.session_state.operator_msgs = []

    # Show conversation history
    for msg in st.session_state.operator_msgs:
        with st.chat_message("assistant" if msg["role"] == "model" else "user"):
            st.markdown(msg["content"])

    # Initial intake controls
    if not st.session_state.operator_msgs:
        st.markdown("### Step 1 · Add a photo or quick description")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_image = st.file_uploader(
                "Image of the found item (optional)",
                type=["jpg", "jpeg", "png"],
                key="operator_image",
            )
        with col2:
            initial_text = st.text_input(
                "Short description",
                placeholder="e.g., black backpack with a NASA patch",
                key="operator_text",
            )

        if st.button("▶ Start Intake"):
            if not uploaded_image and not initial_text:
                st.error("Please upload an image or enter a short description.")
            else:
                message_content = ""
                if uploaded_image:
                    img = Image.open(uploaded_image).convert("RGB")
                    st.image(img, width=220, caption="Found item image")
                    message_content += "I have uploaded an image of the found item. "
                if initial_text:
                    message_content += initial_text

                st.session_state.operator_msgs.append(
                    {"role": "user", "content": message_content}
                )
                with st.spinner("Analyzing item with Gemini..."):
                    response = safe_send(
                        st.session_state.operator_chat,
                        message_content,
                        context="operator intake",
                    )
                st.session_state.operator_msgs.append(
                    {"role": "model", "content": response.text}
                )
                st.rerun()

    # Continue chat
    operator_input = st.chat_input("Add more details for the operator assistant…")
    if operator_input:
        st.session_state.operator_msgs.append(
            {"role": "user", "content": operator_input}
        )
        with st.spinner("Processing..."):
            response = safe_send(
                st.session_state.operator_chat,
                operator_input,
                context="operator follow-up",
            )
        st.session_state.operator_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.rerun()

    # When final structured record appears
    if st.session_state.operator_msgs and is_structured_record(
        st.session_state.operator_msgs[-1]["content"]
    ):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        st.markdown("### Step 2 · Structured description from Gemini")
        st.code(structured_text, language="markdown")

        final_json = standardize_description(structured_text, tag_data)
        if final_json:
            st.markdown("### Step 3 · Standardized tags and JSON record")
            st.json(final_json)

            st.markdown("### Step 4 · Save to Lost & Found inventory")
            contact = st.text_input("Operator contact or badge ID (optional)")

            if st.button("💾 Save Found Item to Vector DB"):
                found_id = save_found_item_to_vectorstore(final_json, contact)
                if found_id > 0:
                    st.success(f"Saved! This item is stored as Found ID `{found_id}`.")
                    st.info("This item can now be matched against future lost item reports.")


# ===============================================================
# PAGE 2 — USER SIDE (RIDER)
# ===============================================================

if page.startswith("🔍"):
    st.markdown('<div class="main-title">Rider View: Report Lost Item</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        "Describe your lost item, and the system will search for similar found items in the inventory."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Step 1 · Optional quick info")
    with st.expander("Add quick filters like station or category"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_choice = st.selectbox(
                "Subway station (optional)", [""] + tag_data["locations"]
            )
        with col2:
            category_choice = st.selectbox(
                "Item category (optional)", [""] + tag_data["categories"]
            )
        with col3:
            type_choice = st.selectbox(
                "Item type (optional)", [""] + tag_data["item_types"]
            )

    st.markdown("### Step 2 · Describe or show your lost item")
    col_img, col_text = st.columns(2)
    with col_img:
        uploaded_image = st.file_uploader(
            "Image of lost item (optional)",
            type=["jpg", "jpeg", "png"],
            key="user_image",
        )
    with col_text:
        initial_text = st.text_input(
            "Short description",
            placeholder="e.g., blue iPhone with cracked screen",
            key="user_text",
        )

    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=USER_SIDE_GENERATOR_PROMPT,
            ),
        )
        st.session_state.user_msgs = []

    # Show chat history
    for msg in st.session_state.user_msgs:
        with st.chat_message("assistant" if msg["role"] == "model" else "user"):
            st.markdown(msg["content"])

    # Start report
    if not st.session_state.user_msgs and st.button("▶ Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
        else:
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=250, caption="Your lost item (photo)")
                message_text += "I have uploaded an image of my lost item. "
            if initial_text:
                message_text += initial_text

            st.session_state.user_msgs.append(
                {"role": "user", "content": message_text}
            )
            with st.spinner("Analyzing your description..."):
                response = safe_send(
                    st.session_state.user_chat,
                    message_text,
                    context="user initial report",
                )
            st.session_state.user_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.rerun()

    # Continue chat
    user_input = st.chat_input("Add more details about your lost item…")
    if user_input:
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

    # When final structured record appears
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

        st.markdown("### Step 3 · Final merged record before standardization")
        st.code(merged_text, language="markdown")

        final_json = standardize_description(merged_text, tag_data)
        if final_json:
            st.markdown("### Step 4 · Standardized record used for matching")
            st.json(final_json)

            st.markdown("### Step 5 · Contact information")
            contact = st.text_input("Phone number (10 digits, numeric only)")
            email = st.text_input("Email address")

            st.markdown("### Step 6 · Matching options")
            col_k, col_thresh = st.columns(2)
            with col_k:
                top_k = st.slider(
                    "Number of candidate matches to retrieve (top-K)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                )
            with col_thresh:
                max_distance = st.slider(
                    "Maximum distance (lower = more similar)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.01,
                    help="Matches with distance greater than this will be hidden.",
                )

            if st.button("🔎 Submit Lost Item & Find Matches"):
                if not validate_phone(contact):
                    st.error("Please enter a ten digit phone number (no spaces).")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    st.success("Lost item report submitted (contact captured locally).")

                    if vector_store is None:
                        st.info(
                            "Vector store is not configured, so no matches can be shown yet."
                        )
                    else:
                        with st.spinner(
                            "Searching for similar found items using embeddings..."
                        ):
                            all_candidates, filtered = search_matches_for_lost_item(
                                final_json, top_k=top_k, max_distance=max_distance
                            )

                        if not all_candidates:
                            st.info(
                                "No items are stored in the vector DB yet, so no matches can be returned."
                            )
                        else:
                            st.markdown("### Step 7 · Candidate matches")
                            if not filtered:
                                st.info(
                                    "No matches under the current distance threshold. "
                                    "Showing raw top-K candidates instead."
                                )
                                to_show = all_candidates
                            else:
                                to_show = filtered

                            for doc, score in to_show:
                                meta = doc.metadata or {}
                                combined, tag_score, reasons = compute_match_explanation(
                                    meta, final_json, score
                                )
                                similarity_pct = max(0.0, (1.0 - score) * 100.0)

                                st.markdown('<div class="match-card">', unsafe_allow_html=True)
                                st.markdown(
                                    f'<div class="match-title">Candidate match — ID '
                                    f'<code>{meta.get("found_id", "N/A")}</code></div>',
                                    unsafe_allow_html=True,
                                )

                                st.markdown(
                                    f"- **Distance:** `{score:.4f}`  "
                                    f"- **Approx. similarity:** `{similarity_pct:.1f}%`  "
                                    f"- **Tag score:** `{tag_score}/3`  "
                                    f"- **Combined confidence:** `{combined*100:.1f}%`",
                                )

                                st.write(meta.get("description", doc.page_content))

                                pills = []
                                if meta.get("subway_location"):
                                    pills.append(
                                        " ".join(
                                            [
                                                f"<span class='pill'>📍 {loc}</span>"
                                                for loc in meta["subway_location"]
                                            ]
                                        )
                                    )
                                if meta.get("color"):
                                    pills.append(
                                        " ".join(
                                            [
                                                f"<span class='pill'>🎨 {c}</span>"
                                                for c in meta["color"]
                                            ]
                                        )
                                    )
                                if meta.get("item_category"):
                                    pills.append(
                                        f"<span class='pill'>📂 {meta['item_category']}</span>"
                                    )
                                if meta.get("item_type"):
                                    pills.append(
                                        " ".join(
                                            [
                                                f"<span class='pill'>👜 {t}</span>"
                                                for t in meta["item_type"]
                                            ]
                                        )
                                    )

                                if pills:
                                    st.markdown(" ".join(pills), unsafe_allow_html=True)

                                if reasons:
                                    st.markdown("**Why this matched:** " + "; ".join(reasons))

                                st.markdown("</div>", unsafe_allow_html=True)
                                st.markdown("---")


# ===============================================================
# PAGE 3 — ADMIN / INVENTORY
# ===============================================================

if page.startswith("📦"):
    st.markdown('<div class="main-title">Admin: Inventory Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        "View all stored found items and basic stats from the Chroma vector database."
        "</div>",
        unsafe_allow_html=True,
    )

    if vector_store is None:
        st.error("Vector store is not available.")
    else:
        df_found = get_all_found_items_as_df()
        if df_found.empty:
            st.info("No found items stored yet.")
        else:
            # Simple stats row
            total_items = len(df_found)
            active_categories = df_found["item_category"].replace("", pd.NA).dropna().nunique()
            top_category = (
                df_found["item_category"].replace("", pd.NA).dropna().value_counts().index[0]
                if active_categories > 0
                else "N/A"
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(
                    f"<div class='stat-card'>📦 <b>Total found items</b><br>{total_items}</div>",
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f"<div class='stat-card'>📂 <b>Distinct categories</b><br>{active_categories}</div>",
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(
                    f"<div class='stat-card'>⭐ <b>Most common category</b><br>{top_category}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("### Found items (table view)")
            st.dataframe(df_found, use_container_width=True)


