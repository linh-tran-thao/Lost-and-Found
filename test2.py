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
# BASIC CONFIG / THEME
# -----------------------

MODEL_NAME = "gemini-2.0-flash"

st.set_page_config(
    page_title="Lost & Found Intake",
    page_icon="🧳",
    layout="wide",
)

# Simple custom CSS for nicer cards / headers
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #555555;
        margin-bottom: 0.8rem;
    }
    .card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #E3E3E3;
        background-color: #FAFAFA;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        border: 1px solid #E3E3E3;
        background-color: #FFFFFF;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        color: #777777;
    }
    .metric-card p {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
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
def get_embeddings():
    if not secrets["openai_key"]:
        st.warning("OPENAI_API_KEY is not set; semantic matching will be disabled.")
        return None
    try:
        return OpenAIEmbeddings(openai_api_key=secrets["openai_key"])
    except Exception as e:
        st.error(f"Error creating OpenAI embeddings: {e}")
        return None


@st.cache_resource
def get_vector_store():
    """
    Create or load a Chroma vector DB using OpenAI embeddings.
    This stores all *found* items with metadata.
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return None

    try:
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
# VECTOR STORE HELPERS
# -----------------------

def get_all_found_items_raw() -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Return raw ids, documents, metadatas from Chroma."""
    if vector_store is None:
        return [], [], []
    try:
        coll = vector_store._collection
        data = coll.get()
        return data.get("ids", []), data.get("documents", []), data.get("metadatas", [])
    except Exception:
        return [], [], []


def get_all_found_items_as_df() -> pd.DataFrame:
    """Pull all 'found' items from Chroma for admin view & metrics."""
    ids, docs, metas = get_all_found_items_raw()
    rows: List[Dict[str, Any]] = []

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


def get_next_found_id() -> int:
    """Generate a simple incremental ID for found items (for display/admin)."""
    if "next_found_id" not in st.session_state:
        # Derive from existing items so IDs don't reset
        df = get_all_found_items_as_df()
        if df.empty:
            st.session_state.next_found_id = 1
        else:
            st.session_state.next_found_id = int(df["found_id"].max()) + 1
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


def search_matches_for_lost_item(
    final_json: Dict, top_k: int, max_distance: float
) -> Tuple[List[Any], List[Any]]:
    """
    Use vector DB (Chroma) to search for similar found items.
    Returns (all_candidates, filtered_by_distance)
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


# -----------------------
# APP INIT
# -----------------------

tag_data = load_tag_data()
if not tag_data:
    st.stop()

if gemini_client is None:
    st.stop()

# -----------------------
# SIDEBAR NAV + MATCHING CONTROLS
# -----------------------

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["👮 Operator: Upload Found Item", "🧍 User: Report Lost Item", "📊 Admin: View Found Items"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Matching Controls")

top_k_sidebar = st.sidebar.slider(
    "Top-K candidates",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Number of candidate matches to retrieve from the vector DB.",
)

max_distance_sidebar = st.sidebar.slider(
    "Max distance (lower = more similar)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.01,
    help="Matches with distance greater than this will be filtered out.",
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Distance is model-dependent. Start with 0.4–0.5 and adjust.")


# -----------------------
# TOP DASHBOARD METRICS
# -----------------------

df_found_for_metrics = get_all_found_items_as_df()
total_found = len(df_found_for_metrics)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown('<div class="metric-card"><h3>Total Found Items</h3><p>{}</p></div>'.format(total_found), unsafe_allow_html=True)
with col_m2:
    st.markdown(
        '<div class="metric-card"><h3>Matching Top-K</h3><p>{}</p></div>'.format(top_k_sidebar),
        unsafe_allow_html=True,
    )

st.markdown("")  # spacing


# ===============================================================
# PAGE 1: OPERATOR – UPLOAD FOUND ITEM
# ===============================================================

if page.startswith("👮"):
    st.markdown('<div class="main-title">👮 Operator View: Upload Found Item</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Frontline staff can quickly record what they found, '
        'then the system standardizes tags and stores it in the vector database for matching later.</div>',
        unsafe_allow_html=True,
    )

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model= MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
            ),
        )
        st.session_state.operator_msgs = []

    # Show conversation history
    if st.session_state.operator_msgs:
        st.markdown('<div class="section-title">🗨️ Intake Conversation</div>', unsafe_allow_html=True)
    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial intake
    if not st.session_state.operator_msgs:
        st.markdown('<div class="section-title">➕ Start a New Found Item</div>', unsafe_allow_html=True)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                uploaded_image = st.file_uploader(
                    "📷 Image of the found item (optional)",
                    type=["jpg", "jpeg", "png"],
                    key="operator_image",
                )
            with col2:
                initial_text = st.text_input(
                    "📝 Short description",
                    placeholder="For example: black backpack with a NASA patch",
                    key="operator_text",
                )

        if st.button("🚀 Start Intake"):
            if not uploaded_image and not initial_text:
                st.error("Please upload an image or enter a short description.")
            else:
                message_content = ""
                if uploaded_image:
                    img = Image.open(uploaded_image).convert("RGB")
                    st.image(img, width=220, caption="Preview of found item")
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
    operator_input = st.chat_input("Add more details for the operator bot, or say 'done' when ready.")
    if operator_input:
        st.session_state.operator_msgs.append(
            {"role": "user", "content": operator_input}
        )
        with st.spinner("Processing operator message..."):
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
        st.markdown('<div class="section-title">📦 Final Structured Description</div>', unsafe_allow_html=True)
        st.code(structured_text)

        final_json = standardize_description(structured_text, tag_data)
        if final_json:
            st.markdown('<div class="section-title">🏷️ Standardized Tags (JSON)</div>', unsafe_allow_html=True)
            st.json(final_json)

            st.markdown('<div class="section-title">👤 Operator Contact</div>', unsafe_allow_html=True)
            contact = st.text_input("Operator contact or badge ID")

            if st.button("💾 Save Found Item to Vector DB"):
                found_id = save_found_item_to_vectorstore(final_json, contact)
                if found_id > 0:
                    st.success(f"Found item saved with ID `{found_id}` (Chroma vector DB).")


# ===============================================================
# PAGE 2: USER – REPORT LOST ITEM & MATCH
# ===============================================================

if page.startswith("🧍"):
    st.markdown('<div class="main-title">🧍 User View: Report Lost Item</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Riders describe what they lost. The system standardizes their description '
        'and searches for similar found items using embeddings.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">⚡ Optional Quick Info</div>', unsafe_allow_html=True)
    with st.expander("Click to pre-select station / category / type"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_choice = st.selectbox(
                "🚉 Subway station (optional)", [""] + tag_data["locations"]
            )
        with col2:
            category_choice = st.selectbox(
                "📂 Item category (optional)", [""] + tag_data["categories"]
            )
        with col3:
            type_choice = st.selectbox(
                "🔖 Item type (optional)", [""] + tag_data["item_types"]
            )

    st.markdown('<div class="section-title">📷 / 📝 Describe Your Lost Item</div>', unsafe_allow_html=True)
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
            placeholder="For example: blue iPhone with cracked screen",
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
    if st.session_state.user_msgs:
        st.markdown('<div class="section-title">🗨️ Chat with the Lost-Item Assistant</div>', unsafe_allow_html=True)
    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Start report
    if not st.session_state.user_msgs and st.button("🚀 Start Lost Item Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
        else:
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=240, caption="Your lost item (preview)")
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
    user_input = st.chat_input("Add more details, answer questions, or say 'done' when ready.")
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

        st.markdown('<div class="section-title">🧩 Final Structured Record (Before Standardization)</div>', unsafe_allow_html=True)
        merged_text = f"""
Subway Location: {location_choice or extract_field(structured_text, 'Subway Location')}
Color: {extract_field(structured_text, 'Color')}
Item Category: {category_choice or extract_field(structured_text, 'Item Category')}
Item Type: {type_choice or extract_field(structured_text, 'Item Type')}
Description: {extract_field(structured_text, 'Description')}
        """
        st.code(merged_text)

        final_json = standardize_description(merged_text, tag_data)
        if final_json:
            st.markdown('<div class="section-title">🏷️ Standardized Lost Item</div>', unsafe_allow_html=True)
            st.json(final_json)

            st.markdown('<div class="section-title">📇 Contact Information</div>', unsafe_allow_html=True)
            contact = st.text_input("Phone number (10 digits, numbers only)")
            email = st.text_input("Email address")

            st.info("Your contact is only used to follow up if a strong match is found.")

            if st.button("🔍 Submit & Search for Matches"):
                if not validate_phone(contact):
                    st.error("Please enter a ten digit phone number (no spaces).")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    st.success("Lost item report received (not stored permanently in DB for this demo).")

                    if vector_store is None:
                        st.info(
                            "Vector store is not configured, so no matches can be shown yet."
                        )
                    else:
                        with st.spinner(
                            "Searching for similar found items using embeddings..."
                        ):
                            all_candidates, filtered = search_matches_for_lost_item(
                                final_json,
                                top_k=top_k_sidebar,
                                max_distance=max_distance_sidebar,
                            )

                        if not all_candidates:
                            st.info(
                                "No items are stored in the vector DB yet, so no matches can be returned."
                            )
                        else:
                            st.markdown('<div class="section-title">📌 Candidate Matches</div>', unsafe_allow_html=True)

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
                                similarity_pct = max(0.0, (1.0 - score) * 100.0)

                                st.markdown(
                                    f"**Distance:** `{score:.4f}`  ·  "
                                    f"**Similarity (approx):** `{similarity_pct:.1f}%`"
                                )

                                st.write("**Description:**", meta.get("description", doc.page_content))

                                if meta.get("subway_location"):
                                    st.write(
                                        "🚉 Location:", ", ".join(meta["subway_location"])
                                    )
                                if meta.get("color"):
                                    st.write("🎨 Color:", ", ".join(meta["color"]))
                                if meta.get("item_category"):
                                    st.write("📂 Category:", meta["item_category"])
                                if meta.get("item_type"):
                                    st.write("🔖 Type:", ", ".join(meta["item_type"]))

                                st.caption(f"Found item ID: {meta.get('found_id', 'N/A')} · Time: {meta.get('time', '')}")
                                with st.expander("View raw metadata"):
                                    st.json(meta)
                                st.markdown("---")


# ===============================================================
# PAGE 3: ADMIN – VIEW FOUND ITEMS
# ===============================================================

if page.startswith("📊"):
    st.markdown('<div class="main-title">📊 Admin: View Stored Found Items</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Quick view of everything operators have logged into the vector DB.</div>',
        unsafe_allow_html=True,
    )

    if vector_store is None:
        st.error("Vector store is not available.")
    else:
        df_found = get_all_found_items_as_df()
        if df_found.empty:
            st.info("No found items stored yet.")
        else:
            st.markdown('<div class="section-title">📦 Found Items Table</div>', unsafe_allow_html=True)
            st.dataframe(df_found, use_container_width=True)
            st.caption("Scroll horizontally to see all columns.")

