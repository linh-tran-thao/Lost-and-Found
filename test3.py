# ================================================================
# LOST & FOUND AI SYSTEM (Clean Rewrite - Chroma Vector DB)
# ================================================================

import json
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


# ================================================================
# CONFIG
# ================================================================

MODEL_NAME = "gemini-2.0-flash"

st.set_page_config(
    page_title="Lost & Found AI System",
    page_icon="🧳",
    layout="wide"
)

# Basic nicer CSS
st.markdown("""
<style>
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 1.0rem;
}
.metric-box {
    background: #fafafa;
    border: 1px solid #ddd;
    padding: 0.8rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# API KEYS + CLIENTS
# ================================================================

def load_secrets():
    try:
        s = st.secrets.to_dict()
    except:
        s = {}
    return s.get("GEMINI_API_KEY"), s.get("OPENAI_API_KEY")

GEMINI_KEY, OPENAI_KEY = load_secrets()

if not GEMINI_KEY:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

gemini_client = genai.Client(api_key=GEMINI_KEY)

if not OPENAI_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)


# ================================================================
# VECTOR STORE (CHROMA)
# ================================================================

@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name="lost_and_found_items",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

vector_store = get_vector_store()


# ================================================================
# PROMPTS
# ================================================================

INTAKE_PROMPT = """
You are a Lost & Found intake assistant.

The user provides either:
- an image, or
- a short description.

Your job:
1. Describe exactly what you see or what the user wrote.
2. Ask 2–3 short clarification questions.
3. Then STOP when the user says "done" or enough info is given.
4. Output ONLY this structured record:

Subway Location: <station or null>
Color: <color or null>
Item Category: <category or null>
Item Type: <type or null>
Description: <concise factual summary>

Do NOT output commentary, questions, or reasoning.
Only output the structured record.
"""

def safe_generate(prompt: str):
    try:
        return gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    except genai_errors.ClientError as e:
        st.error(f"Gemini error: {e}")
        st.stop()


# ================================================================
# CHROMA HELPERS
# ================================================================

def save_found_item(json_data: Dict[str, Any], contact: str) -> int:
    """Save a found item with metadata into Chroma."""
    description = json_data.get("description", "")
    if not description:
        st.error("Missing description.")
        return -1

    # auto ID
    next_id = int(datetime.now().timestamp())

    metadata = {
        "record_type": "found",
        "found_id": next_id,
        "description": description,
        "subway_location": json_data.get("subway_location", []),
        "color": json_data.get("color", []),
        "item_category": json_data.get("item_category", ""),
        "item_type": json_data.get("item_type", []),
        "contact": contact,
        "time": datetime.now(timezone.utc).isoformat()
    }

    vector_store.add_texts(
        texts=[description],
        metadatas=[metadata],
        ids=[str(next_id)]
    )
    vector_store.persist()
    return next_id


def get_found_items_df() -> pd.DataFrame:
    """Return all found items."""
    data = vector_store._collection.get(include=["documents", "metadatas"])
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    rows = []
    for id_, text, meta in zip(ids, docs, metas):
        if meta.get("record_type") != "found":
            continue
        rows.append({
            "found_id": meta.get("found_id", id_),
            "description": meta.get("description", text),
            "subway_location": ", ".join(meta.get("subway_location", [])),
            "color": ", ".join(meta.get("color", [])),
            "item_category": meta.get("item_category", ""),
            "item_type": ", ".join(meta.get("item_type", [])),
            "contact": meta.get("contact", ""),
            "time": meta.get("time", "")
        })

    return pd.DataFrame(rows)


def search_matches(query_desc: str, top_k: int = 5) -> List[Tuple[Any, float]]:
    try:
        return vector_store.similarity_search_with_score(query_desc, k=top_k)
    except:
        return []


# ================================================================
# SIDEBAR NAV
# ================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Operator: Found Item", "User: Lost Item", "Admin: View Items"])
top_k = st.sidebar.slider("Top-K Matches", 1, 10, 5)


# ================================================================
# PAGE 1 — OPERATOR
# ================================================================

if page == "Operator: Found Item":
    st.header("👮 Operator – Upload Found Item")

    uploaded = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png"])
    text_input = st.text_input("Short description")
    contact = st.text_input("Operator contact/badge")

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(system_instruction=INTAKE_PROMPT),
        )
        st.session_state.operator_msgs = []

    if st.button("Start Intake"):
        msg = ""
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width=200)
            msg += "I uploaded an image. "
        if text_input:
            msg += text_input

        st.session_state.operator_msgs.append({"role": "user", "content": msg})
        response = st.session_state.operator_chat.send_message(msg)
        st.session_state.operator_msgs.append({"role": "model", "content": response.text})
        st.rerun()

    # Chat UI
    for m in st.session_state.operator_msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_input = st.chat_input("Answer questions or say 'done'")
    if user_input:
        st.session_state.operator_msgs.append({"role": "user", "content": user_input})
        response = st.session_state.operator_chat.send_message(user_input)
        st.session_state.operator_msgs.append({"role": "model", "content": response.text})
        st.rerun()

    # Detect structured record
    if st.session_state.operator_msgs and st.session_state.operator_msgs[-1]["content"].startswith("Subway Location:"):
        st.subheader("Final Structured Item")
        structured = st.session_state.operator_msgs[-1]["content"]
        st.code(structured)

        # Convert record into JSON dict
        def extract(f): 
            import re
            m = re.search(fr"{f}:\s*(.*)", structured)
            return m.group(1).strip() if m else "null"

        json_data = {
            "subway_location": [] if extract("Subway Location") == "null" else [extract("Subway Location")],
            "color": [] if extract("Color") == "null" else [extract("Color")],
            "item_category": extract("Item Category"),
            "item_type": [] if extract("Item Type") == "null" else [extract("Item Type")],
            "description": extract("Description")
        }

        if st.button("Save to Chroma"):
            if not contact:
                st.error("Add operator contact first.")
            else:
                ID = save_found_item(json_data, contact)
                st.success(f"Saved with ID {ID}")


# ================================================================
# PAGE 2 — USER LOST ITEM
# ================================================================

elif page == "User: Lost Item":
    st.header("🧍 User – Report Lost Item")

    uploaded = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png"])
    text_input = st.text_input("Short description")

    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(system_instruction=INTAKE_PROMPT),
        )
        st.session_state.user_msgs = []

    if st.button("Start Lost Item Report"):
        msg = ""
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width=200)
            msg += "This is my lost item image. "
        if text_input:
            msg += text_input

        st.session_state.user_msgs.append({"role": "user", "content": msg})
        response = st.session_state.user_chat.send_message(msg)
        st.session_state.user_msgs.append({"role": "model", "content": response.text})
        st.rerun()

    # Chat display
    for m in st.session_state.user_msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_input = st.chat_input("Answer questions or say 'done'")
    if user_input:
        st.session_state.user_msgs.append({"role": "user", "content": user_input})
        response = st.session_state.user_chat.send_message(user_input)
        st.session_state.user_msgs.append({"role": "model", "content": response.text})
        st.rerun()

    # Detect structured record
    if st.session_state.user_msgs and st.session_state.user_msgs[-1]["content"].startswith("Subway Location:"):
        st.subheader("Final Structured Lost Item")
        structured = st.session_state.user_msgs[-1]["content"]
        st.code(structured)

        def extract(f):
            import re
            m = re.search(fr"{f}:\s*(.*)", structured)
            return m.group(1).strip() if m else "null"

        json_lost = {
            "description": extract("Description")
        }

        if st.button("Search for Matches"):
            st.subheader("Possible Matches")

            results = search_matches(json_lost["description"], top_k=top_k)

            if not results:
                st.info("No matches found.")
            else:
                for doc, score in results:
                    meta = doc.metadata
                    similarity = max(0, (1 - score) * 100)

                    st.write(f"### Similarity: {similarity:.1f}%")
                    st.write("**Description:**", meta["description"])
                    st.write("🚉", ", ".join(meta.get("subway_location", [])))
                    st.write("🎨", ", ".join(meta.get("color", [])))
                    st.write("🔖", ", ".join(meta.get("item_type", [])))
                    st.caption(f"Found ID: {meta['found_id']} – {meta['time']}")
                    st.markdown("---")


# ================================================================
# PAGE 3 — ADMIN
# ================================================================

elif page == "Admin: View Items":
    st.header("📊 Admin – All Found Items")

    df = get_found_items_df()
    if df.empty:
        st.info("No items stored yet.")
    else:
        st.dataframe(df, use_container_width=True)
