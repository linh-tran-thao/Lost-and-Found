import os
import json
import uuid
from typing import List, Dict, Any

import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

# =========================
# BASIC CONFIG
# =========================

st.set_page_config(
    page_title="Lost & Found AI — Demo",
    page_icon="🧳",
    layout="wide",
)

INVENTORY_FILE = "inventory.json"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "lost_found_items"

# =========================
# SECRETS / CLIENTS
# =========================

@st.cache_resource
def get_openai_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return ""


OPENAI_API_KEY = get_openai_key()

@st.cache_resource
def get_embedder():
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY is not set; embeddings + matching will be disabled.")
        return None
    try:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Error creating OpenAI embedder: {e}")
        return None


@st.cache_resource
def get_chroma_collection():
    """
    Persistent Chroma collection stored in ./chroma_db
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance
    )
    return collection


embedder = get_embedder()
collection = get_chroma_collection()

# =========================
# SIMPLE PERSISTENCE (JSON)
# =========================

def load_inventory() -> List[Dict[str, Any]]:
    if not os.path.exists(INVENTORY_FILE):
        return []
    try:
        with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_inventory(items: List[Dict[str, Any]]):
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


# keep inventory in session for quick access
if "inventory" not in st.session_state:
    st.session_state.inventory = load_inventory()

# =========================
# EMBEDDING HELPERS
# =========================

def embed_text(text: str) -> List[float]:
    if not embedder:
        return []
    try:
        return embedder.embed_query(text)
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        return []


def index_item_in_chroma(item: Dict[str, Any]):
    """
    Store / update an item in Chroma.
    """
    if not embedder or not collection:
        return

    doc_id = item["id"]
    text_for_embedding = (
        f"{item['title']}. {item['description']} "
        f"Location: {item.get('location','')}. "
        f"Color: {item.get('color','')}. "
        f"Category: {item.get('category','')}. "
        f"Material: {item.get('material','')}."
    )

    embedding = embed_text(text_for_embedding)
    if not embedding:
        return

    # upsert in case it already exists
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[item],
        documents=[text_for_embedding],
    )


def rebuild_chroma_from_inventory():
    """
    Optional helper if you ever want to rebuild the DB from scratch.
    Not called automatically, but you can call from a dev button.
    """
    if not embedder or not collection:
        return
    # clear and re-index
    collection.delete(where={})
    for item in st.session_state.inventory:
        index_item_in_chroma(item)


# =========================
# MATCHING
# =========================

def find_matches(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not embedder or not collection or not query_text.strip():
        return []

    query_embedding = embed_text(query_text)
    if not query_embedding:
        return []

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    matches = []
    for meta, dist in zip(metadatas, distances):
        # cosine distance in [0,2]; convert to similarity ~ [0,1]
        similarity = max(0.0, 1.0 - dist)
        matches.append(
            {
                "item": meta,
                "distance": float(dist),
                "similarity": float(similarity),
            }
        )
    # already sorted by distance
    return matches


# =========================
# CHAT STATE (LOST ITEM)
# =========================

QUESTIONS = [
    "What kind of item is it? (e.g., backpack, wallet, phone)",
    "What color is it?",
    "Where do you think you lost it? (station, bus route, etc.)",
    "Any other details? (brand, material, pockets, special marks)",
]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "question_idx" not in st.session_state:
    st.session_state.question_idx = 0
if "final_description" not in st.session_state:
    st.session_state.final_description = ""


def reset_session():
    st.session_state.chat_history = []
    st.session_state.answers = {}
    st.session_state.question_idx = 0
    st.session_state.final_description = ""


# =========================
# LAYOUT
# =========================

top_cols = st.columns([3, 2])

# ------------- LEFT: LOST ITEM CHAT -------------
with top_cols[0]:
    st.markdown("## Lost & Found AI — Demo")
    st.markdown(
        "Photo caption → Chat intake → Tag filter → Cosine ranking → Claims"
    )
    st.markdown("### Describe Your Lost Item")

    # show chat
    if not st.session_state.chat_history:
        # initial system-style messages
        st.session_state.chat_history.append(
            {"role": "system", "text": "Hi! I’ll ask a few questions to describe your item."}
        )
        st.session_state.chat_history.append(
            {"role": "system", "text": QUESTIONS[0]}
        )

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(msg["text"])

    # user input box
    user_input = st.text_input(
        "", placeholder="Type here...", key="lost_item_input"
    )

    send_col, reset_col = st.columns([1, 1])
    with send_col:
        if st.button("Send", type="primary"):
            if user_input.strip():
                # record answer
                idx = st.session_state.question_idx
                if idx < len(QUESTIONS):
                    key = f"q{idx+1}"
                    st.session_state.answers[key] = user_input.strip()

                st.session_state.chat_history.append(
                    {"role": "user", "text": user_input.strip()}
                )

                # move to next question or finalize
                if idx + 1 < len(QUESTIONS):
                    st.session_state.question_idx += 1
                    st.session_state.chat_history.append(
                        {"role": "system", "text": QUESTIONS[idx + 1]}
                    )
                else:
                    # build final description
                    parts = []
                    kind = st.session_state.answers.get("q1", "")
                    color = st.session_state.answers.get("q2", "")
                    location = st.session_state.answers.get("q3", "")
                    extra = st.session_state.answers.get("q4", "")

                    if kind:
                        parts.append(kind)
                    if color:
                        parts.append(f"color: {color}")
                    if location:
                        parts.append(f"likely lost at: {location}")
                    if extra:
                        parts.append(extra)

                    final_desc = ". ".join(parts)
                    st.session_state.final_description = final_desc

                    st.session_state.chat_history.append(
                        {
                            "role": "system",
                            "text": f"Thanks! Here is the summary of your item:\n\n> {final_desc}",
                        }
                    )

                    # clear input
                st.session_state.lost_item_input = ""
                st.experimental_rerun()

    with reset_col:
        if st.button("Reset Session"):
            reset_session()
            st.experimental_rerun()

# ------------- RIGHT: MATCHES -------------
with top_cols[1]:
    st.markdown("### Match threshold")
    threshold = st.slider(
        "", min_value=0, max_value=100, value=20, step=5, format="%d%%"
    )

    st.markdown("### Top Matches")

    if st.session_state.final_description and embedder:
        matches = find_matches(
            st.session_state.final_description, top_k=5
        )

        shown = False
        for m in matches:
            sim_pct = int(m["similarity"] * 100)
            if sim_pct < threshold:
                continue

            item = m["item"]
            shown = True
            with st.container(border=True):
                st.markdown(
                    f"**{item['title']}**  —  similarity: `{sim_pct}%`"
                )
                st.write(item.get("description", ""))
                tags = []
                if item.get("location"):
                    tags.append(f"location: {item['location']}")
                if item.get("color"):
                    tags.append(f"color: {item['color']}")
                if item.get("category"):
                    tags.append(f"category: {item['category']}")
                if item.get("material"):
                    tags.append(f"material: {item['material']}")

                if tags:
                    st.caption(" • ".join(tags))

                if item.get("time_found"):
                    st.caption(f"found: {item['time_found']}")

                if item.get("image_url"):
                    st.image(item["image_url"], use_column_width=True)

        if not shown:
            st.write("No matches above threshold.")
    else:
        st.write("No description yet. Answer the questions on the left to see matches.")

# =========================
# STAFF INTAKE + INVENTORY
# =========================

st.markdown("---")
st.markdown("## Intake (Staff)")

bottom_cols = st.columns([2, 3])

with bottom_cols[0]:
    st.markdown("### Lost & Found Intake")

    with st.form("intake_form"):
        title = st.text_input("Title")
        image_url = st.text_input("Image URL (optional)")
        location = st.text_input("Location found")
        time_found = st.text_input("Time found (YYYY-MM-DD HH:MM)")
        color = st.text_input("Color (optional)")
        category = st.text_input("Category (optional)")
        material = st.text_input("Material (optional)")
        description = st.text_area("Short description")

        submitted = st.form_submit_button("Save Item")

    if submitted:
        if not title.strip() or not description.strip():
            st.error("Title and description are required.")
        else:
            item_id = str(uuid.uuid4())
            item = {
                "id": item_id,
                "title": title.strip(),
                "image_url": image_url.strip(),
                "location": location.strip(),
                "time_found": time_found.strip(),
                "color": color.strip(),
                "category": category.strip(),
                "material": material.strip(),
                "description": description.strip(),
            }

            st.session_state.inventory.append(item)
            save_inventory(st.session_state.inventory)
            index_item_in_chroma(item)

            st.success("Item saved to inventory and indexed for matching.")

    # Optional dev button to rebuild vector DB
    if st.checkbox("Rebuild vector DB from inventory (dev tool)"):
        if st.button("Rebuild now"):
            rebuild_chroma_from_inventory()
            st.success("Rebuilt Chroma index from current inventory.")


with bottom_cols[1]:
    st.markdown("### Inventory")

    if not st.session_state.inventory:
        st.write("No items saved yet.")
    else:
        # latest first
        for item in reversed(st.session_state.inventory):
            with st.container(border=True):
                if item.get("image_url"):
                    st.image(item["image_url"], use_column_width=True)

                st.markdown(f"**{item['title']}**")
                subtitle_parts = []
                if item.get("location"):
                    subtitle_parts.append(item["location"])
                if item.get("time_found"):
                    subtitle_parts.append(item["time_found"])
                if subtitle_parts:
                    st.caption(" · ".join(subtitle_parts))

                st.write(item.get("description", ""))

                chips = []
                if item.get("color"):
                    chips.append(f"color: {item['color']}")
                if item.get("category"):
                    chips.append(f"category: {item['category']}")
                if item.get("material"):
                    chips.append(f"material: {item['material']}")

                if chips:
                    st.caption("  |  ".join(chips))

