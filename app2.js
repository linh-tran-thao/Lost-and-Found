import React, { useMemo, useRef, useState, useEffect } from "react";

// =====================================================
// Lost & Found AI — Single-file Demo (with Auto-Caption)
// What changed vs your two snippets:
// 1) Brought ImageCaptioner into this file and styled it to match.
// 2) Added a "Photo → Auto Caption" panel. The generated caption is auto-
//    appended to the reporter description used for matching.
// 3) If the chat answers are empty, the caption alone is still submitted.
// 4) Passes the photo URL through to the created Report context (optional).
// 5) No external UI libs; sandbox-ready.
//    Tip: set window.GEMINI_API_KEY at runtime to use Gemini; otherwise
//    the local heuristic captioner runs.
// =====================================================

/** ======= tiny UI helpers ======= **/
const Card = ({ children, style }) => (
  <div
    style={{
      border: "1px solid #e3e3e3",
      borderRadius: 16,
      padding: 16,
      background: "#fff",
      ...style,
    }}
  >
    {children}
  </div>
);
const Button = ({ children, onClick, style, disabled, type }) => (
  <button
    type={type || "button"}
    onClick={onClick}
    disabled={disabled}
    style={{
      padding: "8px 12px",
      borderRadius: 10,
      border: "1px solid #ddd",
      background: disabled ? "#9ca3af" : "#111827",
      color: "white",
      cursor: disabled ? "not-allowed" : "pointer",
      ...style,
    }}
  >
    {children}
  </button>
);
const Badge = ({ children }) => (
  <span
    style={{
      border: "1px solid #ddd",
      padding: "2px 8px",
      borderRadius: 999,
      fontSize: 12,
    }}
  >
    {children}
  </span>
);
const Input = React.forwardRef((props, ref) => (
  <input
    ref={ref}
    {...props}
    style={{
      ...props.style,
      padding: 8,
      borderRadius: 8,
      border: "1px solid #ddd",
      width: "100%",
    }}
  />
));
const Textarea = (props) => (
  <textarea
    {...props}
    style={{
      ...props.style,
      padding: 8,
      borderRadius: 8,
      border: "1px solid #ddd",
      width: "100%",
    }}
    rows={4}
  />
);

/** ======= ImageCaptioner (merged & lightly restyled) ======= **/
function readFileAsDataURL(file) {
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = rej;
    fr.readAsDataURL(file);
  });
}
function loadImage(src) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = src;
  });
}
function averageColor(img) {
  const c = document.createElement("canvas");
  const w = (c.width = 160);
  const h = (c.height = Math.round((img.height / img.width) * w) || 160);
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  let r = 0,
    g = 0,
    b = 0,
    n = 0;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i + 0];
    g += data[i + 1];
    b += data[i + 2];
    n++;
  }
  return { r: Math.round(r / n), g: Math.round(g / n), b: Math.round(b / n) };
}
function nearestCommonColor(r, g, b) {
  const palette = {
    black: [0, 0, 0],
    white: [255, 255, 255],
    gray: [128, 128, 128],
    silver: [192, 192, 192],
    red: [220, 20, 60],
    blue: [65, 105, 225],
    green: [34, 139, 34],
    yellow: [255, 215, 0],
    orange: [255, 140, 0],
    purple: [128, 0, 128],
    pink: [255, 105, 180],
    brown: [139, 69, 19],
    beige: [245, 245, 220],
  };
  let best = "gray",
    bestDist = Infinity;
  for (const [name, [pr, pg, pb]] of Object.entries(palette)) {
    const d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2;
    if (d < bestDist) {
      bestDist = d;
      best = name;
    }
  }
  return best;
}
async function localCaption(dataUrl, filename = "") {
  const img = await loadImage(dataUrl);
  const { r, g, b } = averageColor(img);
  const color = nearestCommonColor(r, g, b);
  const lower = (filename || "").toLowerCase();
  const category = /wallet/.test(lower)
    ? "wallet"
    : /backpack|bag/.test(lower)
    ? "backpack"
    : /laptop|macbook/.test(lower)
    ? "laptop"
    : /phone|iphone|pixel|samsung/.test(lower)
    ? "phone"
    : /earbud|airpod|headphone/.test(lower)
    ? "wireless earbuds"
    : "item";
  return `A ${color} ${category} on a flat background.`;
}
async function captionWithGemini(dataUrl) {
  if (!window.GEMINI_API_KEY) throw new Error("Missing GEMINI_API_KEY");
  const base64 = dataUrl.split(",")[1];
  const payload = {
    contents: [
      {
        parts: [
          {
            text: "Generate a concise caption (<= 20 words) describing this item for a Lost & Found database.",
          },
          { inline_data: { mime_type: "image/jpeg", data: base64 } },
        ],
      },
    ],
  };
  const resp = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" +
      window.GEMINI_API_KEY,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!resp.ok)
    throw new Error(`Gemini error: ${resp.status} – ${await resp.text()}`);
  const json = await resp.json();
  const text = json?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
  return text || "An item (caption unavailable).";
}
function ImageCaptioner({ onCaption }) {
  const [preview, setPreview] = useState("");
  const [caption, setCaption] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef(null);
  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = await readFileAsDataURL(file);
    setPreview(url);
    setCaption("");
  }
  async function generate() {
    if (!preview) return;
    setBusy(true);
    try {
      let text;
      if (window.GEMINI_API_KEY) {
        text = await captionWithGemini(preview);
      } else {
        const name = fileRef.current?.files?.[0]?.name || "";
        text = await localCaption(preview, name);
      }
      setCaption(text);
      onCaption?.({ imageUrl: preview, caption: text });
    } catch (err) {
      console.error(err);
      setCaption("Caption failed. Check console / API key.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <Card>
      <h3 style={{ marginTop: 0 }}>Photo → Auto Caption</h3>
      <input type="file" accept="image/*" ref={fileRef} onChange={handleFile} />
      {preview && (
        <img
          src={preview}
          alt="preview"
          style={{
            width: "100%",
            maxWidth: 360,
            height: 200,
            objectFit: "cover",
            borderRadius: 12,
            border: "1px solid #e5e7eb",
            marginTop: 8,
          }}
        />
      )}
      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <Button onClick={generate} disabled={!preview || busy}>
          {busy ? "Generating…" : "Generate caption"}
        </Button>
        {window.GEMINI_API_KEY ? (
          <span style={{ fontSize: 12, color: "#065f46" }}>
            Using Gemini API
          </span>
        ) : (
          <span style={{ fontSize: 12, color: "#6b7280" }}>
            No API key → local heuristic
          </span>
        )}
      </div>
      {caption && (
        <div
          style={{
            fontSize: 14,
            border: "1px solid #e5e7eb",
            borderRadius: 10,
            padding: 8,
            background: "#f8fafc",
            marginTop: 8,
          }}
        >
          <b>Caption:</b> {caption}
        </div>
      )}
    </Card>
  );
}

/** ======= data + utilities ======= **/
const COLORS = [
  "black",
  "white",
  "gray",
  "silver",
  "gold",
  "red",
  "blue",
  "green",
  "yellow",
  "orange",
  "purple",
  "pink",
  "brown",
  "beige",
];
const CATEGORIES = [
  "bag",
  "backpack",
  "wallet",
  "phone",
  "laptop",
  "tablet",
  "keys",
  "card",
  "headphones",
  "glasses",
  "jacket",
  "umbrella",
];
const MATERIALS = [
  "leather",
  "canvas",
  "nylon",
  "plastic",
  "metal",
  "cotton",
  "polyester",
];
const BASE_WEIGHTS = {
  ...Object.fromEntries(COLORS.map((c) => [c, 2.0])),
  ...Object.fromEntries(CATEGORIES.map((c) => [c, 2.5])),
  ...Object.fromEntries(MATERIALS.map((m) => [m, 1.5])),
  train: 1,
  subway: 1,
  station: 1,
  bus: 1,
  stop: 1,
  seat: 0.7,
  brand: 0.8,
  zipper: 0.6,
  pocket: 0.6,
  strap: 0.6,
  screen: 0.6,
  iphone: 1.5,
  samsung: 1.5,
  macbook: 1.5,
  airpods: 1.4,
  pixel: 1.4,
};
function tokenize(s) {
  return (s || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}
function embed(text) {
  const counts = {};
  tokenize(text).forEach((t) => (counts[t] = (counts[t] || 0) + 1));
  const vec = {};
  for (const [k, w] of Object.entries(BASE_WEIGHTS))
    if (counts[k]) vec[k] = counts[k] * w;
  return vec;
}
function cosineSim(a, b) {
  let dot = 0,
    na = 0,
    nb = 0;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    const av = a[k] || 0,
      bv = b[k] || 0;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
function extractTags({
  description = "",
  category = "",
  color = "",
  location = "",
  material = "",
}) {
  const toks = tokenize(description);
  const foundColor = color || COLORS.find((c) => toks.includes(c)) || "";
  const foundCategory =
    category || CATEGORIES.find((c) => toks.includes(c)) || "";
  const foundMaterial =
    material || MATERIALS.find((m) => toks.includes(m)) || "";
  const loc =
    location ||
    (description
      .match(/\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\b/g)
      ?.slice(-1)[0] ??
      "");
  return {
    color: foundColor,
    category: foundCategory,
    location: loc,
    material: foundMaterial,
  };
}
function matchesFilter(item, required) {
  const req = Object.fromEntries(Object.entries(required).filter(([, v]) => v));
  return Object.entries(req).every(
    ([k, v]) => (item.tags?.[k] || "").toLowerCase() === v.toLowerCase()
  );
}
function prettyTags(tags) {
  return Object.entries(tags || {})
    .filter(([, v]) => v)
    .map(([k, v]) => `${k}: ${v}`);
}

// ---- Seed inventory ----
const SEED_ITEMS = [
  {
    id: "F-001",
    imageUrl:
      "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=800&q=80",
    title: "Black leather wallet",
    description:
      "Black leather wallet with zipper found on the 7 train near Queensboro Plaza.",
    locationFound: "Queensboro Plaza",
    timeFound: "2025-10-10 18:45",
    tags: {
      color: "black",
      category: "wallet",
      material: "leather",
      location: "Queensboro Plaza",
    },
  },
  {
    id: "F-002",
    imageUrl:
      "https://images.unsplash.com/photo-1547949003-9792a18a2601?w=800&q=80",
    title: "Blue nylon backpack",
    description:
      "Blue nylon backpack, small tear on strap, found at Times Square station.",
    locationFound: "Times Square",
    timeFound: "2025-10-11 09:10",
    tags: {
      color: "blue",
      category: "backpack",
      material: "nylon",
      location: "Times Square",
    },
  },
  {
    id: "F-003",
    imageUrl:
      "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=800&q=80",
    title: "Silver MacBook laptop",
    description: "Silver MacBook left on bus M15, front seat.",
    locationFound: "Lower East Side",
    timeFound: "2025-10-09 20:05",
    tags: {
      color: "silver",
      category: "laptop",
      material: "metal",
      location: "Lower East Side",
    },
  },
  {
    id: "F-004",
    imageUrl:
      "https://images.unsplash.com/photo-1512314889357-e157c22f938d?w=800&q=80",
    title: "Brown canvas messenger bag",
    description:
      "Brown canvas messenger bag found on the Red Line platform at South Station.",
    locationFound: "South Station",
    timeFound: "2025-10-07 14:20",
    tags: {
      color: "brown",
      category: "bag",
      material: "canvas",
      location: "South Station",
    },
  },
  {
    id: "F-005",
    imageUrl:
      "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800&q=80",
    title: "White wireless earbuds",
    description:
      "White earbuds (likely AirPods) found at 34 St - Penn Station concourse.",
    locationFound: "Penn Station",
    timeFound: "2025-10-12 12:00",
    tags: {
      color: "white",
      category: "headphones",
      material: "plastic",
      location: "Penn Station",
    },
  },
];

// =====================================================
// Stubbed API — shape compatible with future Firebase functions
// =====================================================
const api = {
  async createReport({ description, where_lost, when_lost, tags, photo_url }) {
    const report_id = "R-" + Math.random().toString(36).slice(2, 7);
    const embedding = embed(
      [description, Object.values(tags || {}).join(" ")].join(" ")
    );
    await wait(200);
    return {
      report_id,
      tags: tags || {},
      embedding_preview: Object.values(embedding).slice(0, 6),
      photo_url,
    };
  },
  async searchPrefilter(tags, allItems) {
    await wait(120);
    const filtered = (allItems || []).filter((it) =>
      matchesFilter(it, tags || {})
    );
    return {
      items: filtered.map(({ id, title, description, imageUrl, tags }) => ({
        item_id: id,
        title,
        description,
        image_url: imageUrl,
        tags,
      })),
    };
  },
  async rankCandidates({ reportDescription, candidateItems }) {
    await wait(150);
    const qvec = embed(reportDescription);
    const ranked = (candidateItems || [])
      .map((ci) => ({
        item_id: ci.item_id,
        score: cosineSim(
          qvec,
          embed(
            [
              ci.title,
              ci.description,
              Object.values(ci.tags || {}).join(" "),
            ].join(" ")
          )
        ),
      }))
      .sort((a, b) => b.score - a.score);
    return { ranked };
  },
  async createClaim({ report_id, item_id, contact, secretDetail }) {
    await wait(160);
    return {
      claim_id: "C-" + Math.random().toString(36).slice(2, 7),
      status: "open",
      report_id,
      item_id,
      contact,
      secretDetail,
    };
  },
};
function wait(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

/** ======= Reporter Chat (unchanged flow, but now can consume auto-caption) ======= **/
const CHAT_QUESTIONS = [
  {
    key: "category",
    q: "What kind of item is it? (backpack, wallet, phone)",
    placeholder: "backpack",
  },
  { key: "color", q: "What color is it?", placeholder: "blue" },
  { key: "brand", q: "Any brand or logo?", placeholder: "Nike / no brand" },
  {
    key: "material",
    q: "Material (leather, canvas, nylon, metal)?",
    placeholder: "nylon",
  },
  {
    key: "location",
    q: "Where did you last see it? (station/stop/line)",
    placeholder: "Times Square station",
  },
  {
    key: "details",
    q: "Any distinct features?",
    placeholder: "small tear on strap",
  },
  {
    key: "when",
    q: "Rough time lost? (YYYY-MM-DD HH:mm)",
    placeholder: "2025-10-11 09:00",
  },
];
function ReporterChat({ onComplete, autoCaption }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi! I’ll ask a few questions to describe your item.",
    },
  ]);
  const [idx, setIdx] = useState(0);
  const [answers, setAnswers] = useState({});
  const inputRef = useRef(null);
  useEffect(() => {
    setMessages((m) => [
      ...m,
      { role: "assistant", text: CHAT_QUESTIONS[0].q },
    ]);
  }, []);
  function buildDescription(a) {
    const parts = [
      a.color && a.category
        ? `${a.color} ${a.category}`
        : a.category || a.color || "item",
      a.brand && a.brand !== "no brand" ? `brand ${a.brand}` : "",
      a.material ? `made of ${a.material}` : "",
      a.details ? `with ${a.details}` : "",
      a.location ? `lost around ${a.location}` : "",
      a.when ? `on ${a.when}` : "",
    ].filter(Boolean);
    return parts.join(", ");
  }
  async function submitAnswer(val) {
    const q = CHAT_QUESTIONS[idx];
    const v = (val || "").trim();
    if (!v) return;
    setMessages((m) => [...m, { role: "user", text: v }]);
    const next = { ...answers, [q.key]: v };
    setAnswers(next);
    const nextIdx = idx + 1;
    setIdx(nextIdx);
    if (nextIdx < CHAT_QUESTIONS.length) {
      setTimeout(
        () =>
          setMessages((m) => [
            ...m,
            { role: "assistant", text: CHAT_QUESTIONS[nextIdx].q },
          ]),
        200
      );
    } else {
      // ——— INTEGRATION POINT ———
      // Combine auto-caption (if any) with the structured description.
      const chatDesc = buildDescription(next);
      const description = autoCaption
        ? `${autoCaption}. ${chatDesc}`
        : chatDesc;
      const tags = extractTags({
        description,
        category: next.category,
        color: next.color,
        material: next.material,
        location: next.location,
      });
      setTimeout(
        () =>
          setMessages((m) => [
            ...m,
            { role: "assistant", text: "Thanks! Searching for matches…" },
          ]),
        200
      );
      const { report_id } = await api.createReport({
        description,
        where_lost: next.location,
        when_lost: next.when,
        tags,
      });
      const pre = await api.searchPrefilter(tags, window.__ITEMS__ || []);
      const ranked = await api.rankCandidates({
        reportDescription: description,
        candidateItems: pre.items,
      });
      onComplete({ report_id, description, tags, ranked: ranked.ranked });
    }
    if (inputRef.current) inputRef.current.value = "";
  }
  return (
    <Card style={{ height: 520, display: "flex", flexDirection: "column" }}>
      <h3>Describe Your Lost Item</h3>
      {autoCaption && (
        <div style={{ fontSize: 12, color: "#065f46", marginBottom: 6 }}>
          Auto‑caption added to your description: <em>{autoCaption}</em>
        </div>
      )}
      <div
        style={{
          flex: 1,
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 6,
          paddingRight: 6,
          marginTop: 6,
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              justifyContent: m.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                maxWidth: "85%",
                padding: 8,
                borderRadius: 12,
                background: m.role === "user" ? "#111827" : "#f3f4f6",
                color: m.role === "user" ? "white" : "#111827",
              }}
            >
              {m.text}
            </div>
          </div>
        ))}
      </div>
      {idx < CHAT_QUESTIONS.length ? (
        <div style={{ display: "flex", gap: 8 }}>
          <Input
            ref={inputRef}
            placeholder={CHAT_QUESTIONS[idx].placeholder}
            onKeyDown={(e) => {
              if (e.key === "Enter") submitAnswer(e.currentTarget.value);
            }}
          />
          <Button onClick={() => submitAnswer(inputRef.current?.value || "")}>
            Send
          </Button>
        </div>
      ) : (
        <div style={{ fontSize: 12, color: "#6b7280" }}>
          Collecting matches…
        </div>
      )}
    </Card>
  );
}

/** ======= Results list & Claim Drawer ======= **/
function ItemCard({ item, onDelete }) {
  return (
    <Card style={{ overflow: "hidden" }}>
      <img
        src={item.imageUrl}
        alt={item.title}
        style={{
          width: "100%",
          height: 160,
          objectFit: "cover",
          borderRadius: 12,
        }}
      />
      <div style={{ marginTop: 8 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <div style={{ fontWeight: 600 }}>{item.title}</div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              {item.locationFound} · {item.timeFound}
            </div>
          </div>
          {onDelete && (
            <Button
              onClick={() => onDelete(item.id)}
              style={{
                background: "#f3f4f6",
                color: "#111827",
                border: "1px solid #e5e7eb",
              }}
            >
              Delete
            </Button>
          )}
        </div>
        <p style={{ fontSize: 14, marginTop: 6 }}>{item.description}</p>
        <div
          style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 6 }}
        >
          {prettyTags(item.tags).map((t) => (
            <Badge key={t}>{t}</Badge>
          ))}
        </div>
      </div>
    </Card>
  );
}
function RankMatches({
  query,
  ranked,
  threshold,
  pageSize = 5,
  onPick,
  onShowMore,
}) {
  const visible = (ranked || []).filter((r) => r.score >= threshold);
  const pageCount = Math.ceil(visible.length / pageSize);
  const slice = (page) => visible.slice(0, (page + 1) * pageSize);
  return (
    <Card style={{ height: 520, display: "flex", flexDirection: "column" }}>
      <h3>Top Matches</h3>
      {query && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 14 }}>
            <b>Query:</b> {query.description}
          </div>
          <div
            style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 6 }}
          >
            {prettyTags(query.tags).map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
          </div>
        </div>
      )}
      <div
        style={{
          flex: 1,
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        {visible.length === 0 && (
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            No matches above threshold.
          </div>
        )}
        {slice(query?.page || 0).map(({ item_id, score }) => (
          <div key={item_id}>
            <ItemCard
              item={
                (window.__ITEMS__ || []).find((i) => i.id === item_id) || {
                  imageUrl: "",
                  title: item_id,
                  description: "",
                  locationFound: "",
                  timeFound: "",
                  tags: {},
                }
              }
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: 6,
              }}
            >
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Similarity: {(score * 100).toFixed(1)}%
              </div>
              <Button onClick={() => onPick?.(item_id)}>
                This looks like mine
              </Button>
            </div>
          </div>
        ))}
      </div>
      {query && (query.page || 0) < pageCount - 1 && (
        <Button style={{ marginTop: 8 }} onClick={onShowMore}>
          Show next 5
        </Button>
      )}
    </Card>
  );
}
function ClaimDrawer({ open, onClose, item, reportId, onSubmit }) {
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [secret, setSecret] = useState("");
  const [submitting, setSubmitting] = useState(false);
  if (!open) return null;
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.4)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 50,
      }}
    >
      <Card style={{ width: 520 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <h3 style={{ margin: 0 }}>Claim this item</h3>
          <Button style={{ background: "#6b7280" }} onClick={onClose}>
            Close
          </Button>
        </div>
        <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
          <img
            src={item?.imageUrl}
            alt={item?.title}
            style={{
              width: 140,
              height: 100,
              objectFit: "cover",
              borderRadius: 12,
            }}
          />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600 }}>{item?.title}</div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              {item?.description}
            </div>
            <div
              style={{
                display: "flex",
                gap: 8,
                flexWrap: "wrap",
                marginTop: 6,
              }}
            >
              {prettyTags(item?.tags || {}).map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
          </div>
        </div>
        <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
          <Input
            placeholder="Your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Input
            placeholder="Email or phone"
            value={contact}
            onChange={(e) => setContact(e.target.value)}
          />
          <Textarea
            placeholder="Ownership question: e.g., 'What is inside the front pocket?'"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
          />
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            gap: 8,
            marginTop: 12,
          }}
        >
          <Button style={{ background: "#6b7280" }} onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={submitting}
            onClick={async () => {
              setSubmitting(true);
              await onSubmit?.({
                report_id: reportId,
                item_id: item?.id,
                contact: `${name} | ${contact}`,
                secretDetail: secret,
              });
              setSubmitting(false);
              onClose();
            }}
          >
            {submitting ? "Submitting…" : "Submit Claim"}
          </Button>
        </div>
      </Card>
    </div>
  );
}

/** ======= App (now wires Auto-Caption into search pipeline) ======= **/
export default function App() {
  const [items, setItems] = useState(SEED_ITEMS);
  const [queryState, setQueryState] = useState(null); // { report_id, description, tags, ranked, page }
  const [threshold, setThreshold] = useState(0.2);
  const [claimOpen, setClaimOpen] = useState(false);
  const [claimItem, setClaimItem] = useState(null);
  const [captionState, setCaptionState] = useState({
    imageUrl: "",
    caption: "",
  });

  // expose inventory for child components (used by RankMatches)
  window.__ITEMS__ = items;

  function addItem(it) {
    setItems((prev) => [it, ...prev]);
  }
  function deleteItem(id) {
    setItems((prev) => prev.filter((x) => x.id !== id));
  }
  function handleComplete({ report_id, description, tags, ranked }) {
    setQueryState({ report_id, description, tags, ranked, page: 0 });
  }
  function handleShowMore() {
    setQueryState((q) => ({ ...q, page: (q.page || 0) + 1 }));
  }
  function handlePick(item_id) {
    const it = items.find((x) => x.id === item_id);
    setClaimItem(it);
    setClaimOpen(true);
  }
  async function submitClaim({ report_id, item_id, contact, secretDetail }) {
    const res = await api.createClaim({
      report_id,
      item_id,
      contact,
      secretDetail,
    });
    alert(`Claim submitted!\nClaim ID: ${res.claim_id}\nStatus: ${res.status}`);
  }

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <div>
          <h1 style={{ margin: 0 }}>Lost &amp; Found AI — Demo</h1>
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            Photo caption → Chat intake → Tag filter → Cosine ranking → Claims
          </div>
        </div>
        <Button
          onClick={() => {
            setQueryState(null);
            setClaimItem(null);
            setClaimOpen(false);
            setCaptionState({ imageUrl: "", caption: "" });
          }}
        >
          Reset Session
        </Button>
      </header>

      <div className="grid2" style={{ marginTop: 12 }}>
        {/* LEFT COLUMN: Reporter Chat now shows the auto-caption inline */}
        <ReporterChat
          onComplete={handleComplete}
          autoCaption={captionState.caption}
        />

        {/* RIGHT COLUMN: Controls + Results */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <div>
                Match threshold <Badge>{Math.round(threshold * 100)}%</Badge>
              </div>
              <input
                type="range"
                min="0"
                max="0.8"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: 200 }}
              />
            </div>
          </Card>
          <RankMatches
            query={queryState}
            ranked={queryState?.ranked || []}
            threshold={threshold}
            pageSize={5}
            onPick={handlePick}
            onShowMore={handleShowMore}
          />
        </div>
      </div>

      {/* NEW ROW: Photo → Auto Caption panel */}
      <h2 style={{ marginTop: 24 }}>Add a Photo (Optional)</h2>
      <div className="grid2">
        <ImageCaptioner
          onCaption={({ imageUrl, caption }) =>
            setCaptionState({ imageUrl, caption })
          }
        />
        <Card>
          <h3>How it's used</h3>
          <p style={{ marginTop: 6, fontSize: 14 }}>
            The generated caption is automatically prepended to your chat
            description when we search for matches. If you skip the chat, the
            caption alone is still submitted as your description.
          </p>
          {captionState.caption ? (
            <div
              style={{
                fontSize: 13,
                background: "#f8fafc",
                border: "1px solid #e5e7eb",
                padding: 8,
                borderRadius: 8,
              }}
            >
              <b>Current auto‑caption:</b> {captionState.caption}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              No caption yet.
            </div>
          )}
        </Card>
      </div>

      <h2 style={{ marginTop: 24 }}>Intake (Staff)</h2>
      <div className="grid2">
        <StaffIntake onAdd={addItem} />
        <Card>
          <h3>Inventory</h3>
          <div className="grid3">
            {items.map((it) => (
              <ItemCard key={it.id} item={it} onDelete={deleteItem} />
            ))}
          </div>
        </Card>
      </div>

      <footer style={{ marginTop: 24, fontSize: 12, color: "#6b7280" }}>
        Demo only · Local tag filter + cosine similarity · Replace stubs with
        Firebase/Gemini later.
      </footer>

      <ClaimDrawer
        open={claimOpen}
        onClose={() => setClaimOpen(false)}
        item={claimItem}
        reportId={queryState?.report_id}
        onSubmit={submitClaim}
      />
    </div>
  );
}

/** ======= Staff intake (unchanged) ======= **/
function StaffIntake({ onAdd }) {
  const [form, setForm] = useState({
    title: "",
    description: "",
    locationFound: "",
    timeFound: "",
    imageUrl: "",
    color: "",
    category: "",
    material: "",
  });
  const tags = extractTags({
    description: form.description,
    color: form.color,
    category: form.category,
    material: form.material,
    location: form.locationFound,
  });
  function submit() {
    if (!form.title || !form.description) return;
    const item = {
      id: "F-" + Math.random().toString(36).slice(2, 7),
      imageUrl:
        form.imageUrl ||
        "https://images.unsplash.com/photo-1512314889357-e157c22f938d?w=800&q=80",
      title: form.title,
      description: form.description,
      locationFound: form.locationFound || "",
      timeFound:
        form.timeFound ||
        new Date().toISOString().slice(0, 16).replace("T", " "),
      tags,
    };
    onAdd(item);
    setForm({
      title: "",
      description: "",
      locationFound: "",
      timeFound: "",
      imageUrl: "",
      color: "",
      category: "",
      material: "",
    });
  }
  return (
    <Card>
      <h3 style={{ marginBottom: 8 }}>Lost &amp; Found Intake</h3>
      <div className="grid2">
        <Input
          placeholder="Title"
          value={form.title}
          onChange={(e) => setForm((f) => ({ ...f, title: e.target.value }))}
        />
        <Input
          placeholder="Image URL (optional)"
          value={form.imageUrl}
          onChange={(e) => setForm((f) => ({ ...f, imageUrl: e.target.value }))}
        />
        <Input
          placeholder="Location found"
          value={form.locationFound}
          onChange={(e) =>
            setForm((f) => ({ ...f, locationFound: e.target.value }))
          }
        />
        <Input
          placeholder="Time found (YYYY-MM-DD HH:mm)"
          value={form.timeFound}
          onChange={(e) =>
            setForm((f) => ({ ...f, timeFound: e.target.value }))
          }
        />
        <Input
          placeholder="Color (optional)"
          value={form.color}
          onChange={(e) => setForm((f) => ({ ...f, color: e.target.value }))}
        />
        <Input
          placeholder="Category (optional)"
          value={form.category}
          onChange={(e) => setForm((f) => ({ ...f, category: e.target.value }))}
        />
        <Input
          placeholder="Material (optional)"
          value={form.material}
          onChange={(e) => setForm((f) => ({ ...f, material: e.target.value }))}
        />
        <Textarea
          placeholder="Short description"
          value={form.description}
          onChange={(e) =>
            setForm((f) => ({ ...f, description: e.target.value }))
          }
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 8,
        }}
      >
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {prettyTags(tags).map((t) => (
            <Badge key={t}>{t}</Badge>
          ))}
        </div>
        <Button onClick={submit}>Save Item</Button>
      </div>
    </Card>
  );
}

// Minimal layout helpers (CodeSandbox friendly)
const style = document.createElement("style");
style.innerHTML = `
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji"; background: #f8fafc; }
  h1,h2,h3 { margin: 6px 0; }
  .grid2 { display: grid; grid-template-columns: 1fr; gap: 12px; }
  .grid3 { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px,1fr)); gap: 12px; }
  @media (min-width: 996px){ .grid2 { grid-template-columns: 1fr 1fr; } }
`;
document.head.appendChild(style);
