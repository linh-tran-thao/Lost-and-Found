import React, { useMemo, useRef, useState } from "react";

// =====================================================
// Lost & Found AI — Single-file Demo (CodeSandbox ready)
// Adds:
// 1) "Show next 5" pagination for results
// 2) Minimal Claim Drawer (ownership verification)
// 3) Stubbed API calls (createReport / searchPrefilter / rankCandidates / createClaim)
//    so you can later swap to Firebase/Gemini without UI changes
// No external UI libraries required; simple inline styles for portability.
// =====================================================

/** ======= tiny UI helpers (no external UI libs) ======= **/
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
const Button = ({ children, onClick, style, disabled }) => (
  <button
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
const Input = (props) => (
  <input
    {...props}
    style={{
      ...props.style,
      padding: 8,
      borderRadius: 8,
      border: "1px solid #ddd",
      width: "100%",
    }}
  />
);
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

// ---- Seed data (inventory) ----
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
  async createReport({ description, where_lost, when_lost, tags }) {
    const report_id = "R-" + Math.random().toString(36).slice(2, 7);
    const embedding = embed(
      [description, Object.values(tags || {}).join(" ")].join(" ")
    );
    await wait(200); // simulate network
    return {
      report_id,
      tags: tags || {},
      embedding_preview: Object.values(embedding).slice(0, 6),
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

/** ======= components ======= **/
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

function ReporterChat({ onComplete }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi! I’ll ask a few questions to describe your item.",
    },
  ]);
  const [idx, setIdx] = useState(0);
  const [answers, setAnswers] = useState({});
  const inputRef = useRef(null);
  React.useEffect(() => {
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
      const description = buildDescription(next);
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
      // Use stubbed API pipeline
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

export default function App() {
  const [items, setItems] = useState(SEED_ITEMS);
  const [queryState, setQueryState] = useState(null); // { report_id, description, tags, ranked, page }
  const [threshold, setThreshold] = useState(0.2);
  const [claimOpen, setClaimOpen] = useState(false);
  const [claimItem, setClaimItem] = useState(null);

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
            Chat intake · Tag filter · Cosine similarity ranking · Claims
          </div>
        </div>
        <Button
          onClick={() => {
            setQueryState(null);
            setClaimItem(null);
            setClaimOpen(false);
          }}
        >
          Reset Session
        </Button>
      </header>

      <div className="grid2" style={{ marginTop: 12 }}>
        <ReporterChat onComplete={handleComplete} />
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
        Firebase Functions/Gemini later.
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
