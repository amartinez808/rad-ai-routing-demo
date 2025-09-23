import base64
import csv
import hashlib
import html
import json
import os
import random
import textwrap
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

def ensure_dir(p): os.makedirs(os.path.dirname(p), exist_ok=True)


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


def est_tokens(txt: str) -> int:
    # simple ~4 chars/token heuristic
    return max(1, int(len(txt) / 4))


def est_cost(route: str, in_tokens: int, out_tokens: int) -> float:
    # heuristic prices (adjust if app already has pricing):
    pricing = {
        "FAST": {"in": 0.15 / 1e6, "out": 0.60 / 1e6},  # cheap model family
        "REASONING": {"in": 2.50 / 1e6, "out": 10.00 / 1e6},  # expensive reasoning
    }
    p = pricing.get(route, pricing["FAST"])
    return round(in_tokens * p["in"] + out_tokens * p["out"], 6)


LOG_PATH = "data/routing_log.csv"


def log_decision(row: Dict[str, Any]):
    ensure_dir(LOG_PATH)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_recent(n=50):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


def determine_route(user_text: str, provider: str) -> str:
    lowered = user_text.lower()
    if len(user_text.strip()) > 260 or any(k in lowered for k in ["analyze", "explain", "strategy", "plan", "reason"]):
        return "REASONING"
    if provider in {"Together", "Grok"}:
        return "REASONING"
    return "FAST"


def generate_stubbed_response(provider: str, route: str, prompt: str) -> str:
    snippet = textwrap.shorten(prompt.strip().replace("\n", " "), width=140, placeholder="…") if prompt else ""
    lines = [
        f"{provider} ({route}) summary:",
        "- Evaluated prompt goals and matched to historical routing wins.",
        f"- First impression: '{snippet}'",
        "- Recommendation: FAST handles concise asks; REASONING keeps nuance.",
        "- Outcome: deliver actionable answer with cost awareness.",
    ]
    if route == "FAST":
        lines.append("- Savings: ~3-5x cheaper than reasoning models for this prompt.")
    else:
        lines.append("- Rationale: needs multi-hop reasoning; higher cost but better quality.")
    return "\n".join(lines)


def load_svg(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def provider_badge(provider: str, model: str) -> str:
    return f"""
    <span style="display:inline-flex;align-items:center;gap:8px;
      padding:4px 10px;border:1px solid var(--border,#2a2a2a);
      border-radius:999px;font-size:12px;line-height:1;">
      <strong style="font-weight:600">{provider.title()}</strong>
      <span style="opacity:.7">·</span>
      <span style="opacity:.8">{model}</span>
    </span>
    """

# =========================
# App Config
# =========================
st.set_page_config(page_title="RAD AI – LLM Metasearch (Kayak Demo)", page_icon="🧭", layout="centered")

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

# -------------------------
# Branding / Theme CSS
# -------------------------
st.markdown(
    """
<style>
.badge {display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .6rem; border-radius:999px; font-weight:600; font-size:.85rem;}
.pick {background:#0ea5e9; color:white;}
.alt {background:#e2e8f0; color:#0f172a;}
.chip {display:inline-block; padding:.25rem .5rem; background:#f1f5f9; border-radius:999px; font-size:.8rem; margin-right:.35rem; color:#0f172a;}
.chip-primary {background:#2563eb; color:#f8fafc; box-shadow:0 4px 12px rgba(37,99,235,.25);} 
.row {display:flex; gap:.75rem; align-items:flex-start; margin-bottom:.35rem;}
.bubble {background:white; border:1px solid #e5e7eb; border-radius:14px; padding:.6rem .8rem; box-shadow:0 1px 2px rgba(0,0,0,.04);}
.avatar {width:38px; height:38px; border-radius:999px; display:flex; align-items:center; justify-content:center; font-weight:700; color:white; padding:4px; transition: transform .2s ease;}
.avatar:hover {transform: translateY(-1px) rotate(-2deg);} 
.logo {width:30px; height:30px; display:block; object-fit:contain; background:white; border-radius:6px;}
.sm {font-size:.85rem; color:#475569;}
.fade {animation: fade .5s ease-in-out;}
@keyframes fade {from{opacity:0; transform:translateY(4px);} to{opacity:1; transform:none;}}
.run-card {padding:1rem; border-radius:16px; border:1px solid #cbd5f5; background:linear-gradient(145deg,#f8fafc,#eef2ff); box-shadow:0 8px 20px rgba(59,130,246,.1); color:#0f172a;}
.run-card strong, .run-card b {color:#0f172a;}
.run-card p, .run-card li {color:#0f172a;}
.run-card ul {margin-left:1.1rem; padding-left:0.2rem;}
.run-card li {margin-bottom:.3rem;}
.run-card br {line-height:1.6;}
.stMarkdown h3 {color:#e2e8f0;}
.council-overlay {position: fixed; inset: 0; display: grid; place-items: center; background: radial-gradient(1200px 600px at 50% -10%, rgba(2,6,23,.28), rgba(2,6,23,.72)); backdrop-filter: blur(2px); z-index: 9999;}
.council-modal {width: min(520px, 92vw); border-radius: 20px; padding: 24px; background: linear-gradient(145deg, #0b1220, #0e1626); border: 1px solid rgba(148,163,184,.18); box-shadow: 0 16px 40px rgba(0,0,0,.35); color: #e5e7eb; text-align: center;}
.council-orbit {position: relative; height: 150px; margin: 10px 0 6px;}
.council-orbit .av {position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);}
.council-orbit .av img {width: 36px; height: 36px; border-radius: 10px; background:#fff; object-fit: contain; box-shadow: 0 6px 16px rgba(0,0,0,.25); transition: transform .2s ease, box-shadow .2s ease;}
.council-orbit .fallback {display:flex; align-items:center; justify-content:center; width:36px; height:36px; border-radius:10px; background:#1f2937; color:#e2e8f0; font-weight:700; box-shadow:0 6px 16px rgba(0,0,0,.25);}
.av.active img {transform: scale(1.18) translateZ(0); box-shadow: 0 10px 22px rgba(14,165,233,.35);}
.av.p0 {transform: translate(-50%,-50%) translate(0,-52px);}
.av.p1 {transform: translate(-50%,-50%) translate(46px,-16px);}
.av.p2 {transform: translate(-50%,-50%) translate(28px,40px);}
.av.p3 {transform: translate(-50%,-50%) translate(-28px,40px);}
.av.p4 {transform: translate(-50%,-50%) translate(-46px,-16px);}
.typing {display:inline-block; letter-spacing:.15em;}
.typing span {animation: blink 1.2s infinite;}
.typing span:nth-child(2){ animation-delay:.2s; }
.typing span:nth-child(3){ animation-delay:.4s; }
@keyframes blink {0%,20%{ opacity:0;} 50%{opacity:1;} 100%{opacity:0;} }
.qline {font-size:.95rem; color:#93a4b8; min-height: 24px;}
.badge-sync {display:inline-flex; gap:.6rem; align-items:center; padding:.35rem .7rem; border-radius:999px; background:#0ea5e933; color:#bae6fd; border:1px solid #38bdf8; font-weight:600; font-size:.9rem; justify-content:center;}
</style>
""",
    unsafe_allow_html=True,
)

# ---- Real logos: local -> download/cache -> upload fallback ----
LOGO_MAP = {
    "OpenAI":  {"filename": "openai",   "urls": ["https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_logo_2025.svg"],  "bg": "#6d28d9"},
    "Gemini":  {"filename": "gemini",   "urls": ["https://upload.wikimedia.org/wikipedia/commons/4/4f/Google_Gemini_icon_2025.svg"], "bg": "#2563eb"},
    "Grok":    {"filename": "grok",     "urls": ["https://upload.wikimedia.org/wikipedia/commons/9/9c/Groq_logo.svg"],         "bg": "#ef4444"},
    "Llama":   {"filename": "llama",    "urls": ["https://custom.typingmind.com/tools/model-icons/llama/llama.svg"],            "bg": "#16a34a"},
    "Together":{"filename": "together", "urls": ["https://custom.typingmind.com/tools/model-icons/together/together.svg"],       "bg": "#0ea5e9"},
}

def _mime_for_extension(ext: str) -> str:
    ext = ext.lower()
    if ext == ".svg":
        return "image/svg+xml"
    if ext in {".png", ".apng"}:
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def _encode_bytes(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _filename_with_ext(base: str, content_type: str) -> str:
    sanitized_base = base.rsplit(".", 1)[0]
    lowered = (content_type or "").lower()
    if "svg" in lowered:
        return f"{sanitized_base}.svg"
    if "png" in lowered:
        return f"{sanitized_base}.png"
    if "jpeg" in lowered or "jpg" in lowered:
        return f"{sanitized_base}.jpg"
    if "webp" in lowered:
        return f"{sanitized_base}.webp"
    return f"{sanitized_base}.png"


def _detect_logo_format(data: bytes) -> Optional[Tuple[str, str]]:
    stripped = data.lstrip()
    if stripped.startswith(b"<svg"):
        return "image/svg+xml", ".svg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg", ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    return None


@st.cache_data(show_spinner=False)
def _load_logo_data(provider: str) -> str:
    """Return a data URI for the provider logo. Tries local file, then first good download URL."""

    meta = LOGO_MAP.get(provider)
    if not meta:
        return ""

    base_name = meta["filename"]
    preferred_path = ASSETS_DIR / base_name
    candidate: Optional[Path] = None

    if preferred_path.exists() and preferred_path.is_file():
        candidate = preferred_path
    else:
        for path in sorted(ASSETS_DIR.glob(f"{base_name}.*")):
            if path.is_file():
                candidate = path
                break
        if not candidate and provider.lower() == "grok":
            for path in sorted(ASSETS_DIR.glob("groq.*")):
                if path.is_file():
                    candidate = path
                    break

    if candidate:
        try:
            data = candidate.read_bytes()
        except Exception:
            candidate = None
        else:
            detected = _detect_logo_format(data)
            if detected:
                detected_mime, detected_ext = detected
                if candidate.suffix.lower() != detected_ext:
                    corrected_name = ASSETS_DIR / _filename_with_ext(base_name, detected_mime)
                    try:
                        corrected_name.write_bytes(data)
                        try:
                            candidate.unlink()
                        except Exception:
                            pass
                        candidate = corrected_name
                    except Exception:
                        pass
                return _encode_bytes(data, detected_mime)
            mime = _mime_for_extension(candidate.suffix)
            return _encode_bytes(data, mime)

    for url in meta.get("urls", []):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            raw_type = resp.headers.get("Content-Type") or ""
            content_type = raw_type.split(";")[0].strip().lower()
            if not content_type:
                content_type = "image/png"
            filename = _filename_with_ext(base_name, content_type)
            filepath = ASSETS_DIR / filename
            try:
                filepath.write_bytes(resp.content)
            except Exception:
                pass
            return _encode_bytes(resp.content, content_type)
        except Exception:
            continue

    return ""


def logo_img_html(provider: str, size: int = 30) -> str:
    meta = LOGO_MAP.get(provider, {})
    bg = meta.get("bg", "#64748b")
    data_uri = _load_logo_data(provider)
    if not data_uri:
        return f'<div class="avatar" style="background:{bg}">💬</div>'
    filter_style = ""
    img_background = "background:white;"
    if provider.lower() == "grok":
        img_background = "background:transparent;"
        filter_style = " filter: brightness(0) invert(1);"
    return f'''
      <div class="avatar" style="background:{bg}; padding:4px;">
        <img src="{data_uri}" alt="{provider} logo"
             class="logo" style="width:{size}px;height:{size}px;display:block;object-fit:contain;{img_background}border-radius:6px;{filter_style}">
      </div>
    '''


def show_loading_council(models: List[str], seconds: float = 2.4) -> None:
    """Display an overlay with orbiting provider avatars and quips."""

    display_models = [m for m in models if m][:5]
    if not display_models:
        return

    placeholder = st.empty()
    frame_interval = 0.2
    start = time.perf_counter()
    frame = 0
    positions = ["p0", "p1", "p2", "p3", "p4"]

    avatar_markup: List[str] = []
    for idx, provider in enumerate(display_models):
        data_uri = _load_logo_data(provider)
        if data_uri:
            img_tag = f'<img src="{data_uri}" alt="{provider} logo">'
        else:
            initial = html.escape(provider[:1].upper()) or "?"
            img_tag = f'<div class="fallback">{initial}</div>'
        avatar_markup.append(img_tag)

    quip_cycle = [QUIPS.get(provider, "") for provider in display_models]
    if not quip_cycle:
        quip_cycle = [""]

    typing_html = '<span class="typing"><span>.</span><span>.</span><span>.</span></span>'

    try:
        while time.perf_counter() - start < seconds:
            active_idx = frame % len(display_models)
            avatars_html = []
            for idx, provider in enumerate(display_models):
                pos_cls = positions[idx]
                active_cls = " active" if idx == active_idx else ""
                avatars_html.append(
                    f'<div class="av {pos_cls}{active_cls}">{avatar_markup[idx]}</div>'
                )

            quip_text = html.escape(quip_cycle[active_idx]) if quip_cycle[active_idx] else "&nbsp;"

            overlay_html = (
                '<div class="council-overlay">'
                '<div class="council-modal">'
                f'<div class="badge-sync">🧭 Council syncing {typing_html}</div>'
                '<div class="council-orbit">'
                + "".join(avatars_html)
                + '</div>'
                f'<div class="qline">{quip_text}</div>'
                '</div>'
                '</div>'
            )

            placeholder.markdown(overlay_html, unsafe_allow_html=True)
            time.sleep(frame_interval)
            frame += 1
    finally:
        placeholder.empty()

# Sidebar helper to upload logos if needed

# -------------------------
# Secrets helper / Live mode gates
# -------------------------

def _secret(key: str) -> Optional[str]:
    return st.secrets.get(key) or os.getenv(key)


def _grok_api_key() -> Optional[str]:
    return _secret("GROK_API_KEY") or _secret("GROQ_API_KEY")


HAVE_OPENAI = bool(_secret("OPENAI_API_KEY"))
HAVE_GROK = bool(_grok_api_key())
HAVE_TOGETHER = bool(_secret("TOGETHER_API_KEY"))
HAVE_GEMINI = bool(_secret("GEMINI_API_KEY"))
LIVE_CAPABLE = {
    "OpenAI": HAVE_OPENAI,
    "Grok": HAVE_GROK,
    "Together": HAVE_TOGETHER,
    "Gemini": HAVE_GEMINI,
}


def _escape_html(text: str) -> str:
    """HTML-escape and preserve newlines within our custom cards."""

    escaped = html.escape(text)
    return escaped.replace("\n", "<br>")


def render_simulated_card(provider: str, content: str) -> None:
    """Render the best-pick card with simulated content."""

    safe = _escape_html(content).replace("\n", "<br>")
    html_body = f"""
    <div class='run-card fade'>
      <p><strong>{provider} · simulated response</strong></p>
      <p>{safe}</p>
    </div>
    """
    st.markdown(html_body, unsafe_allow_html=True)


def render_live_card(output: str) -> None:
    """Render live model output inside the branded card."""

    safe = _escape_html(output)
    html_body = f"<div class='run-card fade'><p>{safe}</p></div>"
    st.markdown(html_body, unsafe_allow_html=True)

# -------------------------
# Simulated “council” logic
# -------------------------
MODELS = [
    ("OpenAI", "gpt-4o-mini", {"speed": 4, "reason": 4, "cost": 3}),
    ("Gemini", "gemini-2.0-flash", {"speed": 5, "reason": 3, "cost": 5}),
    ("Llama", "llama-3.1-8b", {"speed": 5, "reason": 3, "cost": 5}),
    ("Together", "llama-3.3-70b", {"speed": 3, "reason": 4, "cost": 5}),
    ("Grok", "llama-3.3-70b", {"speed": 5, "reason": 4, "cost": 4}),
]


QUIPS = {
    "OpenAI": "I can balance speed and reasoning for a clean final answer.",
    "Gemini": "I’m fast with webby tasks and structured responses.",
    "Llama": "Lean and quick—great for short prompts and drafts.",
    "Together": "Big-brain Llama 70B is solid for deeper takes.",
    "Grok": "Blazing low latency—let me snap a result together.",
}


# Lightweight heuristic to keep demo deterministic-ish but lively

def heuristic_vote(user_text: str) -> Tuple[str, str, List[Tuple[str, float, str]]]:
    """Return (winner_provider, winner_model, council_details[(name,score,quip)])."""

    length = len(user_text.strip())
    keywords = user_text.lower()
    base: List[Tuple[str, str, float]] = []

    for prov, model, attr in MODELS:
        score = 0.4 * attr["speed"] + 0.4 * attr["reason"] + 0.2 * attr["cost"]
        if length > 280 or any(k in keywords for k in ["analyze", "explain", "deep", "reason", "compare"]):
            score += 1.2 * attr["reason"]
        if length <= 120 or any(k in keywords for k in ["quick", "fast", "tl;dr", "summary"]):
            score += 0.8 * attr["speed"]
        if any(k in keywords for k in ["free", "budget", "cheap"]):
            score += 0.7 * attr["cost"]
        score += random.uniform(-0.8, 0.8)
        base.append((prov, model, score))

    base.sort(key=lambda x: x[2], reverse=True)
    winner_prov, winner_model, _ = base[0]

    council = [(p, round(s, 2), QUIPS[p]) for (p, _m, s) in base]
    return winner_prov, winner_model, council


# -------------------------
# Optional Live calls (used only if Live Mode toggled AND keys exist)
# -------------------------

def live_generate(provider_label: str, model: str, prompt: str, temperature: float = 0.2) -> str:
    pl = provider_label.lower()
    if "openai" in pl:
        from openai import OpenAI

        api_key = _secret("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Be concise and helpful."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    if "grok" in pl:
        api_key = _grok_api_key()
        if not api_key:
            raise RuntimeError("Missing GROK_API_KEY (or legacy GROQ_API_KEY)")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Be concise and helpful."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    if "together" in pl:
        api_key = _secret("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing TOGETHER_API_KEY")
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct-free",
            "messages": [
                {"role": "system", "content": "Be concise and helpful."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    if "gemini" in pl:
        import google.generativeai as genai

        api_key = _secret("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(prompt, generation_config={"temperature": temperature})
        return resp.text.strip()

    raise RuntimeError("Live mode: provider not supported yet.")


# =========================
# UI
# =========================
st.sidebar.subheader("Mode & Notes")
live = st.sidebar.toggle(
    "LIVE mode",
    value=False,
    help="If off, run against stubbed responses.",
)
sidebar_notes = st.sidebar.text_input("Run note (optional)", value="")

with st.sidebar.expander("About this demo", expanded=False):
    st.write("Vendor-agnostic router. This run uses **your OpenAI subscription** for API calls.")
    cols = st.columns(2)
    if Path("assets/openai.webp").exists():
        cols[0].image("assets/openai.webp", caption="OpenAI", width=72)
    claude_svg_sidebar = load_svg("assets/anthropic_claude.svg")
    if claude_svg_sidebar:
        cols[1].markdown(claude_svg_sidebar, unsafe_allow_html=True)
        cols[1].caption("Claude (Anthropic)")

st.title("🧭 RAD AI – Kayak for LLMs")

with st.container():
    st.markdown("### Supported providers")
    c1, c2 = st.columns(2)
    with c1:
        if Path("assets/openai.webp").exists():
            st.image("assets/openai.webp", caption="OpenAI", width=96)
        else:
            st.caption("OpenAI")
    with c2:
        claude_svg = load_svg("assets/anthropic_claude.svg")
        if claude_svg:
            st.markdown(
                f'<div style="height:28px;color:currentColor">{claude_svg}</div><div style="font-size:12px;opacity:.7">Claude (Anthropic)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Claude (Anthropic)")
    st.divider()

st.caption("Type once. Watch the LLM council confer. Get one **Best Pick** and simple alternates. (Visual demo first; live mode optional.)")

st.session_state.setdefault("alt_preview", "")
demo_mode = not live

prompt = st.text_area(
    "Your prompt",
    "In 5 bullets, pitch RAD AI (Rational Automation Design): problem, solution, why now, traction, ask.",
    height=140,
)

go = st.button("Search all models")

if go:
    st.session_state["alt_preview"] = ""
    overlay_models = [prov for prov, _model, _attr in MODELS]
    overlay_duration = 2.4
    overall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        vote_future = executor.submit(heuristic_vote, prompt)
        show_loading_council(overlay_models, seconds=overlay_duration)
        winner_prov, winner_model, council = vote_future.result()

    elapsed = time.perf_counter() - overall_start
    if elapsed < 1.8:
        time.sleep(1.8 - elapsed)

    st.markdown("#### 🤝 LLM Council")
    for prov, score, quip in council:
        logo_html = logo_img_html(prov)
        bubble_html = (
            f'<div class="bubble">'
            f'<div><b>{prov}</b> · <span class="sm">score {score}</span></div>'
            f'<div class="sm">{quip}</div>'
            f"</div>"
        )
        row_html = f'<div class="row fade">{logo_html}{bubble_html}</div>'
        st.markdown(row_html, unsafe_allow_html=True)
        time.sleep(0.12)

    st.divider()

    st.markdown('<span class="badge pick">Best Pick</span>', unsafe_allow_html=True)
    st.markdown(f"### {winner_prov} · **{winner_model}**")
    st.markdown(logo_img_html(winner_prov), unsafe_allow_html=True)

    feature_chips = ["Speed", "Reasoning", "Budget-friendly"]
    st.markdown(
        "".join([f'<span class="chip chip-primary">{c}</span>' for c in feature_chips]),
        unsafe_allow_html=True,
    )

    route = determine_route(prompt, winner_prov)
    st.caption(f"Routing decision: **{route}**")
    start_time = time.time()
    input_hash = hash_text(prompt)
    in_toks = est_tokens(prompt)
    result_text = ""
    latency_ms = 0

    live_possible = not demo_mode and LIVE_CAPABLE.get(winner_prov, False)
    if winner_prov == "Llama":
        live_possible = False  # Llama entry is simulated-only

    if not demo_mode and not live_possible:
        st.info("Live mode skipped: keys missing or provider not supported. Showing simulated result.")

    if demo_mode or not live_possible:
        result_text = generate_stubbed_response(winner_prov, route, prompt)
        render_simulated_card(winner_prov, result_text)
        latency_ms = int((time.time() - start_time) * 1000)
    else:
        try:
            with st.spinner(f"Running live on {winner_prov}…"):
                live_start = time.perf_counter()
                result_text = live_generate(winner_prov, winner_model, prompt, 0.2)
                live_latency_ms = int((time.perf_counter() - live_start) * 1000)
            latency_ms = int((time.time() - start_time) * 1000)
            st.success(f"Live result in {live_latency_ms:.0f} ms")
            render_live_card(result_text)
        except Exception as exc:  # Graceful failure
            error_msg = f"Live call failed: {exc}"
            st.error(error_msg)
            result_text = error_msg
            latency_ms = int((time.time() - start_time) * 1000)

    try:
        st.markdown(provider_badge(winner_prov, winner_model), unsafe_allow_html=True)
    except NameError:
        st.markdown(provider_badge("openai", "gpt-4o-mini"), unsafe_allow_html=True)

    out_toks = est_tokens(result_text)
    cost = est_cost(route, in_toks, out_toks)
    log_decision(
        {
            "ts_iso": dt.datetime.utcnow().isoformat() + "Z",
            "input_hash": input_hash,
            "route": route,
            "latency_ms": latency_ms,
            "in_tokens": in_toks,
            "out_tokens": out_toks,
            "est_cost_usd": cost,
            "notes": sidebar_notes,
        }
    )

    st.markdown("#### ✨ Good Alternates")
    alts = council[1:4]
    cols = st.columns(len(alts)) if alts else []
    for idx, (prov, _score, quip) in enumerate(alts):
        with cols[idx]:
            st.markdown(f'<span class="badge alt">{prov}</span>', unsafe_allow_html=True)
            st.markdown(logo_img_html(prov), unsafe_allow_html=True)
            st.caption(quip)
            if st.button(f"Preview {prov}", key=f"alt-{prov}"):
                preview = textwrap.dedent(
                    f"""
                    **{prov} (simulated take)**  
                    - Signal: {quip}  
                    - Would highlight alternate POV and give a second draft.
                    """
                )
                st.session_state["alt_preview"] = preview

    if st.session_state.get("alt_preview"):
        st.markdown(st.session_state["alt_preview"])

st.divider()
st.caption("Recent routing decisions")
rows = load_recent(50)
if rows:
    st.dataframe(list(reversed(rows)), use_container_width=True)
else:
    st.info("No logs yet. Run a few queries.")

st.divider()
st.markdown("##### Connection checklist")
st.write("- 🔑 OpenAI: `OPENAI_API_KEY` (paid)")
st.write("- 🆓 Grok: `GROK_API_KEY` (dev tier; legacy `GROQ_API_KEY` also works)")
st.write("- 🆓 Together: `TOGETHER_API_KEY` (llama-3.3-70b free endpoint)")
st.write("- 🆓 Gemini: `GEMINI_API_KEY` (free tier)")
st.caption("Simulated Mode requires no keys. Live Mode only runs if a key exists for the chosen provider.")
st.caption("Logos © their respective owners; used here for product identification in a demo UI.")
