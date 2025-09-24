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
    if provider in {"Together", "Grok", "Claude"}:
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

COUNCIL_CSS = """
<style>
.llm-chip{display:inline-flex;align-items:center;gap:.5rem;
  padding:.35rem .6rem;border:1px solid var(--border,#2a2a2a);
  border-radius:999px;font-size:.9rem}
.llm-chip img, .llm-chip svg{height:18px;width:18px}
.pulse{animation: pulse 1.2s ease-in-out infinite}
@keyframes pulse{0%{opacity:.4;transform:scale(.98)}50%{opacity:1;transform:scale(1)}100%{opacity:.4;transform:scale(.98)}}
</style>
"""

# =========================
# App Config
# =========================
st.set_page_config(page_title="RAD AI – LLM Metasearch (Kayak Demo)", page_icon="🧭", layout="wide")

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
.council-overlay {position: fixed; inset: 0; display: grid; place-items: center; background: rgba(8,15,35,.68); backdrop-filter: blur(8px); z-index: 9999; padding: 12px;}
.council-modal {width: min(560px, 92vw); border-radius: 24px; padding: 26px; background: linear-gradient(150deg, rgba(15,23,42,.96), rgba(30,41,59,.88)); border: 1px solid rgba(148,163,184,.28); box-shadow: 0 18px 40px rgba(8,15,35,.38); color: #e5e7eb; text-align: left; display: grid; gap: 18px;}
.council-hero {display:flex; align-items:center; gap: 0.9rem; padding: 16px 18px; border-radius: 18px; background: linear-gradient(135deg, rgba(30,58,138,.35), rgba(6,182,212,.14)); border:1px solid rgba(148,163,184,.32); box-shadow: inset 0 1px 0 rgba(255,255,255,.05);}
.council-hero .hero-logo {width: 62px; height: 62px; border-radius: 18px; display:flex; align-items:center; justify-content:center; background: rgba(15,23,42,.82); box-shadow: 0 12px 24px rgba(14,165,233,.22); overflow:hidden;}
.council-hero .hero-logo img {width: 46px; height: 46px; object-fit: contain; filter: drop-shadow(0 6px 18px rgba(56,189,248,.35));}
.council-hero .hero-logo .hero-fallback {width: 100%; height: 100%; display:flex; align-items:center; justify-content:center; font-size: 1.4rem; font-weight:700; color:#f8fafc;}
.council-hero .hero-copy {display:flex; flex-direction:column; gap:4px;}
.council-hero .hero-title {font-size:1rem; font-weight:600; color:#f8fafc; margin:0;}
.council-hero .hero-sub {font-size:.85rem; color:#cbd5f5; margin:0; line-height:1.4;}
.council-orbit {position: relative; height: 128px; margin: 2px 0 4px;}
.council-orbit .av {position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);}
.council-orbit .av img {width: 40px; height: 40px; border-radius: 12px; background:#fff; object-fit: contain; box-shadow: 0 6px 18px rgba(0,0,0,.28); transition: transform .2s ease, box-shadow .2s ease;}
.council-orbit .fallback {display:flex; align-items:center; justify-content:center; width:40px; height:40px; border-radius:12px; background:#1f2937; color:#e2e8f0; font-weight:700; box-shadow:0 6px 18px rgba(0,0,0,.28);}
.av.active img {transform: scale(1.16) translateZ(0); box-shadow: 0 12px 26px rgba(56,189,248,.32);}
.av.p0 {transform: translate(-50%,-50%) translate(0,-52px);}
.av.p1 {transform: translate(-50%,-50%) translate(48px,-12px);}
.av.p2 {transform: translate(-50%,-50%) translate(32px,42px);}
.av.p3 {transform: translate(-50%,-50%) translate(-32px,42px);}
.av.p4 {transform: translate(-50%,-50%) translate(-48px,-12px);}
.av.p5 {transform: translate(-50%,-50%) translate(0,66px);}
.typing {display:inline-block; letter-spacing:.15em;}
.typing span {animation: blink 1.2s infinite;}
.typing span:nth-child(2){ animation-delay:.2s; }
.typing span:nth-child(3){ animation-delay:.4s; }
@keyframes blink {0%,20%{ opacity:0;} 50%{opacity:1;} 100%{opacity:0;} }
.qline {font-size:.95rem; color:#9db4d9; min-height: 24px; text-align:left;}
.badge-sync {display:inline-flex; gap:.6rem; align-items:center; padding:.4rem .75rem; border-radius:12px; background:rgba(14,165,233,.12); color:#e0f2fe; border:1px solid rgba(56,189,248,.35); font-weight:600; font-size:.85rem; justify-content:flex-start; width:fit-content; box-shadow:0 10px 24px rgba(56,189,248,.16);}
.council-progress {height:6px; border-radius:999px; background:rgba(148,163,184,.25); overflow:hidden;}
.council-progress span {display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,#38bdf8,#6366f1); box-shadow:0 6px 16px rgba(99,102,241,.42); transition: width .18s ease;}
.hero-wrap{display:flex;flex-direction:column;gap:.4rem;}
.hero-kicker{font-size:.8rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#38bdf8;}
.hero-title{font-size:2.25rem;font-weight:700;color:#0f172a;margin:0;}
.hero-sub{font-size:1rem;color:#475569;max-width:620px;}
.hero-card{border:1px solid rgba(148,163,184,.35);background:linear-gradient(145deg,#0f172a,#1f2937);color:#e2e8f0;padding:1rem 1.1rem;border-radius:18px;display:grid;gap:.55rem;box-shadow:0 12px 30px rgba(15,23,42,.22);}
.hero-card.is-muted{background:linear-gradient(145deg,#1e293b,#111827);}
.hero-card__title{font-size:.95rem;font-weight:600;}
.hero-card__meta{display:flex;flex-wrap:wrap;gap:.35rem;}
.hero-card__meta .chip{margin-right:0; margin-bottom:0;}
.helper-card{border:1px solid #e2e8f0;background:#f8fafc;padding:.9rem 1rem;border-radius:16px;box-shadow:0 8px 18px rgba(148,163,184,.18);display:grid;gap:.55rem;}
.helper-card h4{margin:0;font-size:.95rem;color:#0f172a;}
.helper-card ul{margin:0;padding-left:1.1rem;color:#334155;font-size:.85rem;}
.helper-card li{margin-bottom:.35rem;}
.helper-card li:last-child{margin-bottom:0;}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"]{background:linear-gradient(90deg,#2563eb,#7c3aed)!important;color:#f8fafc!important;border:none!important;box-shadow:0 12px 24px rgba(79,70,229,.32)!important;font-weight:600!important;height:48px;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(COUNCIL_CSS, unsafe_allow_html=True)

# ---- Real logos: local -> download/cache -> upload fallback ----
LOGO_MAP = {
    "OpenAI":  {"filename": "openai",   "urls": ["https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_logo_2025.svg"],  "bg": "#6d28d9"},
    "Gemini":  {"filename": "gemini",   "urls": ["https://upload.wikimedia.org/wikipedia/commons/4/4f/Google_Gemini_icon_2025.svg"], "bg": "#2563eb"},
    "Grok":    {"filename": "grok",     "urls": ["https://upload.wikimedia.org/wikipedia/commons/9/9c/Groq_logo.svg"],         "bg": "#ef4444"},
    "Llama":   {"filename": "llama",    "urls": ["https://custom.typingmind.com/tools/model-icons/llama/llama.svg"],            "bg": "#16a34a"},
    "Together":{"filename": "together", "urls": ["https://custom.typingmind.com/tools/model-icons/together/together.svg"],       "bg": "#0ea5e9"},
    "Claude":  {"filename": "claude-ai-icon", "urls": [], "bg": "#d97757"},
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


def _logo_cache_token(provider: str) -> float:
    meta = LOGO_MAP.get(provider)
    if not meta:
        return 0.0
    base_name = meta.get("filename", "")
    candidates = list(ASSETS_DIR.glob(f"{base_name}.*"))
    if provider.lower() == "grok":
        candidates.extend(ASSETS_DIR.glob("groq.*"))
    latest = 0.0
    for path in candidates:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


@st.cache_data(show_spinner=False)
def _load_logo_data(provider: str, cache_token: Optional[float] = None) -> str:
    """Return a data URI for the provider logo. Tries local file, then first good download URL."""

    _ = cache_token  # parameter keeps cache aware of asset timestamp without altering logic

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
    data_uri = _load_logo_data(provider, _logo_cache_token(provider))
    if not data_uri:
        return f'<div class="avatar" style="background:{bg}">💬</div>'
    filter_style = ""
    img_background = "background:white;"
    lower_provider = provider.lower()
    if lower_provider == "grok":
        img_background = "background:transparent;"
        filter_style = " filter: brightness(0) invert(1);"
    elif lower_provider == "claude":
        img_background = "background:transparent;"
    return f'''
      <div class="avatar" style="background:{bg}; padding:4px;">
        <img src="{data_uri}" alt="{provider} logo"
             class="logo" style="width:{size}px;height:{size}px;display:block;object-fit:contain;{img_background}border-radius:6px;{filter_style}">
      </div>
    '''


def show_loading_council(models: List[str], seconds: float = 1.8) -> None:
    """Display an overlay with orbiting provider avatars and quips."""

    display_models = [m for m in models if m][:6]
    if not display_models:
        return

    placeholder = st.empty()
    frame_interval = 0.18
    start = time.perf_counter()
    frame = 0
    positions = ["p0", "p1", "p2", "p3", "p4", "p5"]

    avatar_markup: List[str] = []
    for idx, provider in enumerate(display_models):
        data_uri = _load_logo_data(provider, _logo_cache_token(provider))
        if data_uri:
            img_tag = f'<img src="{data_uri}" alt="{provider} logo">'
        else:
            initial = html.escape(provider[:1].upper()) or "?"
            img_tag = f'<div class="fallback">{initial}</div>'
        avatar_markup.append(img_tag)

    quip_cycle = [QUIPS.get(provider, "") for provider in display_models]
    if not quip_cycle:
        quip_cycle = [""]

    hero_copy_map = {
        "claude": "Claude balances structured reasoning with grounded tradeoffs.",
        "openai": "OpenAI blends fast iteration with multi-step reasoning cues.",
        "gemini": "Gemini maps real-time snippets into concise structure.",
        "grok": "Grok stress-tests low-latency paths for speed wins.",
        "together": "Together orchestrates tuned llama variants for balance.",
        "llama": "Llama keeps token efficiency tight across drafts.",
    }
    default_hero_sub = "Council balances speed, depth, and spend before the vote closes."

    def _hero_section(provider: str) -> str:
        hero_bg = LOGO_MAP.get(provider, {}).get("bg", "#38bdf8")
        hero_logo_data = _load_logo_data(provider, _logo_cache_token(provider))
        if hero_logo_data:
            hero_logo_inner = f'<img src="{hero_logo_data}" alt="{provider} logo">'
        else:
            hero_initial = html.escape(provider[:1].upper()) or "?"
            hero_logo_inner = f'<div class="hero-fallback">{hero_initial}</div>'
        hero_sub = hero_copy_map.get(provider.lower(), default_hero_sub)
        return (
            '<div class="council-hero">'
            f'<div class="hero-logo" style="background:{hero_bg}">{hero_logo_inner}</div>'
            '<div class="hero-copy">'
            f'<p class="hero-title">{html.escape(provider)} is sharing its vote</p>'
            f'<p class="hero-sub">{html.escape(hero_sub)}</p>'
            '</div>'
            '</div>'
        )

    typing_html = '<span class="typing"><span>.</span><span>.</span><span>.</span></span>'

    def _progress_bar(elapsed: float) -> str:
        ratio = min(1.0, elapsed / max(seconds, 0.1))
        width = max(6, int(ratio * 100))
        return f'<div class="council-progress"><span style="width:{width}%"></span></div>'

    try:
        while time.perf_counter() - start < seconds:
            elapsed = time.perf_counter() - start
            active_idx = frame % len(display_models)
            avatars_html = []
            for idx, provider in enumerate(display_models):
                pos_cls = positions[idx % len(positions)]
                active_cls = " active" if idx == active_idx else ""
                avatars_html.append(
                    f'<div class="av {pos_cls}{active_cls}">{avatar_markup[idx]}</div>'
                )

            current_provider = display_models[active_idx]
            quip_text = "Council balancing best-fit signals."
            if quip_cycle[active_idx]:
                quip_text = (
                    f"<strong>{html.escape(current_provider)}</strong> · "
                    f"{html.escape(quip_cycle[active_idx])}"
                )

            hero_html = _hero_section(current_provider)

            overlay_html = (
                '<div class="council-overlay">'
                '<div class="council-modal">'
                f'<div class="badge-sync">🧭 Council syncing {typing_html}</div>'
                f"{hero_html}"
                '<div class="council-orbit">'
                + "".join(avatars_html)
                + '</div>'
                f'<div class="qline">{quip_text}</div>'
                f"{_progress_bar(elapsed)}"
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
    ("Claude", "claude-3-5-sonnet", {"speed": 3, "reason": 5, "cost": 4}),
]


QUIPS = {
    "OpenAI": "I can balance speed and reasoning for a clean final answer.",
    "Gemini": "I’m fast with webby tasks and structured responses.",
    "Llama": "Lean and quick—great for short prompts and drafts.",
    "Together": "Big-brain Llama 70B is solid for deeper takes.",
    "Grok": "Blazing low latency—let me snap a result together.",
    "Claude": "Strong reasoning with grounded, concise answers.",
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


def run_council_decision(prompt: str, demo_mode: bool, sidebar_notes: str) -> Dict[str, Any]:
    """Execute the council vote and generation flow, returning structured run data."""

    overlay_models = [prov for prov, _model, _attr in MODELS]
    overlay_duration = 1.8
    min_overlay_time = 1.4
    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor:
        vote_future = executor.submit(heuristic_vote, prompt)
        show_loading_council(overlay_models, seconds=overlay_duration)
        winner_prov, winner_model, council = vote_future.result()

    elapsed = time.perf_counter() - overall_start
    if elapsed < min_overlay_time:
        time.sleep(min_overlay_time - elapsed)

    notifications: List[Tuple[str, str]] = []

    route = determine_route(prompt, winner_prov)
    start_time = time.time()
    input_hash = hash_text(prompt)
    in_toks = est_tokens(prompt)
    result_text = ""
    latency_ms = 0
    live_latency_ms: Optional[int] = None

    live_possible = not demo_mode and LIVE_CAPABLE.get(winner_prov, False)
    if winner_prov == "Llama":
        live_possible = False

    if demo_mode:
        result_text = generate_stubbed_response(winner_prov, route, prompt)
        latency_ms = int((time.time() - start_time) * 1000)
        notifications.append(("info", "Simulated mode: showing routed council summary."))
    elif not live_possible:
        notifications.append(("info", "Live mode skipped: keys missing or provider not supported. Showing simulated result."))
        result_text = generate_stubbed_response(winner_prov, route, prompt)
        latency_ms = int((time.time() - start_time) * 1000)
    else:
        try:
            with st.spinner(f"Running live on {winner_prov}…"):
                live_start = time.perf_counter()
                result_text = live_generate(winner_prov, winner_model, prompt, 0.2)
                live_latency_ms = int((time.perf_counter() - live_start) * 1000)
            latency_ms = int((time.time() - start_time) * 1000)
            notifications.append(("success", f"Live result in {live_latency_ms:.0f} ms"))
        except Exception as exc:  # Graceful failure
            error_msg = f"Live call failed: {exc}"
            notifications.append(("error", error_msg))
            result_text = error_msg
            latency_ms = int((time.time() - start_time) * 1000)

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

    alternates = council[1:4]

    return {
        "prompt": prompt,
        "winner": {"provider": winner_prov, "model": winner_model},
        "route": route,
        "council": council,
        "result_text": result_text,
        "latency_ms": latency_ms,
        "live_latency_ms": live_latency_ms,
        "in_tokens": in_toks,
        "out_tokens": out_toks,
        "cost": cost,
        "demo_mode": demo_mode,
        "live_possible": live_possible,
        "was_live": live_latency_ms is not None,
        "notifications": notifications,
        "notes": sidebar_notes,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "alternates": alternates,
    }


def render_run(run: Dict[str, Any]) -> None:
    if not run:
        return

    summary_tab, council_tab, alternates_tab = st.tabs(["Summary", "Council", "Alternates"])

    with summary_tab:
        st.markdown("#### Decision Summary")
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Routing", run["route"])
        metrics_cols[1].metric("Latency", f"{run['latency_ms']} ms")
        metrics_cols[2].metric(
            "Tokens",
            f"{run['in_tokens']} in / {run['out_tokens']} out",
        )
        metrics_cols[3].metric("Est. cost", f"${run['cost']:.4f}")

        note_bits = []
        if run.get("notes"):
            note_bits.append(f"Note: {run['notes']}")
        if run.get("timestamp"):
            note_bits.append(f"Logged {run['timestamp']}")
        if note_bits:
            st.caption(" · ".join(note_bits))

        for level, message in run.get("notifications", []):
            if level == "info":
                st.info(message)
            elif level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)
            else:
                st.write(message)

        winner = run.get("winner", {})
        winner_prov = winner.get("provider", "")
        winner_model = winner.get("model", "")

        st.markdown("### Best Pick")
        if winner_prov and winner_model:
            st.markdown('<span class="badge pick">Best Pick</span>', unsafe_allow_html=True)
            st.markdown(f"**{winner_prov} · {winner_model}**")
            st.markdown(provider_badge(winner_prov, winner_model), unsafe_allow_html=True)
            st.markdown(logo_img_html(winner_prov), unsafe_allow_html=True)

        feature_chips = ["Speed", "Reasoning", "Budget-friendly"]
        if run["route"] == "FAST":
            feature_chips = ["Latency saver", "Budget friendly", "Single-shot"]
        else:
            feature_chips = ["Deep reasoning", "High fidelity", "Context stitch"]

        st.markdown(
            "".join([f'<span class="chip chip-primary">{c}</span>' for c in feature_chips]),
            unsafe_allow_html=True,
        )

        had_error = any(level == "error" for level, _ in run.get("notifications", []))
        if run.get("was_live") and not had_error:
            render_live_card(run["result_text"])
        elif had_error:
            st.error(run["result_text"])
        else:
            render_simulated_card(winner_prov or "Council", run["result_text"])

    with council_tab:
        st.markdown("#### Council Ranking")
        council = run.get("council", [])
        if council:
            council_cols = st.columns(2)
            for idx, (prov, score, quip) in enumerate(council):
                with council_cols[idx % len(council_cols)]:
                    st.markdown(logo_img_html(prov), unsafe_allow_html=True)
                    st.markdown(f"**{prov}** · score {score}")
                    if quip:
                        st.caption(quip)
        else:
            st.caption("No council data available yet.")

    with alternates_tab:
        st.markdown("#### Alternate Takes")
        alternates = run.get("alternates", [])
        if alternates:
            alt_cols = st.columns(len(alternates))
            for idx, (prov, _score, quip) in enumerate(alternates):
                with alt_cols[idx]:
                    st.markdown(f'<span class="badge alt">{prov}</span>', unsafe_allow_html=True)
                    st.markdown(logo_img_html(prov), unsafe_allow_html=True)
                    if quip:
                        st.caption(quip)
                    btn_key = f"alt-{hash_text(run['prompt'])[:4]}-{prov}"
                    if st.button(f"Preview {prov}", key=btn_key):
                        preview = textwrap.dedent(
                            f"""
                            **{prov} (simulated take)**  
                            - Signal: {quip}  
                            - Would highlight alternate POV and give a second draft.
                            """
                        )
                        st.session_state["alt_preview"] = preview
        else:
            st.caption("Run once more to surface alternate takes.")

        if st.session_state.get("alt_preview"):
            st.markdown(st.session_state["alt_preview"])


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
    st.write("Model-agnostic router. This run uses **your API subscriptions** when Live mode is enabled.")
    providers = list(LOGO_MAP.keys())
    logo_cols = st.columns(2)
    for idx, provider in enumerate(providers):
        col = logo_cols[idx % len(logo_cols)]
        with col:
            col.markdown(logo_img_html(provider, size=42), unsafe_allow_html=True)
            col.caption(provider)

DEFAULT_PROMPT = "In 5 bullets, pitch RAD AI (Rational Automation Design): problem, solution, why now, traction, ask."

st.session_state.setdefault("alt_preview", "")
st.session_state.setdefault("prompt_text", DEFAULT_PROMPT)
demo_mode = not live

ready_keys = [name for name, ok in LIVE_CAPABLE.items() if ok]
missing_keys = [name for name, ok in LIVE_CAPABLE.items() if not ok]

hero_left, hero_right = st.columns([2.4, 1], gap="large")

with hero_left:
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="hero-kicker">Model-agnostic routing</div>
          <h1 class="hero-title">Kayak for LLMs—one council, unified output</h1>
          <p class="hero-sub">Every prompt checks speed, reasoning depth, and budget across OpenAI, Gemini, Claude, Grok, Llama, Together, and more.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    status_class = "hero-card"
    if demo_mode:
        status_class += " is-muted"
        status_title = "Simulated council"
        status_body = "Live mode is off. Toggle it once credentials are configured."
    else:
        status_title = "Live council"
        status_body = "Winning provider executes live with your available keys."

    meta_sections: List[str] = []
    ready_markup = "".join(
        f"<span class='chip chip-primary'>{html.escape(name)}</span>" for name in ready_keys
    )
    if ready_markup:
        meta_sections.append(
            f"<div class='hero-card__meta'><span class='chip'>Keys ready</span>{ready_markup}</div>"
        )

    missing_markup = "".join(
        f"<span class='chip'>{html.escape(name)}</span>" for name in missing_keys
    )
    if missing_markup:
        meta_sections.append(
            f"<div class='hero-card__meta'><span class='chip'>Add keys</span>{missing_markup}</div>"
        )

    status_html = f"""
    <div class="{status_class}">
      <div class="hero-card__title">{html.escape(status_title)}</div>
      <p class="sm">{html.escape(status_body)}</p>
      {''.join(meta_sections)}
    </div>
    """
    hero_right.markdown(status_html, unsafe_allow_html=True)

with st.form("router-form", enter_to_submit=False):
    prompt_col, info_col = st.columns([3, 1], gap="large")
    with prompt_col:
        prompt = st.text_area(
            "Your prompt",
            height=180,
            key="prompt_text",
        )
    with info_col:
        info_col.markdown(
            """
            <div class="helper-card">
              <h4>How routing works</h4>
              <ul>
                <li>Heuristics balance latency, reasoning depth, and estimated cost.</li>
                <li>Live mode only calls providers you have keys for.</li>
                <li>Logs store hashed inputs plus metrics for quick audits.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    submit = st.form_submit_button("Search the council", use_container_width=True, type="primary")

prompt_value = st.session_state.get("prompt_text", "")

results_container = st.container()

if submit:
    st.session_state["alt_preview"] = ""
    st.session_state["last_run"] = run_council_decision(prompt_value, demo_mode, sidebar_notes)

run_data = st.session_state.get("last_run")

with results_container:
    if run_data:
        render_run(run_data)
    else:
        st.caption("Run a prompt to watch the council collaborate.")

st.divider()

with st.expander("Recent routing decisions", expanded=False):
    rows = load_recent(50)
    if rows:
        st.dataframe(list(reversed(rows)), use_container_width=True)
    else:
        st.info("No logs yet. Run a few queries.")

with st.expander("Connection checklist", expanded=False):
    st.write("- 🔑 OpenAI: `OPENAI_API_KEY` (paid)")
    st.write("- 🆓 Grok: `GROK_API_KEY` (dev tier; legacy `GROQ_API_KEY` also works)")
    st.write("- 🆓 Together: `TOGETHER_API_KEY` (llama-3.3-70b free endpoint)")
    st.write("- 🆓 Gemini: `GEMINI_API_KEY` (free tier)")
    st.caption("Simulated Mode requires no keys. Live Mode only runs if a key exists for the chosen provider.")
    st.caption("Logos © their respective owners; used here for product identification in a demo UI.")
