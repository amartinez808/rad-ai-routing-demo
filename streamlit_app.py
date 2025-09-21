import os
import time
import random
import json
import textwrap
import html
from typing import Optional, Dict, List, Tuple

import requests
import streamlit as st

# =========================
# App Config
# =========================
st.set_page_config(page_title="RAD AI – LLM Metasearch (Kayak Demo)", page_icon="🧭", layout="centered")

# -------------------------
# Branding / Theme CSS
# -------------------------
st.markdown(
    """
<style>
.badge {display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .6rem; border-radius:999px; font-weight:600; font-size:.85rem;}
.pick {background: #0ea5e9; color:white;}
.alt {background: #e2e8f0; color:#0f172a;}
.chip {display:inline-block; padding:.25rem .5rem; background:#f1f5f9; border-radius:999px; font-size:.8rem; margin-right:.35rem;}
.bubble {background:white; border:1px solid #e5e7eb; border-radius:14px; padding:.6rem .8rem; box-shadow:0 1px 2px rgba(0,0,0,.04);}
.row {display:flex; gap:.75rem; align-items:flex-start;}
.avatar {width:38px; height:38px; border-radius:999px; display:flex; align-items:center; justify-content:center; font-weight:700; color:white; padding:4px; transition: transform .2s ease;}
.avatar:hover { transform: translateY(-1px) rotate(-2deg); }
.logo { width:30px; height:30px; display:block; object-fit:contain; background:white; border-radius:6px; }
.sm {font-size:.85rem; color:#475569;}
.fade {animation: fade .5s ease-in-out;}
@keyframes fade {from{opacity:0; transform:translateY(4px);} to{opacity:1; transform:none;}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.chip {color:#0f172a;}
.chip-primary {background:#2563eb; color:#f8fafc; box-shadow:0 4px 12px rgba(37, 99, 235, .25);}
.row {margin-bottom:.35rem;}
.run-card {padding:1rem; border-radius:16px; border:1px solid #cbd5f5; background:linear-gradient(145deg,#f8fafc,#eef2ff); box-shadow:0 8px 20px rgba(59,130,246,.1); color:#0f172a;}
.run-card strong, .run-card b {color:#0f172a;}
.run-card p, .run-card li {color:#0f172a;}
.run-card ul {margin-left:1.1rem; padding-left:0.2rem;}
.run-card li {margin-bottom:.3rem;}
.run-card br {line-height:1.6;}
.stMarkdown h3 {color:#e2e8f0;}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Brand icons
# -------------------------
ICONS = {
    "OpenAI": {
        "src": "https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_logo_2025.svg",
        "bg": "#6d28d9",
    },
    "Gemini": {
        "src": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Google_Gemini_icon_2025.svg",
        "bg": "#2563eb",
    },
    "Groq": {
        "src": "https://upload.wikimedia.org/wikipedia/commons/9/9c/Groq_logo.svg",
        "bg": "#ef4444",
    },
    "Llama": {
        "src": "https://custom.typingmind.com/tools/model-icons/llama/llama.svg",
        "bg": "#16a34a",
    },
    "Together": {
        "src": "https://custom.typingmind.com/tools/model-icons/together/together.svg",
        "bg": "#0ea5e9",
    },
}


def logo_img_html(provider: str) -> str:
    icon = ICONS.get(provider)
    if not icon:
        return '<div class="avatar" style="background:#64748b">💬</div>'
    return f"""
      <div class="avatar" style="background:{icon['bg']};">
        <img src="{icon['src']}" alt="{provider} logo" class="logo">
      </div>
    """

# -------------------------
# Secrets helper / Live mode gates
# -------------------------

def _secret(key: str) -> Optional[str]:
    return st.secrets.get(key) or os.getenv(key)


HAVE_OPENAI = bool(_secret("OPENAI_API_KEY"))
HAVE_GROQ = bool(_secret("GROQ_API_KEY"))
HAVE_TOGETHER = bool(_secret("TOGETHER_API_KEY"))
HAVE_GEMINI = bool(_secret("GEMINI_API_KEY"))
LIVE_CAPABLE = {
    "OpenAI": HAVE_OPENAI,
    "Groq": HAVE_GROQ,
    "Together": HAVE_TOGETHER,
    "Gemini": HAVE_GEMINI,
}


def _escape_html(text: str) -> str:
    """HTML-escape and preserve newlines within our custom cards."""

    escaped = html.escape(text)
    return escaped.replace("\n", "<br>")


def render_simulated_card(provider: str) -> None:
    """Render the best-pick card with friendly demo copy."""

    html_body = f"""
    <div class='run-card fade'>
      <p><strong>Why {provider}?</strong> Balanced signal from the council for your request.</p>
      <p><strong>Draft answer (simulated)</strong></p>
      <ul>
        <li>Problem: founders juggle dozens of LLMs without clear guidance.</li>
        <li>Solution: RAD AI routes prompts like Kayak surfaces the best flight.</li>
        <li>Why now: model quality exploded; buyers want outcomes, not model soup.</li>
        <li>Traction: demo pilots show faster time-to-answer and lower cost.</li>
        <li>Ask: intros to 3 design partners ready to ship co-branded pilots.</li>
      </ul>
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
    ("Groq", "llama-3.3-70b", {"speed": 5, "reason": 4, "cost": 4}),
]


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

    quips = {
        "OpenAI": "I can balance speed and reasoning for a clean final answer.",
        "Gemini": "I’m fast with webby tasks and structured responses.",
        "Llama": "Lean and quick—great for short prompts and drafts.",
        "Together": "Big-brain Llama 70B is solid for deeper takes.",
        "Groq": "Blazing low latency—let me snap a result together.",
    }
    council = [(p, round(s, 2), quips[p]) for (p, _m, s) in base]
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

    if "groq" in pl:
        api_key = _secret("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")
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
st.title("🧭 RAD AI – Kayak for LLMs")
st.caption("Type once. Watch the LLM council confer. Get one **Best Pick** and simple alternates. (Visual demo first; live mode optional.)")

st.session_state.setdefault("alt_preview", "")

demo_mode = st.toggle(
    "Simulated Mode (recommended for demo)",
    value=True,
    help="If off and keys exist, the Best Pick will run live.",
)

prompt = st.text_area(
    "Your prompt",
    "In 5 bullets, pitch RAD AI (Rational Automation Design): problem, solution, why now, traction, ask.",
    height=140,
)

go = st.button("Search all models")

if go:
    st.session_state["alt_preview"] = ""
    with st.spinner("Gathering the council…"):
        winner_prov, winner_model, council = heuristic_vote(prompt)
        time.sleep(0.6)

    st.markdown("#### 🤝 LLM Council")
    for prov, score, quip in council:
        st.markdown(
            f"""
            <div class="row fade">
              {logo_img_html(prov)}
              <div class="bubble">
                <div><b>{prov}</b> · <span class="sm">score {score}</span></div>
                <div class="sm">{quip}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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

    live_possible = not demo_mode and LIVE_CAPABLE.get(winner_prov, False)
    if winner_prov == "Llama":
        live_possible = False  # Llama entry is simulated-only

    if not demo_mode and not live_possible:
        st.info("Live mode skipped: keys missing or provider not supported. Showing simulated result.")

    if demo_mode or not live_possible:
        render_simulated_card(winner_prov)
    else:
        try:
            with st.spinner(f"Running live on {winner_prov}…"):
                start = time.perf_counter()
                result = live_generate(winner_prov, winner_model, prompt, 0.2)
                latency_ms = (time.perf_counter() - start) * 1000
            st.success(f"Live result in {latency_ms:.0f} ms")
            render_live_card(result)
        except Exception as exc:  # Graceful failure
            st.error(f"Live call failed: {exc}")

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
st.markdown("##### Connection checklist")
st.write("- 🔑 OpenAI: `OPENAI_API_KEY` (paid)")
st.write("- 🆓 Groq: `GROQ_API_KEY` (dev tier)")
st.write("- 🆓 Together: `TOGETHER_API_KEY` (llama-3.3-70b free endpoint)")
st.write("- 🆓 Gemini: `GEMINI_API_KEY` (free tier)")
st.caption("Simulated Mode requires no keys. Live Mode only runs if a key exists for the chosen provider.")
st.caption("Logos © their respective owners; used here for product identification in a demo UI.")
