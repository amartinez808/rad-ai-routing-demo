"""Streamlit demo for a lightweight routing agent.

The app exposes a simple UI that classifies customer prompts into two lanes: a
FAST lane for cost-effective answers and a REASONING lane for deeper analysis.
The routing logic is intentionally rule-based so that it can serve as an
explainable baseline before swapping in real LLM calls.
"""

import re
import time
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAD AI – Routing Agent (MVP)", page_icon="🤖", layout="centered")

# ----------------------------
# UI
# ----------------------------
st.title("RAD AI – Routing Agent (MVP)")
st.caption("Demo: auto-routes a customer question to the best model class for cost/speed vs. depth.")

with st.expander("What this demo shows"):
    st.markdown("""
- **Auto routing** between *FAST* (cheap/quick) and *REASONING* (slower/deeper) tracks  
- **Transparent logs** you can download for your evidence packet  
- **Rule-based v1** (no keys needed). You can later swap in real LLM calls via `LIVE_MODE`.  
""")

colA, colB = st.columns([3,2], vertical_alignment="bottom")

with colA:
    prompt = st.text_area("Paste a customer question / task", height=140,
                          placeholder="Example: I was double charged last month, can you fix it and apply my credit?")
with colB:
    route_mode = st.selectbox("Routing mode",
                              ["Auto (recommended)", "Force FAST", "Force REASONING"])
    st.write("")
    run_btn = st.button("Run demo", use_container_width=True)

# ----------------------------
# Router v1 (simple, explainable rules)
# ----------------------------
FAST = "FAST"
REASONING = "REASONING"

# Keyword bundles and patterns used by the rule-based router. Keeping them in a
# dedicated block makes future tuning straightforward.
MATH_PATTERN = re.compile(
    r"\d+\s*[-+*/^]\s*\d+|integral|derivative|equation|optimi[sz]e|complexity"
)
CODE_KEYWORDS = ["stack trace", "exception", "traceback", "python", "sql", "snippet"]
MULTISTEP_KEYWORDS = ["first", "second", "step", "steps", "plan", "workflow"]
INVESTIGATION_KEYWORDS = ["why", "root cause", "analysis", "compare", "tradeoff"]
HIGH_STAKES_KEYWORDS = [
    "legal",
    "contract",
    "security",
    "privacy",
    "compliance",
    "finance",
]


def _contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Return ``True`` if any keyword is contained in ``text``."""

    lower_text = text.lower()
    return any(keyword in lower_text for keyword in keywords)

def needs_reasoning(text: str) -> bool:
    """Return ``True`` when a prompt should be escalated to the reasoning lane.

    Heuristics (tunable):
    - mathy / analytical language
    - code or debugging terminology
    - long or multi-question prompts
    - multi-step verbs ("first", "second", "workflow", ...)
    - investigative "why" prompts or comparisons
    - high-stakes topics (legal, compliance, security, finance)
    """

    lower_text = text.lower()

    # Signals that hint the user is doing math, analytics, or coding.
    mathy = bool(MATH_PATTERN.search(lower_text))
    codey = _contains_keywords(lower_text, CODE_KEYWORDS)

    # Long or multi-part requests are typically more ambiguous and warrant the
    # deeper lane so that the response can reason step by step.
    long_enough = len(text) > 180 or text.count("?") > 1
    multi_step = _contains_keywords(lower_text, MULTISTEP_KEYWORDS)

    # "Why" questions, comparisons, and high-stakes situations (legal, privacy,
    # finance...) benefit from slower, more deliberate answers.
    investigative = _contains_keywords(lower_text, INVESTIGATION_KEYWORDS)
    high_stakes = _contains_keywords(lower_text, HIGH_STAKES_KEYWORDS)

    return any(
        [long_enough, mathy, codey, multi_step, investigative, high_stakes]
    )

def choose_route(text: str, mode: str) -> str:
    """Compute the final routing lane after honoring manual overrides."""

    if mode == "Force FAST":
        return FAST
    if mode == "Force REASONING":
        return REASONING
    # Auto
    return REASONING if needs_reasoning(text) else FAST

def mock_fast_answer(text: str) -> str:
    """Return a canned response representing a quick, low-latency answer."""

    return (
        "Here’s a quick, helpful reply (FAST lane). "
        "I’ve summarized your request and next steps to keep response time low. "
        "If you need deeper investigation, escalate to the REASONING lane."
    )

def mock_reasoning_answer(text: str) -> str:
    """Return a canned response representing a deeper, step-by-step answer."""

    return (
        "Deep-dive analysis (REASONING lane):\n"
        "1) Interpreting the request and constraints\n"
        "2) Laying out assumptions and edge cases\n"
        "3) Proposing a structured resolution plan\n"
        "4) Risks & follow-ups\n"
        "→ This lane trades latency for depth & accuracy."
    )

# ----------------------------
# (Optional) LIVE MODE skeleton
# ----------------------------
LIVE_MODE = False  # flip to True once you wire a real API
def live_answer(text: str, lane: str) -> str:
    """
    Wire in your provider(s) here, e.g. OpenAI/Anthropic/Gemini.
    Use Streamlit secrets on Streamlit Cloud:
      - Settings > Secrets > add OPENAI_API_KEY, etc.
    """
    # Example sketch (pseudocode):
    # from openai import OpenAI
    # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    # model = "gpt-4o-mini" if lane==FAST else "o3"
    # resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":text}])
    # return resp.choices[0].message.content
    return mock_reasoning_answer(text) if lane==REASONING else mock_fast_answer(text)

# ----------------------------
# Logging helpers
# ----------------------------
if "logs" not in st.session_state:
    # Maintain a simple in-memory audit trail for the current Streamlit session.
    st.session_state.logs = []

def append_log(prompt_text: str, route: str, tokens_est: int):
    """Append the latest run to the in-memory audit trail."""

    st.session_state.logs.append({
        "ts_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "route": route,
        "chars": len(prompt_text),
        "tokens_est": tokens_est,
    })

def estimate_tokens(text: str) -> int:
    """Approximate token usage using a four-characters-per-token heuristic."""

    # The value is intentionally rounded down so that the estimate remains
    # conservative and easy to reason about. In production you may want to swap
    # this for a tokenizer call.
    return max(1, len(text) // 4)

# ----------------------------
# Run
# ----------------------------
if run_btn:
    if not prompt.strip():
        st.warning("Add a prompt to route.")
    else:
        # Evaluate the heuristics, render an answer, and log the run.
        route = choose_route(prompt, route_mode)
        tokens = estimate_tokens(prompt)
        with st.spinner(f"Routing to {route} lane…"):
            time.sleep(0.4)
            if LIVE_MODE:
                answer = live_answer(prompt, route)
            else:
                answer = mock_reasoning_answer(prompt) if route==REASONING else mock_fast_answer(prompt)
            time.sleep(0.2)
        st.success(f"Route: **{route}**  •  est. tokens: ~{tokens}")
        st.write(answer)
        with st.expander("Routing rules & explanation"):
            st.code(needs_reasoning.__doc__ or "Rule-based heuristics", language="text")
            st.markdown("- Long/complex, code/math, multi-step, investigative/why, high-stakes → **REASONING**\n- Otherwise → **FAST**")

        append_log(prompt, route, tokens)

# ----------------------------
# Logs table + download
# ----------------------------
st.subheader("Live Logs")
if st.session_state.logs:
    # Surface the audit trail so users can download or clear it easily.
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download logs (CSV)", csv, file_name="radai_routing_logs.csv", mime="text/csv")
    if st.button("Clear logs"):
        st.session_state.logs = []
        st.rerun()
else:
    st.info("Run a few prompts to generate logs you can include in your evidence packet.")
