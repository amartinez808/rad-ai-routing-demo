"""Streamlit demo for a lightweight routing agent.

The app exposes a simple UI that classifies customer prompts into two lanes: a
FAST lane for cost-effective answers and a REASONING lane for deeper analysis.
The routing logic is intentionally rule-based so that it can serve as an
explainable baseline before swapping in real LLM calls.
"""

import csv
import io
import json
import re
import time
from collections.abc import Iterable
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="RAD AI – Routing Agent (MVP)", page_icon="🤖", layout="centered")

# ----------------------------
# Constants & scenarios
# ----------------------------
FAST = "FAST"
REASONING = "REASONING"

FAST_MODEL = "mock-fast-lane"
REASONING_MODEL = "mock-reasoning-lane"

MOCK_LATENCY_MS = {FAST: 220, REASONING: 640}
MOCK_COST_USD = {FAST: 0.0002, REASONING: 0.0012}

LOG_COLUMNS = [
    "ts_iso",
    "track",
    "model",
    "stakes",
    "expected_type",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "prompt",
]

SIGNAL_WEIGHTS = {
    "length": 0.7,
    "math": 1.0,
    "code": 1.0,
    "multi_step": 0.8,
    "investigative": 0.6,
    "high_stakes": 1.2,
}

SIGNAL_LABELS = {
    "length": "long / multi-question input",
    "math": "math or analytic language",
    "code": "coding or debugging terms",
    "multi_step": "explicit multi-step request",
    "investigative": "investigative why/compare phrasing",
    "high_stakes": "high-stakes domain keywords",
}

STAKE_OPTIONS = ["Low", "Standard", "High"]
EXPECTED_TYPE_OPTIONS = ["Summary", "Resolution plan", "Root-cause analysis"]

SCENARIO_NONE = "Custom prompt"
SCENARIOS = {
    "Billing dispute follow-up": {
        "prompt": "I was double charged last month, can you fix it and apply my credit?",
        "stakes": "Standard",
        "expected_type": "Resolution plan",
    },
    "Security questionnaire": {
        "prompt": "Our legal team needs to understand how customer data is encrypted at rest across regions. Provide the details and any compliance policies we align to.",
        "stakes": "High",
        "expected_type": "Root-cause analysis",
    },
    "Feature comparison": {
        "prompt": "Lay out the differences between the FAST and REASONING plans, and recommend which one we should offer to a regulated healthcare client.",
        "stakes": "High",
        "expected_type": "Summary",
    },
}

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

# ----------------------------
# Session state defaults
# ----------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""
if "stakes" not in st.session_state:
    st.session_state.stakes = STAKE_OPTIONS[1]
if "expected_type" not in st.session_state:
    st.session_state.expected_type = EXPECTED_TYPE_OPTIONS[0]
if "route_mode" not in st.session_state:
    st.session_state.route_mode = "Auto (recommended)"
if "scenario_choice" not in st.session_state:
    st.session_state.scenario_choice = SCENARIO_NONE

# ----------------------------
# Helpers
# ----------------------------
def _contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Return ``True`` if any keyword is contained in ``text``."""

    lower_text = text.lower()
    return any(keyword in lower_text for keyword in keywords)


def evaluate_signals(text: str) -> dict[str, bool]:
    """Return the individual heuristic signals that inform routing."""

    lower_text = text.lower()
    return {
        "length": len(text) > 180 or text.count("?") > 1,
        "math": bool(MATH_PATTERN.search(lower_text)),
        "code": _contains_keywords(lower_text, CODE_KEYWORDS),
        "multi_step": _contains_keywords(lower_text, MULTISTEP_KEYWORDS),
        "investigative": _contains_keywords(lower_text, INVESTIGATION_KEYWORDS),
        "high_stakes": _contains_keywords(lower_text, HIGH_STAKES_KEYWORDS),
    }


def compute_scores(signals: dict[str, bool]) -> tuple[float, float]:
    """Return reasoning vs. fast scores derived from weighted signals."""

    reasoning_score = sum(SIGNAL_WEIGHTS[name] for name, hit in signals.items() if hit)
    reasoning_score = round(reasoning_score, 2)
    fast_score = round(max(0.0, 1.0 - min(reasoning_score, 1.0)), 2)
    return reasoning_score, fast_score


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

    return any(evaluate_signals(text).values())


def route_prompt(text: str, mode: str) -> dict[str, object]:
    """Evaluate routing signals and return structured decision metadata."""

    signals = evaluate_signals(text)
    reasoning_score, fast_score = compute_scores(signals)
    auto_route = REASONING if reasoning_score > 0 else FAST

    if mode == "Force FAST":
        route = FAST
    elif mode == "Force REASONING":
        route = REASONING
    else:
        route = auto_route

    sorted_hits = sorted(
        ((name, SIGNAL_WEIGHTS[name]) for name, hit in signals.items() if hit),
        key=lambda item: item[1],
        reverse=True,
    )
    top_signals = [SIGNAL_LABELS[name] for name, _ in sorted_hits[:2]]

    return {
        "route": route,
        "auto_route": auto_route,
        "signals": signals,
        "reasoning_score": reasoning_score,
        "fast_score": fast_score,
        "top_signals": top_signals,
    }


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


def live_answer(text: str, lane: str):
    """
    Wire in your provider(s) here, e.g. OpenAI/Anthropic/Gemini.
    Use Streamlit secrets on Streamlit Cloud:
      - Settings > Secrets > add OPENAI_API_KEY, etc.
    Return either a plain string or a dict with answer metadata.
    """

    # Example sketch (pseudocode):
    # from openai import OpenAI
    # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    # model = "gpt-4o-mini" if lane==FAST else "o3"
    # resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":text}])
    # return {
    #     "answer": resp.choices[0].message.content,
    #     "model": model,
    #     "tokens_out": resp.usage.completion_tokens,
    #     "cost_usd": resp.usage.total_cost,
    # }
    return mock_reasoning_answer(text) if lane == REASONING else mock_fast_answer(text)


def get_response(text: str, lane: str) -> tuple[str, str, int, float]:
    """Return answer text, model name, token estimate, and cost for a lane."""

    if LIVE_MODE:
        payload = live_answer(text, lane)
        if isinstance(payload, dict):
            answer = payload.get("answer", "")
            model = payload.get("model", FAST_MODEL if lane == FAST else REASONING_MODEL)
            tokens_out = payload.get("tokens_out", estimate_tokens(answer))
            cost_usd = payload.get("cost_usd", 0.0)
            return answer, model, tokens_out, cost_usd
        answer = str(payload)
        model = FAST_MODEL if lane == FAST else REASONING_MODEL
        tokens_out = estimate_tokens(answer)
        return answer, model, tokens_out, 0.0

    answer = mock_reasoning_answer(text) if lane == REASONING else mock_fast_answer(text)
    model = REASONING_MODEL if lane == REASONING else FAST_MODEL
    tokens_out = estimate_tokens(answer)
    cost_usd = MOCK_COST_USD[lane]
    return answer, model, tokens_out, cost_usd


# ----------------------------
# Logging helpers
# ----------------------------
def append_log(row: dict) -> None:
    """Append a normalized audit row to the in-session log store."""

    normalized = {field: row.get(field) for field in LOG_COLUMNS}
    st.session_state.logs.append(normalized)


def estimate_tokens(text: str) -> int:
    """Approximate token usage using a four-characters-per-token heuristic."""

    # The value is intentionally rounded down so that the estimate remains
    # conservative and easy to reason about. In production you may want to swap
    # this for a tokenizer call.
    return max(1, len(text) // 4)


def _apply_scenario() -> None:
    """Populate form inputs when a canned scenario is selected."""

    choice = st.session_state.scenario_choice
    if choice == SCENARIO_NONE:
        return
    scenario = SCENARIOS[choice]
    st.session_state.prompt_input = scenario["prompt"]
    st.session_state.stakes = scenario["stakes"]
    st.session_state.expected_type = scenario["expected_type"]


def _format_rationale(decision: dict[str, object]) -> str:
    """Create a one-line rationale summarizing the router decision."""

    sr = decision["reasoning_score"]
    sf = decision["fast_score"]
    top_signals = decision["top_signals"]
    if top_signals:
        reason = " and ".join(top_signals)
    else:
        reason = "no escalation signals detected"
    return f"Router chose **{decision['route']}** (sr={sr:.2f}, sf={sf:.2f}) due to {reason}."


# ----------------------------
# UI
# ----------------------------
st.title("RAD AI – Routing Agent (MVP)")
st.caption("Demo: auto-routes a customer question to the best model class for cost/speed vs. depth.")

with st.expander("What this demo shows"):
    st.markdown(
        """
- **Auto routing** between *FAST* (cheap/quick) and *REASONING* (slower/deeper) tracks  
- **Transparent logs** you can download for your evidence packet  
- **Rule-based v1** (no keys needed). You can later swap in real LLM calls via `LIVE_MODE`.  
"""
    )

st.selectbox(
    "Canned scenarios",
    [SCENARIO_NONE, *SCENARIOS.keys()],
    key="scenario_choice",
    on_change=_apply_scenario,
)

colA, colB = st.columns([3, 2], vertical_alignment="top")

with colA:
    prompt_value = st.text_area(
        "Paste a customer question / task",
        height=140,
        key="prompt_input",
        placeholder="Example: I was double charged last month, can you fix it and apply my credit?",
    )

with colB:
    route_mode = st.selectbox(
        "Routing mode",
        ["Auto (recommended)", "Force FAST", "Force REASONING"],
        key="route_mode",
    )
    stakes_value = st.selectbox("Stakes", STAKE_OPTIONS, key="stakes")
    expected_type_value = st.selectbox("Expected output", EXPECTED_TYPE_OPTIONS, key="expected_type")
    st.write("")
    run_btn = st.button("Run demo", use_container_width=True)

# ----------------------------
# Run & report
# ----------------------------
prompt_clean = prompt_value.strip()

if run_btn:
    if not prompt_clean:
        st.warning("Add a prompt to route.")
    else:
        decision = route_prompt(prompt_clean, route_mode)
        tokens_in = estimate_tokens(prompt_clean)
        start_ts = time.perf_counter()
        with st.spinner(f"Routing to {decision['route']} lane…"):
            time.sleep(0.25)
            answer_text, model_name, tokens_out, cost_usd = get_response(prompt_clean, decision["route"])
            time.sleep(0.15)
        latency_ms = max(1, int((time.perf_counter() - start_ts) * 1000))

        st.success(f"Route: **{decision['route']}**  •  est. tokens: ~{tokens_in}")
        st.write(answer_text)

        with st.expander("Decision & Metrics", expanded=True):
            st.markdown(_format_rationale(decision))
            st.json(
                {
                    "selected_mode": route_mode,
                    "auto_route": decision["auto_route"],
                    "final_route": decision["route"],
                    "scores": {"sr": decision["reasoning_score"], "sf": decision["fast_score"]},
                    "signals": decision["signals"],
                    "top_signals": decision["top_signals"],
                }
            )

        with st.expander("Routing rules & explanation"):
            st.code(needs_reasoning.__doc__ or "Rule-based heuristics", language="text")
            st.markdown(
                "- Long/complex, code/math, multi-step, investigative/why, high-stakes → **REASONING**\n"
                "- Otherwise → **FAST**"
            )

        append_log(
            {
                "ts_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "track": decision["route"],
                "model": model_name,
                "stakes": stakes_value,
                "expected_type": expected_type_value,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost_usd, 6),
                "latency_ms": latency_ms,
                "prompt": prompt_clean,
            }
        )

# ----------------------------
# Logs UI
# ----------------------------
st.subheader("Audit logs")
if st.session_state.logs:
    logs_view = list(reversed(st.session_state.logs))
    st.dataframe(logs_view, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=LOG_COLUMNS)
    writer.writeheader()
    writer.writerows(logs_view)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    json_bytes = json.dumps(logs_view, indent=2).encode("utf-8")

    col_csv, col_json, col_clear = st.columns([1, 1, 0.8])
    with col_csv:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="rad-ai-routing-logs.csv",
            mime="text/csv",
        )
    with col_json:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="rad-ai-routing-logs.json",
            mime="application/json",
        )
    with col_clear:
        if st.button("Clear logs"):
            st.session_state.logs.clear()
            st.experimental_rerun()
else:
    st.info("No runs logged yet. Run the demo to generate an audit trail.")
