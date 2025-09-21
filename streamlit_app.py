"""Investor-facing Streamlit demo for the RAD AI routing agent."""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

try:  # optional OpenAI import for live mode
    from openai import OpenAI
    from openai import OpenAIError
except ImportError:  # pragma: no cover - allow running without SDK
    OpenAI = None  # type: ignore

    class OpenAIError(Exception):  # type: ignore
        """Fallback OpenAI exception type when SDK missing."""


logger = logging.getLogger(__name__)

FAST = "FAST"
REASONING = "REASONING"
MOCK_FAST_MODEL = "mock-fast-lane"
MOCK_REASONING_MODEL = "mock-reasoner-x"

COLORS = {
    "primary": "#7C3AED",
    "success": "#22C55E",
    "warn": "#F59E0B",
    "danger": "#EF4444",
    "surface": "#0F172A",
    "text": "#E5E7EB",
    "subtle": "#94A3B8",
}

BASE_TRACK_METRICS = {
    FAST: {
        "cost": 0.0002,
        "latency": 220,
        "depth": 0.45,
        "model": MOCK_FAST_MODEL,
        "pricing": {"input": 0.00015, "output": 0.0006},
    },
    REASONING: {
        "cost": 0.0014,
        "latency": 720,
        "depth": 0.92,
        "model": MOCK_REASONING_MODEL,
        "pricing": {"input": 0.0009, "output": 0.0025},
    },
}

LIVE_TRACK_METRICS = {
    FAST: {
        "cost": 0.0004,
        "latency": 650,
        "depth": 0.62,
        "model": "gpt-4o-mini",
        "pricing": {"input": 0.00015, "output": 0.0006},
    },
    REASONING: {
        "cost": 0.0075,
        "latency": 1500,
        "depth": 0.95,
        "model": "gpt-4o",
        "pricing": {"input": 0.005, "output": 0.015},
    },
}

MODEL_INFO = {
    FAST: {
        "label": "FAST lane",
        "strengths": ["Low latency triage", "Cost efficient", "Great for FAQs"],
        "weaknesses": ["Limited step-by-step reasoning", "No deep audit trail"],
    },
    REASONING: {
        "label": "REASONING lane",
        "strengths": ["Structured plans", "Handles ambiguity", "Risk-aware"],
        "weaknesses": ["Higher cost", "Slightly slower"],
    },
}

SENSITIVE_KEYWORDS = {
    "bleeding",
    "overdose",
    "lawsuit",
    "legal",
    "medical",
    "lawsuit",
    "outage",
    "privacy",
    "pii",
    "hipaa",
    "malpractice",
    "breach",
    "safety",
}

EXTREME_KEYWORDS = {"bleeding heavily", "not breathing", "chest pain", "overdose", "suicidal"}

PROGRESS_STEPS = ["Scoring", "Selecting track", "Generating", "Summarizing", "Logging"]

st.set_page_config(page_title="RAD AI – Routing Agent", page_icon="🤖", layout="wide")


def get_track_metrics() -> Dict[str, Dict[str, Any]]:
    """Return the active track metrics for the current session."""

    metrics = st.session_state.get("track_metrics")
    if metrics is None:
        return BASE_TRACK_METRICS
    return metrics


def update_track_metrics(live: bool) -> None:
    """Copy track metrics into session state, applying live overrides when needed."""

    source = LIVE_TRACK_METRICS if live else BASE_TRACK_METRICS
    metrics_copy = {lane: dict(values) for lane, values in source.items()}
    if live:
        fast_override = st.secrets.get("FAST_MODEL_ID")
        reasoning_override = st.secrets.get("REASONING_MODEL_ID")
        if fast_override:
            metrics_copy[FAST]["model"] = fast_override
        if reasoning_override:
            metrics_copy[REASONING]["model"] = reasoning_override
    st.session_state.track_metrics = metrics_copy


def is_live_mode() -> bool:
    """Return whether LIVE mode is toggled on for this session."""

    return bool(st.session_state.get("live_mode", False))


def get_openai_client() -> OpenAI | None:
    """Create (or reuse) an OpenAI client if credentials are configured."""

    if OpenAI is None:
        return None

    if "openai_client" in st.session_state:
        return st.session_state.openai_client

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = st.secrets.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    st.session_state.openai_client = OpenAI(**client_kwargs)
    return st.session_state.openai_client


def inject_css() -> None:
    """Inject custom styling for the dark, compact layout."""

    st.markdown(
        f"""
        <style>
            body {{ background: {COLORS['surface']}; color: {COLORS['text']}; }}
            .stApp {{ background: {COLORS['surface']}; }}
            .rad-header {{ display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem; }}
            .rad-pill {{ background:{COLORS['primary']}; color:#fff; padding:0.2rem 0.65rem; border-radius:999px; font-size:0.75rem; font-weight:600; }}
            .rad-card {{ background:rgba(15,23,42,0.8); border:1px solid rgba(148,163,184,0.25); border-radius:1rem; padding:1.25rem; box-shadow:0 12px 35px rgba(15,23,42,0.45); }}
            .route-card {{ border-radius:1.25rem; padding:1.5rem; border:1px solid rgba(124,58,237,0.45); position:relative; overflow:hidden; background:linear-gradient(135deg, rgba(124,58,237,0.08), rgba(15,23,42,0.75)); }}
            .route-badge {{ display:inline-flex; align-items:center; gap:0.35rem; padding:0.2rem 0.65rem; border-radius:999px; font-size:0.78rem; font-weight:600; }}
            .metric-chip {{ background:rgba(148,163,184,0.1); padding:0.55rem 0.75rem; border-radius:999px; font-size:0.78rem; margin-right:0.35rem; display:inline-flex; align-items:center; gap:0.35rem; }}
            .metric-chip span {{ font-weight:600; color:{COLORS['text']}; }}
            .message-bubble {{ background:rgba(15,23,42,0.65); border:1px solid rgba(148,163,184,0.25); border-radius:1rem; padding:1rem 1.25rem; position:relative; font-size:0.95rem; line-height:1.5; color:{COLORS['text']}; }}
            .message-bubble button {{ background:none; border:none; color:{COLORS['subtle']}; position:absolute; top:0.55rem; right:0.55rem; cursor:pointer; transition:color 0.2s ease; }}
            .message-bubble button:hover {{ color:{COLORS['primary']}; }}
            .model-card {{ border-radius:1rem; padding:1rem; border:1px solid rgba(148,163,184,0.2); background:rgba(15,23,42,0.65); margin-bottom:1rem; transition:transform 0.2s ease, box-shadow 0.2s ease; }}
            .model-card:hover {{ transform:translateY(-2px); box-shadow:0 12px 30px rgba(124,58,237,0.15); }}
            .model-card.selected {{ border-color:{COLORS['primary']}; box-shadow:0 0 0 1px rgba(124,58,237,0.35); position:relative; }}
            .model-card.selected::after {{ content:"Selected"; position:absolute; top:0.75rem; right:0.75rem; background:{COLORS['primary']}; color:white; padding:0.15rem 0.55rem; border-radius:999px; font-size:0.7rem; font-weight:600; }}
            .footer {{ margin-top:3rem; padding:1.5rem 0; border-top:1px solid rgba(148,163,184,0.2); font-size:0.8rem; color:{COLORS['subtle']}; text-align:center; }}
            .hover-raise:hover {{ transform:translateY(-1px); box-shadow:0 10px 25px rgba(124,58,237,0.2); }}
            .risk-meter {{ width:100%; height:14px; border-radius:10px; background:rgba(148,163,184,0.15); position:relative; overflow:hidden; margin-bottom:0.65rem; }}
            .risk-meter::after {{ content:""; position:absolute; top:0; left:0; bottom:0; width:var(--risk-level); border-radius:10px; background:linear-gradient(90deg, {COLORS['warn']}, {COLORS['danger']}); transition:width 0.35s ease; }}
            .safety-banner {{ border-radius:0.75rem; padding:0.9rem 1rem; border:1px solid rgba(239,68,68,0.55); background:rgba(239,68,68,0.15); color:{COLORS['text']}; margin-bottom:0.9rem; font-size:0.85rem; }}
            .emergency-banner {{ border-radius:0.75rem; padding:0.9rem 1rem; border:1px solid rgba(239,68,68,0.85); background:rgba(239,68,68,0.35); color:{COLORS['text']}; margin-bottom:0.9rem; font-size:0.9rem; font-weight:600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_canned_cases() -> Dict[str, Dict[str, str]]:
    """Return canned scenarios for quick demos."""

    return {
        "Refund request": {
            "message": "Hi team, I was billed twice on invoice #4432. Please refund the duplicate charge and confirm when the credit lands.",
            "stakes": "Standard",
            "expected": "Answer",
        },
        "Kid swallowed penny": {
            "message": "My kid swallowed a penny and I'm worried it's stuck. She's breathing but says her stomach hurts. What should I do?",
            "stakes": "High",
            "expected": "Step-by-step",
        },
        "Travel reschedule": {
            "message": "My flight tomorrow was canceled. Please rebook me for the earliest flight to Boston and keep the aisle seat if possible.",
            "stakes": "Standard",
            "expected": "Action plan",
        },
        "Small-Biz lead qual": {
            "message": "We’re a 12-person marketing agency evaluating AI tools. What plan fits under $500/month and supports team reporting?",
            "stakes": "Low",
            "expected": "Summary",
        },
        "Outage incident update": {
            "message": "Our production API has been down for 23 minutes. Customers are escalating. Need a coordinated status update with mitigation steps.",
            "stakes": "High",
            "expected": "Action plan",
        },
    }


def estimate_tokens(text: str) -> int:
    """Approximate token usage with a four-characters-per-token heuristic."""

    stripped = text.strip()
    return max(1, len(stripped) // 4) if stripped else 1


def detect_sensitive_flags(message: str) -> tuple[bool, bool]:
    """Return booleans indicating sensitive and extreme keyword hits."""

    lowered = message.lower()
    sensitive = any(keyword in lowered for keyword in SENSITIVE_KEYWORDS)
    extreme = any(keyword in lowered for keyword in EXTREME_KEYWORDS)
    return sensitive, extreme


def build_summary_points(decision: Dict[str, Any]) -> List[str]:
    """Create TL;DR bullet points for the summary tab."""

    return [
        f"Track: {decision['track']} using {decision['model']} with risk score {decision['risk_score']:.2f}.",
        f"Customer intent mapped to `{decision['expected_type']}` output — optimized for {('depth' if decision['track'] == REASONING else 'speed')}.",
        f"Estimated cost ${decision['cost_usd']:.4f} with {decision['tokens_in']} tokens in / {decision['tokens_out']} out.",
    ]


def build_mock_answer(lane: str) -> str:
    """Return the mocked answer text for a given routing lane."""

    if lane == FAST:
        return (
            "FAST lane response: concise next steps sent. "
            "We’ll confirm the resolution and keep costs low."
        )
    return (
        "REASONING lane response: detailed plan drafted with risks, mitigation, and follow-ups."
    )


def route_request(
    message: str,
    stakes: str,
    expected_type: str,
    mode: str,
) -> Dict[str, Any]:
    """Mock the routing decision, returning explainable metadata for the UI."""

    lowered = message.lower()
    sensitive, extreme = detect_sensitive_flags(message)
    metrics_map = get_track_metrics()

    tokens_in = estimate_tokens(message)
    reasons: List[str] = []
    citations: List[str] = []

    risk_score = 0.18
    if stakes == "High":
        risk_score += 0.35
        reasons.append("High stakes → safety-first routing")
        citations.append("Routing rule: Stakes=High triggers escalation")
    elif stakes == "Low":
        risk_score -= 0.05

    if sensitive:
        risk_score += 0.28
        reasons.append("Contains med/legal/safety keywords")
        citations.append("Routing rule: Safety keywords escalate")

    if len(message) > 220 or message.count("?") > 1:
        risk_score += 0.08
        reasons.append("Multi-part or lengthy prompt requires structured reasoning")
        citations.append("Routing rule: Long prompts → more tokens")

    if "refund" in lowered or "invoice" in lowered:
        reasons.append("Billing intent detected → FAST handles transactional updates")
    if "outage" in lowered or "incident" in lowered:
        risk_score += 0.12
        reasons.append("Incident response topic → deeper coordination")
        citations.append("Routing rule: Outage keywords escalate")

    risk_score = max(0.0, min(1.0, round(risk_score, 2)))

    auto_track = FAST
    if stakes == "High" or sensitive:
        auto_track = REASONING
    elif len(message) > 220 or message.count("?") > 1 or "plan" in lowered:
        auto_track = REASONING
    elif "refund" in lowered or "invoice" in lowered:
        auto_track = FAST

    if mode == "Force FAST":
        track = FAST
        reasons.append("Manual override → forced FAST lane")
        citations.append("Operator override applied")
    elif mode == "Force REASONING":
        track = REASONING
        reasons.append("Manual override → forced REASONING lane")
        citations.append("Operator override applied")
    else:
        track = auto_track
        if track == FAST and "Manual override" not in " ".join(reasons):
            reasons.append("Low-risk prompt prioritized for cost and speed")
        elif track == REASONING and "High stakes" not in " ".join(reasons):
            reasons.append("Depth-first reasoning selected for richer guidance")

    metrics = metrics_map[track]
    latency_ms = metrics["latency"] + (60 if risk_score > 0.6 else 0)
    cost_usd = round(metrics["cost"] * (1.15 if risk_score > 0.6 else 1.0), 5)
    tokens_out = max(1, int(tokens_in * (1.35 if track == REASONING else 0.85)))

    answer_text = build_mock_answer(track)

    decision: Dict[str, Any] = {
        "track": track,
        "auto_track": auto_track,
        "why": reasons,
        "risk_score": risk_score,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "expected_type": expected_type,
        "model": metrics["model"],
        "answer": answer_text,
        "citations": citations or ["Routing rule library consulted"],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "summary_points": [],
        "message": message,
        "sensitive": sensitive,
        "extreme": extreme,
        "stakes": stakes,
        "mode": mode,
        "comparison": {
            FAST: metrics_map[FAST],
            REASONING: metrics_map[REASONING],
        },
    }
    decision["summary_points"] = build_summary_points(decision)
    return decision


def live_answer(message: str, lane: str, expected_type: str) -> Dict[str, Any]:
    """Call the configured LLM provider to generate a live answer."""

    client = get_openai_client()
    if client is None:
        raise RuntimeError("LIVE mode requires an OPENAI_API_KEY secret.")

    metrics = get_track_metrics()[lane]
    model_name = metrics["model"]
    tone = (
        "Be concise, deterministic, and focus on immediate resolution." if lane == FAST else
        "Deliver a structured, risk-aware plan with clear steps and caveats."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are RAD AI's routing agent (lane: "
                f"{lane}). Respond in the tone described and tailor the answer to a {expected_type.lower()} output. "
                "Reference safety or compliance considerations when relevant."
            ),
        },
        {"role": "user", "content": message},
    ]

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2 if lane == FAST else 0.15,
        messages=messages,
    )

    choice = response.choices[0]
    answer_text = (choice.message.content or "").strip()
    usage = getattr(response, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", None) or estimate_tokens(message)
    tokens_out = getattr(usage, "completion_tokens", None) or estimate_tokens(answer_text)
    pricing = metrics.get("pricing", {})
    cost_usd = (
        (tokens_in / 1000) * pricing.get("input", 0.0)
        + (tokens_out / 1000) * pricing.get("output", 0.0)
    )

    return {
        "answer": answer_text or build_mock_answer(lane),
        "model": model_name,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 6),
    }


def get_response(message: str, lane: str, expected_type: str) -> Dict[str, Any]:
    """Return answer payload, using LIVE mode when available."""

    metrics = get_track_metrics()[lane]
    if is_live_mode():
        try:
            payload = live_answer(message, lane, expected_type)
            payload.setdefault("model", metrics["model"])
            return payload
        except (OpenAIError, RuntimeError) as exc:  # pragma: no cover - runtime path
            logger.warning("Live model call failed; falling back to mock. Error: %s", exc)
            return {
                "answer": build_mock_answer(lane),
                "model": metrics["model"],
                "tokens_in": estimate_tokens(message),
                "tokens_out": estimate_tokens(build_mock_answer(lane)),
                "cost_usd": round(metrics["cost"], 6),
                "notes": f"Live model fallback: {exc}",
            }

    return {
        "answer": build_mock_answer(lane),
        "model": metrics["model"],
        "tokens_in": estimate_tokens(message),
        "tokens_out": estimate_tokens(build_mock_answer(lane)),
        "cost_usd": round(metrics["cost"], 6),
    }


def render_route_card(decision: Dict[str, Any]) -> None:
    """Render the central routing explanation card and supporting details."""

    track = decision["track"]
    metrics_map = get_track_metrics()
    badge_color = COLORS["success"] if track == FAST else COLORS["warn"]
    if track == REASONING and decision["risk_score"] >= 0.65:
        badge_color = COLORS["danger"]

    reason_text = decision["why"][0] if decision["why"] else "default routing logic"

    st.markdown(
        f"""
        <div class="route-card hover-raise">
            <span class="route-badge" style="background:{badge_color}; color:#0B1120;">{track}</span>
            <h2 style="margin-top:0.75rem; margin-bottom:0.2rem; color:{COLORS['text']};">Routed to {track}</h2>
            <p style="margin:0; color:{COLORS['subtle']};">because {reason_text}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chip_container = st.container()
    with chip_container:
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(
            f"<div class='metric-chip'><span>Cost</span> ${decision['cost_usd']:.4f}</div>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"<div class='metric-chip'><span>Latency</span> {decision['latency_ms']} ms</div>",
            unsafe_allow_html=True,
        )
        col3.markdown(
            f"<div class='metric-chip'><span>Risk</span> {decision['risk_score']:.2f}</div>",
            unsafe_allow_html=True,
        )
        col4.markdown(
            f"<div class='metric-chip'><span>Tokens</span> {decision['tokens_in']} → {decision['tokens_out']}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Decision details", expanded=False):
        risk_percentage = f"{int(decision['risk_score'] * 100)}%"
        st.markdown(
            f"<div class='risk-meter' style='--risk-level:{int(decision['risk_score'] * 100)}%;'></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Risk meter (0-1 scale): {risk_percentage}")

        categories = ["Cost", "Latency", "Depth"]
        fast_vals = [
            metrics_map[FAST]["cost"],
            metrics_map[FAST]["latency"],
            metrics_map[FAST]["depth"],
        ]
        reasoning_vals = [
            metrics_map[REASONING]["cost"],
            metrics_map[REASONING]["latency"],
            metrics_map[REASONING]["depth"],
        ]

        fig, ax = plt.subplots(figsize=(4, 2.8))
        x_positions = range(len(categories))
        width = 0.35
        ax.bar([x - width / 2 for x in x_positions], fast_vals, width, label="FAST")
        ax.bar([x + width / 2 for x in x_positions], reasoning_vals, width, label="REASONING")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(categories)
        ax.set_ylabel("Relative units")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Why we routed this way**")
        for reason in decision["why"]:
            st.markdown(f"- {reason}")
        if decision["citations"]:
            st.caption("; ".join(decision["citations"]))

    answer_tab, summary_tab, trace_tab = st.tabs(["Answer", "Summary", "Trace"])
    answer_text = decision["answer"]
    esc_answer = json.dumps(answer_text)
    with answer_tab:
        st.markdown(
            f"""
            <div class="message-bubble">
                <button onclick="navigator.clipboard.writeText({esc_answer});">📋</button>
                {answer_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_tab:
        st.markdown("\n".join(f"- {point}" for point in decision["summary_points"]))
    with trace_tab:
        st.json(decision)


def render_model_cards(selected_track: str) -> None:
    """Render comparison cards for the FAST and REASONING lanes."""

    metrics_map = get_track_metrics()
    for track in (FAST, REASONING):
        info = MODEL_INFO[track]
        metrics = metrics_map[track]
        classes = "model-card selected" if track == selected_track else "model-card"
        strengths_markup = "".join(f"<li>{item}</li>" for item in info["strengths"])
        weaknesses_markup = "".join(f"<li>{item}</li>" for item in info["weaknesses"])
        st.markdown(
            f"""
            <div class="{classes}">
                <h4 style="margin-bottom:0.2rem; color:{COLORS['text']};">{info['label']}</h4>
                <p style="font-size:0.8rem; color:{COLORS['subtle']}; margin-top:0;">{metrics['model']}</p>
                <p style="font-size:0.8rem; margin-bottom:0.4rem;">
                    Cost ≈ ${metrics['cost']:.4f}/request · Latency ≈ {metrics['latency']}ms · Depth score {metrics['depth']:.2f}
                </p>
                <div style="display:flex; gap:1rem;">
                    <div style="flex:1;">
                        <strong>Strengths</strong>
                        <ul style="padding-left:1.1rem; margin-top:0.35rem; font-size:0.8rem;">{strengths_markup}</ul>
                    </div>
                    <div style="flex:1;">
                        <strong>Trade-offs</strong>
                        <ul style="padding-left:1.1rem; margin-top:0.35rem; font-size:0.8rem;">{weaknesses_markup}</ul>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**FAST vs REASONING**")
    comparison_rows = [
        {
            "Metric": "Cost",
            "FAST": f"${metrics_map[FAST]['cost']:.4f}",
            "REASONING": f"${metrics_map[REASONING]['cost']:.4f}",
        },
        {
            "Metric": "Speed",
            "FAST": f"~{int(metrics_map[FAST]['latency'])} ms",
            "REASONING": f"~{int(metrics_map[REASONING]['latency'])} ms",
        },
        {
            "Metric": "Depth",
            "FAST": "Transactional",
            "REASONING": "Analytical",
        },
    ]
    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)


def download_buttons(data: List[Dict[str, Any]], filename_prefix: str) -> None:
    """Render CSV and JSON download buttons for the provided dataset."""

    if not data:
        return
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(data[0].keys()))
    writer.writeheader()
    writer.writerows(data)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    json_bytes = json.dumps(data, indent=2).encode("utf-8")

    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_json:
        st.download_button(
            label="Download JSON",
            data=json_bytes,
            file_name=f"{filename_prefix}.json",
            mime="application/json",
            use_container_width=True,
        )


def render_logs(logs: List[Dict[str, Any]]) -> None:
    """Display the session audit logs with filter, pagination, and downloads."""

    st.markdown("## Audit logs")
    if not logs:
        st.info("No runs logged yet. Run the demo to generate an audit trail.")
        return

    search_term = st.text_input("Search logs", placeholder="Filter by keyword, track, or model...")
    filtered = logs
    if search_term:
        lowered = search_term.lower()
        filtered = [
            row for row in logs if lowered in json.dumps(row).lower()
        ]

    page_size = st.selectbox("Rows per page", options=[5, 10, 20], index=1)
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]

    st.dataframe(paginated, use_container_width=True, hide_index=True)
    download_buttons(filtered, "rad-ai-routing-logs")

    if st.button("Clear logs", type="secondary"):
        st.session_state.logs.clear()
        st.success("Audit trail cleared for this session.")


def reset_state() -> None:
    """Reset form inputs to their defaults without clearing logs."""

    st.session_state.scenario = "Custom prompt"
    st.session_state.message = ""
    st.session_state.route_mode = "Auto (recommended)"
    st.session_state.stakes = "Standard"
    st.session_state.expected_output = "Answer"
    st.session_state.decision = None
    st.session_state.scroll_to_input = True


def ensure_state_initialized() -> None:
    """Prime session state with default values if missing."""

    if "initialized" in st.session_state:
        return
    st.session_state.logs = []
    st.session_state.message = ""
    st.session_state.scenario = "Custom prompt"
    st.session_state.route_mode = "Auto (recommended)"
    st.session_state.stakes = "Standard"
    st.session_state.expected_output = "Answer"
    st.session_state.decision = None
    st.session_state.is_running = False
    st.session_state.scroll_to_input = False
    live_secret = str(st.secrets.get("LIVE_MODE", "false")).lower()
    api_key_available = bool(st.secrets.get("OPENAI_API_KEY"))
    st.session_state.live_mode = api_key_available and live_secret in {"1", "true", "yes", "on"}
    update_track_metrics(st.session_state.live_mode)
    st.session_state.initialized = True


def main() -> None:
    """Entry point for the Streamlit app."""

    inject_css()
    ensure_state_initialized()
    cases = load_canned_cases()
    st.session_state.cases = cases

    api_key_available = bool(st.secrets.get("OPENAI_API_KEY"))
    with st.sidebar:
        st.subheader("Run configuration")
        if api_key_available:
            live_toggle = st.toggle(
                "Use LIVE models",
                value=st.session_state.live_mode,
                key="live_mode_toggle",
                help="Requires OPENAI_API_KEY in Streamlit secrets",
            )
            if live_toggle != st.session_state.live_mode:
                st.session_state.live_mode = live_toggle
                update_track_metrics(live_toggle)
        else:
            st.info("Add OPENAI_API_KEY to secrets to enable live LLM responses.")
            if st.session_state.live_mode:
                st.session_state.live_mode = False
                update_track_metrics(False)
        st.caption("FAST lane ≈ low-cost triage, REASONING lane ≈ depth & risk coverage.")

    st.markdown(
        """
        <div class="rad-header">
            <h1 style="margin-bottom:0; color:#FDF4FF;">RAD AI – Routing Agent</h1>
            <span class="rad-pill">MVP Demo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Investor showcase: route every customer message to the smartest, safest lane — complete with transparent decisioning and audit trail.")
    if is_live_mode():
        st.markdown(
            f"<p style='color:{COLORS['success']}; font-size:0.85rem;'>LIVE mode active — responses generated by {get_track_metrics()[FAST]['model']} / {get_track_metrics()[REASONING]['model']}.</p>",
            unsafe_allow_html=True,
        )
    elif not api_key_available:
        st.markdown(
            f"<p style='color:{COLORS['subtle']}; font-size:0.85rem;'>LIVE mode locked — add an OPENAI_API_KEY in Streamlit secrets to enable real model calls.</p>",
            unsafe_allow_html=True,
        )

    with st.expander("What this demo shows"):
        st.markdown(
            """
            - Auto-selects between FAST (cost/speed) and REASONING (depth/risk) tracks
            - Explains *why* a route was chosen with metrics, charts, and traceability
            - Captures every run in a downloadable audit log for compliance-ready handoffs
            """
        )

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        def on_scenario_change() -> None:
            selected = st.session_state.scenario
            if selected != "Custom prompt" and selected in st.session_state.cases:
                scenario = st.session_state.cases[selected]
                st.session_state.message = scenario["message"]
                st.session_state.stakes = scenario.get("stakes", "Standard")
                st.session_state.expected_output = scenario.get("expected", "Answer")
            elif selected == "Custom prompt":
                st.session_state.message = ""
                st.session_state.stakes = "Standard"
                st.session_state.expected_output = "Answer"
            st.session_state.scroll_to_input = True

        st.selectbox(
            "Scenario",
            options=["Custom prompt", *cases.keys()],
            key="scenario",
            on_change=on_scenario_change,
            disabled=st.session_state.is_running,
        )

        with st.form("routing-form"):
            st.text_area(
                "Customer message",
                key="message",
                height=160,
                placeholder="Paste or type the customer message here...",
                disabled=st.session_state.is_running,
            )
            mode = st.selectbox(
                "Routing mode",
                options=["Auto (recommended)", "Force FAST", "Force REASONING"],
                key="route_mode",
            )
            stakes = st.selectbox(
                "Stakes",
                options=["Low", "Standard", "High"],
                key="stakes",
                help="High = safety-sensitive or high-liability scenario",
            )
            expected = st.selectbox(
                "Expected output",
                options=["Answer", "Summary", "Step-by-step", "Action plan"],
                key="expected_output",
            )

            run_col, clear_col = st.columns([1, 1])
            run_clicked = run_col.form_submit_button(
                "Run demo",
                use_container_width=True,
                disabled=st.session_state.is_running,
                help="Shortcut: Ctrl/Cmd + Enter",
            )
            clear_clicked = clear_col.form_submit_button(
                "Clear",
                use_container_width=True,
                type="secondary",
                disabled=st.session_state.is_running,
            )

        if clear_clicked:
            reset_state()
            st.experimental_rerun()

        message = st.session_state.message
        sensitive_flag, extreme_flag = detect_sensitive_flags(message)
        if sensitive_flag:
            st.markdown(
                "<div class='safety-banner'>This demo is not medical or legal advice. In real deployments we escalate to human review.</div>",
                unsafe_allow_html=True,
            )
        if extreme_flag:
            st.markdown(
                "<div class='emergency-banner'>Call emergency services immediately for urgent safety issues.</div>",
                unsafe_allow_html=True,
            )

        if run_clicked and not st.session_state.is_running:
            if not message.strip():
                st.warning("Add a customer message to run the demo.")
            else:
                st.session_state.is_running = True
                with st.spinner("Routing demo in progress…"):
                    progress_container = st.container()
                    status_placeholder = progress_container.empty()
                    progress_bar = progress_container.progress(0)

                    decision: Dict[str, Any] | None = None
                    for index, step in enumerate(PROGRESS_STEPS, start=1):
                        status_placeholder.markdown(f"**{step}…**")
                        if step == "Selecting track":
                            decision = route_request(message, stakes, expected, mode)
                        elif step == "Summarizing" and decision:
                            time.sleep(0.1)
                        time.sleep(0.15)
                        progress_bar.progress(index / len(PROGRESS_STEPS))

                if decision is None:
                    decision = route_request(message, stakes, expected, mode)

                response_started = time.perf_counter()
                response_payload = get_response(message, decision["track"], expected)
                latency_ms = max(1, int((time.perf_counter() - response_started) * 1000))

                decision["answer"] = response_payload["answer"]
                decision["model"] = response_payload["model"]
                decision["tokens_out"] = response_payload["tokens_out"]
                decision["cost_usd"] = response_payload["cost_usd"]
                decision["latency_ms"] = latency_ms

                tokens_in_actual = response_payload.get("tokens_in")
                if tokens_in_actual:
                    decision["tokens_in"] = tokens_in_actual

                note = response_payload.get("notes")
                if note:
                    decision["citations"].append(note)
                elif is_live_mode():
                    decision["citations"].append("Live model invocation (OpenAI)")

                decision["summary_points"] = build_summary_points(decision)

                log_entry = {
                    "ts_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "track": decision["track"],
                    "model": decision["model"],
                    "stakes": stakes,
                    "expected_type": expected,
                    "tokens_in": decision["tokens_in"],
                    "tokens_out": decision["tokens_out"],
                    "cost_usd": decision["cost_usd"],
                    "risk_score": decision["risk_score"],
                    "message_preview": (message[:120] + "…") if len(message) > 120 else message,
                }
                st.session_state.logs.append(log_entry)
                st.session_state.decision = decision

                progress_bar.progress(1.0)
                time.sleep(0.1)
                progress_container.empty()
                st.session_state.is_running = False

                if decision["risk_score"] >= 0.6:
                    st.warning("High-risk scenario routed to REASONING lane.")
                else:
                    st.success("Run completed and logged.")

        if st.session_state.decision:
            render_route_card(st.session_state.decision)

    with right_col:
        render_model_cards(
            st.session_state.decision["track"] if st.session_state.decision else FAST
        )

    render_logs(st.session_state.logs)

    st.markdown(
        """
        <div class="footer">
            RAD AI © — Not medical/legal advice. For demo only.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("scroll_to_input"):
        components.html(
            """
            <script>
                const textarea = window.parent.document.querySelector('textarea');
                if (textarea) { textarea.scrollIntoView({behavior: 'smooth', block: 'center'}); }
            </script>
            """,
            height=0,
        )
        st.session_state.scroll_to_input = False


if __name__ == "__main__":
    main()
