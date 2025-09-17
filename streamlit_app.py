import time, re, json, io
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

def needs_reasoning(text: str) -> bool:
    text_l = text.lower()
    # math / analytical signals
    mathy = bool(re.search(r"\d+\s*[-+*/^]\s*\d+|integral|derivative|equation|optimi[sz]e|complexity", text_l))
    # code / debugging signals
    codey = any(k in text_l for k in ["stack trace", "exception", "traceback", "python", "sql", "snippet"])
    # complexity signals
    long_enough = len(text) > 180 or text.count("?") > 1
    multi_step  = any(k in text_l for k in ["first", "second", "step", "steps", "plan", "workflow"])
    investigative = any(k in text_l for k in ["why", "root cause", "analysis", "compare", "tradeoff"])
    high_stakes = any(k in text_l for k in ["legal", "contract", "security", "privacy", "compliance", "finance"])
    return long_enough or mathy or codey or multi_step or investigative or high_stakes

def choose_route(text: str, mode: str) -> str:
    if mode == "Force FAST":
        return FAST
    if mode == "Force REASONING":
        return REASONING
    # Auto
    return REASONING if needs_reasoning(text) else FAST

def mock_fast_answer(text: str) -> str:
    return ("Here’s a quick, helpful reply (FAST lane). "
            "I’ve summarized your request and next steps to keep response time low. "
            "If you need deeper investigation, escalate to the REASONING lane.")

def mock_reasoning_answer(text: str) -> str:
    return ("Deep-dive analysis (REASONING lane):\n"
            "1) Interpreting the request and constraints\n"
            "2) Laying out assumptions and edge cases\n"
            "3) Proposing a structured resolution plan\n"
            "4) Risks & follow-ups\n"
            "→ This lane trades latency for depth & accuracy.")

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
    st.session_state.logs = []

def append_log(prompt_text: str, route: str, tokens_est: int):
    st.session_state.logs.append({
        "ts_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "route": route,
        "chars": len(prompt_text),
        "tokens_est": tokens_est,
    })

def estimate_tokens(text: str) -> int:
    # rough token estimation ~ 4 chars per token
    return max(1, len(text)//4)

# ----------------------------
# Run
# ----------------------------
if run_btn:
    if not prompt.strip():
        st.warning("Add a prompt to route.")
    else:
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
            st.code(needs_reasoning.__code__.co_consts[0] if needs_reasoning.__code__.co_consts else "Rule-based heuristics", language="text")
            st.markdown("- Long/complex, code/math, multi-step, investigative/why, high-stakes → **REASONING**\n- Otherwise → **FAST**")

        append_log(prompt, route, tokens)

# ----------------------------
# Logs table + download
# ----------------------------
st.subheader("Live Logs")
if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download logs (CSV)", csv, file_name="radai_routing_logs.csv", mime="text/csv")
    if st.button("Clear logs"):
        st.session_state.logs = []
        st.rerun()
else:
    st.info("Run a few prompts to generate logs you can include in your evidence packet.")
