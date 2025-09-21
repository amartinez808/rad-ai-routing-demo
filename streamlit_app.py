import os, time, json
from typing import Optional, Dict, List
import requests
import streamlit as st

# ----------------------------
# App shell
# ----------------------------
st.set_page_config(page_title="RAD AI – Multi-Model Demo", page_icon="🤖", layout="centered")
st.title("RAD AI – Multi-Model Demo")
st.caption("OpenAI (paid) + actually-free tiers: Groq (Llama), Together (Llama 3.3 70B FREE), and Google Gemini.")

# ----------------------------
# Secrets helper
# ----------------------------
def get_secret(key: str) -> Optional[str]:
    return st.secrets.get(key) or os.getenv(key)

# ----------------------------
# Providers & models
# ----------------------------
PROVIDERS: Dict[str, List[str]] = {
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-4.1-mini"
    ],
    "Groq (Llama)": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ],
    "Together (Llama 3.3 70B FREE)": [
        "meta-llama/llama-3.3-70b-instruct-free"
    ],
    "Gemini (free tier)": [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro-latest"
    ],
}

# ----------------------------
# Provider call functions
# ----------------------------
def _friendly_http_error(resp: requests.Response) -> str:
    code = resp.status_code
    brief = resp.text[:300]
    if code == 401:
        return "Unauthorized (check API key)"
    if code == 403:
        return "Forbidden (key lacks access or model blocked)"
    if code == 429:
        return "Rate limited/Quota exceeded"
    if 500 <= code < 600:
        return f"Provider server error {code}"
    return f"HTTP {code}: {brief}"


def call_openai(model: str, prompt: str, temperature: float) -> str:
    # Notes: paid provider; requires OPENAI_API_KEY
    from openai import OpenAI
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Be concise and helpful."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def call_groq(model: str, prompt: str, temperature: float) -> str:
    # Notes: actually-free/dev tier available; requires GROQ_API_KEY
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Be concise and helpful."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    except requests.RequestException as ex:
        raise RuntimeError(f"Groq network error: {ex}") from ex
    if r.status_code != 200:
        raise RuntimeError(f"Groq: {_friendly_http_error(r)}")
    return r.json()["choices"][0]["message"]["content"].strip()


def call_together(model: str, prompt: str, temperature: float) -> str:
    # Notes: actually-free endpoint: meta-llama/llama-3.3-70b-instruct-free; requires TOGETHER_API_KEY
    api_key = get_secret("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TOGETHER_API_KEY")
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Be concise and helpful."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    except requests.RequestException as ex:
        raise RuntimeError(f"Together network error: {ex}") from ex
    if r.status_code != 200:
        raise RuntimeError(f"Together: {_friendly_http_error(r)}")
    return r.json()["choices"][0]["message"]["content"].strip()


def call_gemini(model: str, prompt: str, temperature: float) -> str:
    # Notes: Google free tier; requires GEMINI_API_KEY
    import google.generativeai as genai
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    try:
        resp = model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
    except Exception as ex:  # SDK surfaces errors directly
        raise RuntimeError(f"Gemini error: {ex}") from ex
    if not getattr(resp, "text", None):
        raise RuntimeError("Gemini returned empty response.")
    return resp.text.strip()


def generate(provider_label: str, model: str, prompt: str, temperature: float) -> str:
    pl = provider_label.lower()
    if "openai" in pl:
        return call_openai(model, prompt, temperature)
    if "groq" in pl:
        return call_groq(model, prompt, temperature)
    if "together" in pl:
        return call_together(model, prompt, temperature)
    if "gemini" in pl:
        return call_gemini(model, prompt, temperature)
    raise RuntimeError("Unknown provider")


# ----------------------------
# UI
# ----------------------------
with st.expander("How this works", expanded=False):
    st.markdown("""
- **OpenAI** uses your paid key; the others are **actually free tiers** (expect rate limits).
- If one throttles, use the fallback buttons to try another provider.
- We don’t store prompts; this is a live demo client.
""")

col1, col2 = st.columns(2)
provider_label = col1.selectbox("Provider", list(PROVIDERS.keys()))
model = col2.selectbox("Model", PROVIDERS[provider_label])
default_prompt = "In 5 bullets, pitch RAD AI (Rational Automation Design) to hackathon judges: problem, solution, why now, traction, ask."
prompt = st.text_area("Your prompt", default_prompt, height=140)
temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

run = st.button("Run on selected provider")

# Fallback order rotates after the selected provider
order = list(PROVIDERS.keys())
primary_idx = order.index(provider_label)
fallback_cycle = order[primary_idx+1:] + order[:primary_idx]

if run:
    start = time.perf_counter()
    try:
        with st.spinner("Calling model..."):
            out = generate(provider_label, model, prompt, temperature)
        latency = (time.perf_counter() - start) * 1000
        st.success(f"Done in {latency:.0f} ms via {provider_label} · {model}")
        st.markdown("**Response:**")
        st.write(out)
    except Exception as e:
        st.error(f"{provider_label} failed: {e}")
        st.info("Try a fallback provider:")
        for fb in fallback_cycle[:2]:
            if st.button(f"Retry on {fb}", key=f"fallback-{fb}"):
                try:
                    fb_model = PROVIDERS[fb][0]
                    start2 = time.perf_counter()
                    with st.spinner(f"Calling {fb}..."):
                        out2 = generate(fb, fb_model, prompt, temperature)
                    latency2 = (time.perf_counter() - start2) * 1000
                    st.success(f"Done in {latency2:.0f} ms via {fb} · {fb_model}")
                    st.markdown("**Response:**")
                    st.write(out2)
                except Exception as e2:
                    st.error(f"{fb} also failed: {e2}")

st.divider()
st.markdown("### Connection checklist")
st.write("- 🔑 **OpenAI** needs `OPENAI_API_KEY`")
st.write("- 🆓 **Groq** needs `GROQ_API_KEY` (dev/free tier available)")
st.write("- 🆓 **Together** needs `TOGETHER_API_KEY` (Llama 3.3 70B **FREE** endpoint)")
st.write("- 🆓 **Gemini** needs `GEMINI_API_KEY` (free tier available)")
st.caption("Put keys in `.streamlit/secrets.toml` (do not commit).")
