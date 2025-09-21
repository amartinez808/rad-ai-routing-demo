# rad-ai-routing-demo

RAD AI routing demo for Hackathon Providence.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Enable LIVE mode (optional)
- Add an OpenAI API key to `.streamlit/secrets.toml` on Streamlit Cloud (or `export OPENAI_API_KEY=...` locally).
- Optional overrides:
  - `LIVE_MODE=true` to start with live models enabled.
  - `FAST_MODEL_ID` / `REASONING_MODEL_ID` if you want to swap the default `gpt-4o-mini` / `gpt-4o` pairing.
- The sidebar toggle lets you switch between mock responses and live calls at runtime.

## Deploy to itsradai.com

The repo now includes a production-ready Docker stack (`Dockerfile`, `docker-compose.yml`, `deployment/nginx.conf`) so you can serve the Streamlit app from `https://itsradai.com`.

### 1. Prep a host
- Use any Linux VM (Ubuntu 22.04+, Debian 12, etc.) with ports 80/443 open.
- Install Docker Engine and Docker Compose Plugin.

### 2. Point DNS to the host
- Create an `A` record for `itsradai.com` (and optionally `www`) pointing to the VM's public IP.
- Allow time for propagation (usually <1 hour).

### 3. Clone and launch the stack
```bash
# on the server
sudo git clone https://github.com/<your-org>/rad-ai-routing-demo.git /opt/rad-ai-routing-demo
cd /opt/rad-ai-routing-demo
sudo docker compose up -d --build
```
- The `app` service runs Streamlit on the internal Docker network.
- The `nginx` service terminates HTTP on port 80 and proxies requests to the app.

### 4. Add TLS (strongly recommended)
- Install certbot: `sudo apt install certbot python3-certbot-nginx`.
- Issue a certificate: `sudo certbot --nginx -d itsradai.com -d www.itsradai.com`.
- Certbot updates the nginx config in-place and handles renewals.

### 5. Keep the stack healthy
- View logs: `sudo docker compose logs -f`
- Rebuild after code updates: `sudo docker compose up -d --build`
- Update certs monthly with `certbot renew` (automated via systemd timer).

### Customise if needed
- Edit `deployment/nginx.conf` if you want to serve the app from a sub-path or change domains.
- To pin a specific Streamlit version, set it in `requirements.txt` before rebuilding.

## Repo layout
- `streamlit_app.py` – the routing demo UI and heuristics.
- `Dockerfile` – container image for the Streamlit app.
- `docker-compose.yml` – runs the app plus the nginx reverse proxy.
- `deployment/nginx.conf` – nginx virtual host configuration for `itsradai.com`.
