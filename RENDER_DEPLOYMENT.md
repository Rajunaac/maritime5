# 🚀 Maritime AI Sentinel — Render Deployment Guide

This project is fully configured and ready for **1-click deployment on [Render](https://render.com/)**.

---

## 📁 What Has Been Prepared in This Repository

| File | Purpose |
| :--- | :--- |
| **[`server.py`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/server.py)** | Production HTTP server supporting Render's dynamic `$PORT` environment variable, streaming large assets (`samples.js`, `maritime.mp4`), CORS headers, and live REST API endpoints (`/api/status`, `/api/state`, `/api/step`, `/api/set_route`, `/api/classify`, `/api/cyberattack`, `/api/replan`). |
| **[`requirements.txt`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/requirements.txt)** | Production dependencies optimized with CPU-only PyTorch and NumPy (fast install that runs comfortably within Render's 512MB RAM free tier). |
| **[`requirements-minimal.txt`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/requirements-minimal.txt)** | Ultra-lean configuration (NumPy only) with automatic high-accuracy analytical feature classification fallback. |
| **[`render.yaml`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/render.yaml)** | Official Render Infrastructure-as-Code Blueprint for automated zero-configuration deployment. |
| **[`Procfile`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/Procfile)** | Process declaration file (`web: python server.py`) for Render process management. |
| **[`.gitignore`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/.gitignore)** | Excludes heavy dataset archives (`.7z`, `.npz`, >100MB files) so `git push` succeeds without exceeding GitHub's 100MB file limit. |
| **[`best_custom_cnn.pth`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/best_custom_cnn.pth)** | Pre-trained PyTorch Custom CNN model weights (973 KB) bundled right in the root folder. |
| **[`maritime_engine.py`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/maritime_engine.py)** | End-to-end maritime threat engine (SAR inference, cyclone drift, CPA/TCPA, dynamic $A^*$ routing, MarineChain PoAh ledger). |
| **[`Dockerfile`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/Dockerfile)** & **[`.dockerignore`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/.dockerignore)** | Production container specification if you choose Docker deployment on Render. |

---

## 🛠️ Step-by-Step Deployment Instructions

### Step 1: Push Your Code to GitHub

Open a terminal in this folder and push to your GitHub account:

```bash
git init
git add .
git commit -m "Initial commit: Ready for Render deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

*(Thanks to the configured `.gitignore`, huge datasets like `test.json.7z` and `train.npz` are skipped automatically so your git push will succeed in seconds!)*

---

### Step 2: Deploy on Render

Choose **one** of the methods below:

#### Method A: Render Web Service (Recommended — Full AI Backend + Frontend)
1. Go to [dashboard.render.com](https://dashboard.render.com/) and sign in.
2. Click **New +** → **Web Service**.
3. Select your GitHub repository.
4. Fill in the service configuration:
   - **Name**: `maritime-ai-sentinel` *(or any name you prefer)*
   - **Region**: Any region close to you (e.g., *Oregon (US West)* or *Frankfurt (EU)*)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Plan**: `Free`
5. Click **Create Web Service**.

---

#### Method B: 1-Click Blueprint (Using `render.yaml`)
1. In the Render Dashboard, click **New +** → **Blueprint**.
2. Connect your GitHub repository.
3. Render automatically detects [`render.yaml`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/render.yaml) and pre-fills all build commands, start commands, health checks, and environment settings.
4. Click **Apply**.

---

#### Method C: Render Static Site (Free Global CDN, Frontend UI Only)
If you only need the interactive client-side tactical radar dashboard:
1. Click **New +** → **Static Site**.
2. Connect your repository.
3. Set:
   - **Build Command**: `echo "Assets ready"`
   - **Publish Directory**: `.`
4. Click **Create Static Site**.

---

#### Method D: Docker Web Service
1. Click **New +** → **Web Service**.
2. Select your repository.
3. Set **Runtime** to `Docker`.
4. Render will automatically build the included [`Dockerfile`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/Dockerfile).

---

## 🌐 Verifying Your Deployed Application

Once the build finishes (usually 1–2 minutes), your application will be live at:

```
https://<your-service-name>.onrender.com/
```

### Key Endpoints:
- **Main Interactive Dashboard**: `https://<your-service-name>.onrender.com/`
- **Documentation & Technical Portal**: `https://<your-service-name>.onrender.com/landing_documentation.html`
- **System Healthcheck API**: `https://<your-service-name>.onrender.com/api/status`
- **Live State Telemetry**: `https://<your-service-name>.onrender.com/api/state`
- **Ports & Presets Registry**: `https://<your-service-name>.onrender.com/api/ports`
- **Ablation Studies Data**: `https://<your-service-name>.onrender.com/api/ablation`

---

## 💡 Important Notes for Render Free Tier

1. **Dynamic Port Binding**:
   Render automatically assigns an internal port via `$PORT` (typically 10000). [`server.py`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/server.py) is already programmed to read `os.environ.get("PORT")` and bind to `0.0.0.0` automatically.
2. **Free Tier Inactivity Sleep**:
   On Render's Free tier, services spin down after 15 minutes of inactivity. When a new visitor opens the URL, Render wakes the service up within ~30–50 seconds.
3. **Healthcheck**:
   Render uses `/api/status` (configured in [`render.yaml`](file:///c:/Users/HP/Downloads/statoil-iceberg-classifier-challenge/render.yaml)) as the zero-downtime healthcheck endpoint.
