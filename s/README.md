# AI Business Intelligence Report Assistant

A zero-dependency Python model and interactive web dashboard that converts uploaded sales and business reports (`.csv`, `.tsv`, `.json`, `.txt`, `.md`) into structured insights.

## Features
- **Sign In / Create Account + 1-Click Demo Workspace**: SQLite-backed user authentication (`bi_users.db`, PBKDF2-SHA256) with isolated per-session report workspaces for multi-user online hosting.
- **Upload Sales & Business Reports**: Supports tabular sales sheets (`CSV`, `TSV`, `JSON`) and narrative reports (`TXT`, `MD`).
- **Retrieve Relevant Business Information**: TF-IDF + phrase-boosted retrieval engine (`BusinessIntelligenceModel.retrieve` & `answer_question` in `Ai_assistant.py`) with cited source snippets.
- **Summarize Performance & Key Findings**: Aggregates revenue, profit, profit margin, units, average growth, top category, and top channel.
- **Challenge — Best-Performing Region & Explanation**: Ranks regions across revenue, profit, margin, and growth, then combines quantitative drivers with retrieved qualitative evidence from narrative reports to explain *why* the top region won.
- **Generate Recommendations & Export**: Produces prioritized, report-grounded business actions and lets you export insights as a Markdown report (`.md`).

## Quick Start (Local)

### 1. Run the CLI Model on Sample Reports
```powershell
python3 Ai_assistant.py
```

### 2. Launch the Interactive Web Dashboard
```powershell
python3 server.py
```
Then open `http://127.0.0.1:8000` in your browser, click **1-Click Demo Workspace** (or create an account), and click **Load Sample Reports** or upload your own files.

### 3. Run Tests
```powershell
python3 -m unittest test_bi_assistant.py -v
```

## Deploy to Render

This repository includes [render.yaml](render.yaml) and [requirements.txt](requirements.txt) for 1-click or manual Render deployment:

1. Push this folder to a **GitHub** (or GitLab) repository.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** -> **Web Service** (or **Blueprint** to auto-read `render.yaml`).
3. Connect your repository and verify these settings:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Health Check Path**: `/health`
4. Click **Deploy Web Service**. Render will automatically provide `PORT` and route HTTPS traffic to your live URL.

> **Note on Render Free Tier:** Ephemeral disk storage resets when a free instance restarts, so newly created user accounts in `bi_users.db` reset on redeploy unless you attach a Render Persistent Disk and set `BI_DB_PATH=/var/data/bi_users.db`. Visitors can always use **1-Click Demo Workspace** immediately.
