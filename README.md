# BreathMap (TartanHacks 2026).

BreathMap is a lightweight web app that analyzes short performance videos to extract **breathing signals from pose motion** and optionally detect **singing strain/collapse** from audio. It returns a single JSON report + diagnostic artifacts that the frontend renders into an interactive timeline.

---

## Repo Structure

TartanHacks26/
backend/ # FastAPI server + video/audio analysis pipeline
frontend/ # React + TypeScript UI

## Demo Flow

1. Upload a short `.mp4` / `.mov` clip in the frontend
2. Backend saves the upload and runs analysis
3. Frontend renders:
   - replayable video
   - breathing waveform + inhale markers
   - per-second stability/quality timeline
   - summary + annotations
   - optional singing/strain results
   - optional “where to breathe” feedback


## Backend API

### `POST /api/analyze`

Accepts:
- `multipart/form-data` with a video file

Returns one JSON payload containing:
- `video_id`
- `videoUrl` (static replay URL)
- `plot_url` (diagnostic PNG)
- `timelineScores`
- `annotations`
- `summary`
- `shortExplanation`
- `report` (raw breathing arrays/events)
- optional `singing_analysis`
- optional `breath_feedback`

## Static File Serving

The backend serves runtime files directly:

- `/media/...` → uploaded videos (stored in `uploads/`)
- `/outputs/...` → generated artifacts (stored in `outputs/`)

Examples:
- `http://localhost:8000/media/<video_id>.mp4`
- `http://localhost:8000/outputs/breath_plot_<video_id>.png`

## Running Locally

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Backend runs at:
http://localhost:8000


### Frontend

```bash
cd frontend
npm install
npm run dev
```
