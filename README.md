# Cricket Ball Tracker — React UI + FastAPI service

Implementation of the June 2026 spec. Skips the PitchMap SVG (per user
request); everything else from the spec is in place.

```
tracker_app/
├── backend/
│   ├── pipeline/           # headless process_video() + state-isolated detector
│   ├── tests/              # double-run isolation test (skips without fixtures)
│   ├── app.py              # FastAPI routes
│   ├── jobs.py             # in-process job store + execution lock
│   ├── storage.py          # LocalStorage (dev) / GCSStorage (prod)
│   └── requirements.txt
└── frontend/               # Vite + React + TS
    ├── src/features/tracking/
    │   ├── TrackingPage.tsx        ← /tracking (Upload → Calibrate → Review)
    │   ├── JobDetailPage.tsx       ← /tracking/jobs/:jobId
    │   ├── CalibrationCanvas.tsx   ← 6-marker placement with magnifier loupe
    │   ├── ResultsView.tsx         ← video + event cards + downloads
    │   └── ...
    └── package.json
```

## What was built

| Spec § | Status |
|---|---|
| 3. Pipeline refactor (RunState, no module globals) | ✅ `pipeline/state.py`, `filters_v2.py`, `detector_v2.py` |
| 3.6 Double-run isolation test | ✅ `backend/tests/test_pipeline_isolation.py` (skipped without fixtures) |
| 4. Results JSON contract | ✅ Emitted by `pipeline/process.py` |
| 5. FastAPI routes | ✅ `app.py` — `/first-frame`, `/analyze`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/result`, `/jobs/{id}/cancel` |
| 5.4 Single-job lock | ✅ `JobStore.exec_lock` (threading.Lock) |
| 6. React UI | ✅ Upload + Calibration + Review + JobDetail + JobList |
| 6.4 Calibration canvas | ✅ Guided order, draggable markers, magnifier loupe, validation |
| 6.6 Polling hook | ✅ `useJobPolling.ts` (2s fast, 5s after 2min, abort on unmount) |
| 7. TypeScript interfaces | ✅ `features/tracking/types.ts` |
| 8. Storage | ✅ Pluggable (LocalStorage default, GCSStorage when `GCS_BUCKET` set) |
| 9. H.264 + faststart re-encode | ✅ `_reencode_h264()` in `pipeline/process.py` |
| 10. Error code mapping | ✅ `PipelineError` subclasses → JSON `error.code` → `ERROR_MESSAGES` |
| 11. Env vars | ✅ See below |
| **PitchMap SVG** | ❌ Skipped per request |

## Reconciliation notes (spec vs reality)

These differ from the spec's assumptions about the source files — see
`pipeline/process.py` for the implementation:

- `BounceDetector` lives in `tracker.py` (not `impact_detector.py`).
  `impact_detector.py` is the stub spec §2.1 describes; `events.impact` is
  always `null`.
- The "impact" event in the UI maps to nothing today; the deviation point
  (`Trajectory._dir_changes[0]`) drives the SEG C frozen-frame animation and
  is not surfaced in the JSON contract. If you want to expose it, add a
  separate `events.deviation` field.
- Original `main_V2.1.py` is a 4-pass renderer (analysis → SEG A → SEG B
  0.5× replay → SEG C frozen-frame). The headless pipeline preserves this
  but **gracefully degrades** when events are missing: SEG B requires
  release+bounce, SEG C requires release+bounce+deviation. SEG A always
  renders (the basic annotated walkthrough).
- The legacy `cv2.imshow`/`namedWindow`/`waitKey` calibration UI is gone —
  calibration is now data passed via the API.

## Environment

```
TRACKER_WEIGHTS_PATH=D:\intership\may\version 2\model_weights\best.onnx
TRACKER_LEGACY_DIR=D:\intership\June\working perfection\Zone-div(final)
TRACKER_WORK_DIR=./_tracker_work
GCS_BUCKET=                     # optional — enables GCSStorage when set
SIGNED_URL_TTL_SECONDS=3600
MAX_UPLOAD_MB=500
```

Frontend `.env`:

```
VITE_API_BASE=/api
```

## Dev run

Backend (Python 3.10+):

```powershell
cd D:\intership\tracker_app\backend
pip install -r requirements.txt
# ffmpeg must be on PATH for H.264 re-encoding
uvicorn app:app --reload --port 8000
```

Frontend (Node 18+):

```powershell
cd D:\intership\tracker_app\frontend
npm install
npm run dev
# http://localhost:5173  (proxies /api to localhost:8000)
```

## Tests

```powershell
cd D:\intership\tracker_app\backend
pytest tests/  # double-run isolation test; skips without model weights + test clip
```

## Cloud Run deployment (us-central1 + L4 GPU)

Single Cloud Run service serves the React bundle and the FastAPI backend.
Frontend is built into the image; legacy tracker folders and ONNX weights
(white + red) are staged from disk and baked into the image too.

### One-time setup

```powershell
$PROJECT = "your-gcp-project-id"
$BUCKET  = "your-tracker-bucket"   # globally unique

gcloud auth login
gcloud config set project $PROJECT

gcloud services enable `
    run.googleapis.com cloudbuild.googleapis.com `
    artifactregistry.googleapis.com storage.googleapis.com

gcloud artifacts repositories create tracker `
    --repository-format=docker --location=us-central1

gcloud storage buckets create gs://$BUCKET --location=us-central1 --uniform-bucket-level-access

# Cloud Run runtime service account needs to read/write the GCS bucket.
$RUNSA = "$(gcloud projects describe $PROJECT --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding gs://$BUCKET `
    --member="serviceAccount:$RUNSA" --role="roles/storage.objectAdmin"
# Needed so the runtime SA can sign v4 URLs against itself.
gcloud iam service-accounts add-iam-policy-binding $RUNSA `
    --member="serviceAccount:$RUNSA" --role="roles/iam.serviceAccountTokenCreator"
```

### Build + deploy

```powershell
cd D:\intership\tracker_app
.\deploy\deploy.ps1 -ProjectId $PROJECT -Bucket $BUCKET
```

This runs `deploy/stage.ps1` (copies `Zone-div(final)`, `Zone-div RED`, and
the two `.onnx` weights into `./vendor`), submits a Cloud Build, then deploys:

- `--gpu 1 --gpu-type nvidia-l4`, execution-environment gen2
- `--cpu 8 --memory 32Gi --no-cpu-throttling`
- `--concurrency 1 --max-instances 1` — the in-process `JobStore` and
  `exec_lock` cannot be sharded; one L4 at a time.
- `--min-instances 0` (scale to zero). First request after idle: ~30–60s cold
  start while CUDA + ONNX model warm up. Use `-MinInstances 1` to keep one
  warm (~$510/mo).
- `--no-allow-unauthenticated` — IAM-protected. Grant yourself access:

```powershell
gcloud run services add-iam-policy-binding cricket-tracker `
    --project=$PROJECT --region=us-central1 `
    --member="user:you@example.com" --role="roles/run.invoker"
```

### Accessing the URL

The browser can't attach an identity token automatically — for a real user
flow you need either an [IAP/Cloud-Run-invoker proxy](https://cloud.google.com/run/docs/securing/identity-token-proxying)
or to relax auth + add your own. For a sanity check:

```powershell
$URL = gcloud run services describe cricket-tracker `
    --project=$PROJECT --region=us-central1 --format="value(status.url)"
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" "$URL/api/track/health"
```

### What gets baked vs externalized

| Thing | Where | Why |
|---|---|---|
| Backend code, frontend dist | Image | Atomic deploys |
| Legacy `Zone-div(final)` + `Zone-div RED` (~400 KB each) | Image (`/app/legacy/...`) | Small + version-locked with code |
| ONNX weights white (28 MB) + red (83 MB) | Image (`/app/weights/...`) | Avoid GCS round-trip at cold start |
| Job artifacts (videos, results.json, CSVs) | GCS (`gs://$BUCKET/jobs/<id>/`) | Survive instance recycling, signed-URL delivery |
| Job state (status, progress, calibration) | **In-memory only** | ⚠️ See gap #3 below — lost on container recycle |

### Cost notes

- L4 on Cloud Run: ~$0.71/hr active. Scale-to-zero means you only pay during
  jobs (+ a few seconds of warm tail). A 30 s clip processes in ≈30–60 s wall
  time, so ≈$0.01–0.02 per job.
- Cloud Build: minutes are free up to 120/day, then $0.003/min. The full
  image build (CUDA + Python deps + frontend) runs ≈10 min.
- Artifact Registry storage: ~5 GB image × $0.10/GB-month ≈ $0.50/mo.

## Known gaps to verify before production

1. **Multipart upload size**: FastAPI streams to memory unless you wire
   `python-multipart`'s spool threshold. The 500 MB guard is enforced after
   the bytes hit disk. For very large uploads, switch to chunked uploads or
   pre-signed PUTs to GCS.
2. **`ImpactDetector` re-enable**: when the stub is replaced, populate
   `events.impact` in `process.py` from `impact_detector.impact`. The UI card
   is already wired for it.
3. **In-memory job store**: jobs are lost on server restart. Swap `JobStore`
   for a Redis-backed equivalent before relying on it.
4. **Auth**: not implemented. The spec assumes the existing app's auth
   protects these routes. The `/api/track/files/{job_id}/...` local-storage
   endpoint is open — only use it in dev or behind your auth middleware.
