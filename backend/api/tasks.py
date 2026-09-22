"""Cloud Tasks enqueuer — drops a note in the mailbox for the worker service.

The worker is a Cloud Run service with a fixed HTTPS URL. We enqueue an HTTP
task that POSTs to <WORKER_URL>/process with the job_id in the body. Cloud
Tasks handles retries (queue config: max_attempts=3) and delivers OIDC-auth'd
requests so the worker service can require `roles/run.invoker`.
"""
from __future__ import annotations

import os

from google.cloud import tasks_v2

PROJECT_ID = os.environ.get("GCP_PROJECT", "sportsanalytics-495612")
REGION = os.environ.get("GCP_REGION", "us-central1")
QUEUE = os.environ.get("TASKS_QUEUE", "tracker-jobs")
WORKER_URL = os.environ.get("WORKER_URL", "")
INVOKER_SA = os.environ.get(
    "WORKER_INVOKER_SA",
    f"worker-invoker@{PROJECT_ID}.iam.gserviceaccount.com",
)


def enqueue_job(job_id: str) -> str:
    """Enqueue a process task. Returns the task name."""
    if not WORKER_URL:
        raise RuntimeError("WORKER_URL env var not set")

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(PROJECT_ID, REGION, QUEUE)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{WORKER_URL.rstrip('/')}/process",
            "headers": {"Content-Type": "application/json"},
            "body": ('{"job_id":"' + job_id + '"}').encode(),
            "oidc_token": {
                "service_account_email": INVOKER_SA,
                "audience": WORKER_URL,
            },
        }
    }
    response = client.create_task(parent=parent, task=task)
    return response.name
