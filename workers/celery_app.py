import sys
import os

# Ensure the project root is on sys.path so worker subprocesses can import
# project modules (data, ml, configs, etc.) regardless of where Celery was launched.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from celery import Celery
from configs.config import get_settings

settings = get_settings()

celery_app = Celery(
    "tilt_radar",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
)
