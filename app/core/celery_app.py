from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.summarization",
        "app.tasks.outbox_relay",  # NEW — worker must import this to execute it
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,          # redelivers if worker dies mid-task
    worker_prefetch_multiplier=1, # don't hoard tasks ahead of slow ones
    task_reject_on_worker_lost=True,
)

# NEW — Register the periodic schedule
celery_app.conf.beat_schedule = {
    "relay-outbox-every-2-seconds": {
        "task": "app.tasks.outbox_relay.relay_outbox_batch",
        "schedule": 2.0,
    },
}