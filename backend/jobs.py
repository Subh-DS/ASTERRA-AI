"""Bounded background job manager for CPU/GPU pipeline work."""

from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .pipeline import run_pipeline
from .storage import JobStore


class JobManager:
    def __init__(self, store=None):
        self.store = store or JobStore(settings.jobs_root)
        self.executor = ThreadPoolExecutor(max_workers=settings.worker_count, thread_name_prefix="asterra-pipeline")
        self._futures = {}

    def startup(self):
        self.store.recover_incomplete()

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=False)

    def queue_depth(self):
        return sum(1 for future in self._futures.values() if not future.done())

    def create(self, original_name, content_type, gcps, dem_provider="auto", dem_path=None, submit=False):
        if self.queue_depth() >= settings.max_queue_size:
            raise RuntimeError("Processing queue is full; try again shortly.")
        job = self.store.create(original_name, content_type, gcps, dem_provider, dem_path)
        if submit:
            self._submit(job["job_id"])
        return job

    def start(self, job_id):
        if self.queue_depth() >= settings.max_queue_size:
            raise RuntimeError("Processing queue is full; try again shortly.")
        self._submit(job_id)

    def _submit(self, job_id):
        future = self.executor.submit(run_pipeline, job_id, self.store, settings)
        self._futures[job_id] = future

        def done(completed):
            self._futures.pop(job_id, None)
            exception = completed.exception()
            if exception:
                try:
                    stage = self.store.get(job_id).get("stage")
                except KeyError:
                    stage = None
                self.store.fail(job_id, str(exception), stage)

        future.add_done_callback(done)

    def retry(self, job_id):
        self.store.reset_for_retry(job_id)
        if self.queue_depth() >= settings.max_queue_size:
            self.store.fail(job_id, "Processing queue is full; try again shortly.")
            raise RuntimeError("Processing queue is full; try again shortly.")
        self._submit(job_id)


manager = JobManager()
