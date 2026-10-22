"""Background indexing for one public Drive folder at a time per id."""

from __future__ import annotations

import asyncio
import logging

import aiofiles

from app.config import Settings
from app.database import Database
from app.drive_client import DriveError, GoogleDriveClient
from app.face_engine import decode_image, preview_jpeg

log = logging.getLogger("facescan.index")

_DB_FIELDS = {
    "name",
    "status",
    "total_photos",
    "indexed_photos",
    "face_count",
    "failed",
    "message",
    "error",
}


class DriveFolderIndexer:
    def __init__(
        self,
        db: Database,
        engine,
        drive: GoogleDriveClient,
        settings: Settings,
    ) -> None:
        self.db = db
        self.engine = engine
        self.drive = drive
        self.settings = settings
        self.live: dict[str, dict] = {}
        self._jobs: dict[str, asyncio.Task] = {}
        self._cancel: set[str] = set()
        self._slots: asyncio.Semaphore | None = None

    def running(self, drive_id: str) -> bool:
        task = self._jobs.get(drive_id)
        return task is not None and not task.done()

    def start(self, drive_id: str) -> None:
        if self.running(drive_id):
            return
        self._cancel.discard(drive_id)
        self.live[drive_id] = {
            "status": "discovering",
            "message": "Looking through the folder…",
            "current_file": "",
            "done": 0,
            "failed": 0,
            "error": None,
        }
        self.db.update_folder(
            drive_id,
            status="discovering",
            message="Looking through the folder…",
            error=None,
            failed=0,
        )
        task = asyncio.create_task(self._run(drive_id), name=f"index-{drive_id}")
        task.add_done_callback(self._log_task)
        self._jobs[drive_id] = task

    async def stop(self, drive_id: str) -> None:
        self._cancel.add(drive_id)
        task = self._jobs.get(drive_id)
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=8)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass

    async def _run(self, drive_id: str) -> None:
        try:
            if not getattr(self.engine, "ready", False):
                await asyncio.to_thread(self.engine.load)
            if self.db.index is None or self.db.model != self.engine.name:
                self.db.attach_index(self.engine.dim, self.engine.name)
            try:
                name = await self.drive.get_folder_name(drive_id)
                self._write(drive_id, name=name)
            except Exception:
                log.warning("could not read folder name for %s", drive_id, exc_info=True)
            if drive_id in self._cancel:
                return

            listing = await self.drive.list_images(drive_id)
            files = listing.files
            if drive_id in self._cancel:
                return
            if not files:
                self._write(
                    drive_id,
                    status="completed",
                    total_photos=0,
                    done=0,
                    failed=0,
                    error=None,
                    message="This folder has no photos.",
                    current_file="",
                )
                return

            folder = self.db.get_folder(drive_id)
            if folder is None:
                return
            processed = self.db.processed_file_ids(folder["id"], self.engine.name)
            pending = [item for item in files if item.file_id not in processed]
            self._write(
                drive_id,
                status="indexing",
                total_photos=len(files),
                done=len(files) - len(pending),
                failed=0,
                error=None,
                message="Reading photos",
                current_file="",
            )

            lock = asyncio.Lock()
            for chunk in _chunks(pending, self.settings.index_concurrency * 2):
                if drive_id in self._cancel:
                    return
                await asyncio.gather(
                    *(
                        self._index_one(drive_id, folder["id"], item, len(files), lock)
                        for item in chunk
                    )
                )
            if drive_id in self._cancel:
                return

            fresh = self.db.get_folder(drive_id) or folder
            failed = int(self.live.get(drive_id, {}).get("failed") or 0)
            notes = []
            if failed:
                noun = "photo" if failed == 1 else "photos"
                notes.append(f"{failed} {noun} couldn't be read")
            if listing.truncated:
                notes.append(f"Stopped after {self.settings.max_photos:,} photos")
            self._write(
                drive_id,
                status="completed",
                message=" · ".join(notes),
                error=None,
                done=len(files),
                failed=failed,
                total_photos=len(files),
                current_file="",
            )
            log.info(
                "indexed %s: %s photos, %s faces, %s failed",
                drive_id,
                fresh["indexed_photos"],
                fresh["face_count"],
                failed,
            )
        except DriveError as exc:
            if drive_id in self._cancel:
                return
            self._fail(drive_id, str(exc))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("indexing failed for %s", drive_id)
            if drive_id not in self._cancel:
                self._fail(drive_id, "Indexing stopped because of an unexpected error.")

    async def _index_one(
        self,
        drive_id: str,
        folder_pk: int,
        item,
        total: int,
        lock: asyncio.Lock,
    ) -> None:
        if drive_id in self._cancel:
            return
        self.live.setdefault(drive_id, {})["current_file"] = item.name
        failed_inc = 0
        try:
            async with self._download_slots():
                if drive_id in self._cancel:
                    return
                data = await self.drive.download_for_index(item.file_id)
            image = await asyncio.to_thread(decode_image, data, self.settings.max_image_edge)
            preview = await asyncio.to_thread(preview_jpeg, image, self.settings.preview_edge)
            preview_path = self.settings.cache_path / f"{item.file_id}.jpg"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(preview_path, "wb") as handle:
                await handle.write(preview)
            faces = await asyncio.to_thread(self.engine.analyze, image)
            height, width = image.shape[:2]
            self.db.save_photo(
                folder_pk=folder_pk,
                file_id=item.file_id,
                name=item.name,
                width=width,
                height=height,
                model=self.engine.name,
                faces=faces,
            )
        except Exception:
            log.warning("skipped %s (%s)", item.name, item.file_id, exc_info=True)
            failed_inc = 1
        if drive_id in self._cancel:
            return
        async with lock:
            live = self.live.setdefault(drive_id, {})
            done = int(live.get("done") or 0) + 1
            failed = int(live.get("failed") or 0) + failed_inc
            self._write(
                drive_id,
                status="indexing",
                total_photos=total,
                done=done,
                failed=failed,
                message="Reading photos",
            )

    def _download_slots(self) -> asyncio.Semaphore:
        if self._slots is None:
            self._slots = asyncio.Semaphore(self.settings.index_concurrency)
        return self._slots

    def _write(self, drive_id: str, **fields) -> None:
        live = self.live.setdefault(drive_id, {})
        for key, value in fields.items():
            if value is None and key not in {"error", "message", "current_file"}:
                continue
            live[key] = value
        db_fields = {key: live[key] for key in _DB_FIELDS if key in fields}
        if db_fields:
            self.db.update_folder(drive_id, **db_fields)

    def _fail(self, drive_id: str, message: str) -> None:
        self._write(
            drive_id,
            status="error",
            error=message,
            message=message,
            current_file="",
        )

    @staticmethod
    def _log_task(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("index task crashed", exc_info=exc)


def _chunks(items: list, size: int):
    step = max(1, size)
    for index in range(0, len(items), step):
        yield items[index : index + step]
