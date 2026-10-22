"""FaceScan Pro HTTP API and the static search page."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, Settings
from app.database import Database
from app.drive_client import (
    FolderLinkError,
    GoogleDriveClient,
    parse_folder_id,
    sniff_ext,
)
from app.face_engine import crop_jpeg, decode_image, preview_jpeg
from app.face_engine import FaceEngine
from app.indexer import DriveFolderIndexer

log = logging.getLogger("facescan.api")

STATIC = ROOT / "static"
ID_RE = re.compile(r"^[a-zA-Z0-9_-]{10,}$")
TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")
REF_TTL_SECONDS = 6 * 60 * 60
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_ZIP_FILES = 400


@dataclass
class RefSession:
    embeddings: list[np.ndarray]
    created: float


class RefStore:
    def __init__(self, limit: int = 80) -> None:
        self._items: OrderedDict[str, RefSession] = OrderedDict()
        self._lock = threading.Lock()
        self._limit = limit

    def put(self, embeddings: list[np.ndarray]) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._items[token] = RefSession(embeddings, time.time())
            self._expire_unlocked()
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
        return token

    def get(self, token: str) -> RefSession | None:
        with self._lock:
            item = self._items.get(token)
            if item is None:
                return None
            if time.time() - item.created > REF_TTL_SECONDS:
                self._items.pop(token, None)
                return None
            self._items.move_to_end(token)
            return item

    def _expire_unlocked(self) -> None:
        cutoff = time.time() - REF_TTL_SECONDS
        stale = [token for token, item in self._items.items() if item.created < cutoff]
        for token in stale:
            self._items.pop(token, None)


class FolderRequest(BaseModel):
    url: str = Field(min_length=8, max_length=500)


class SearchRequest(BaseModel):
    token: str
    face_index: int = 0
    folder_id: str
    threshold: float = 0.3
    limit: int = 400


class DownloadRequest(BaseModel):
    file_ids: list[str] = Field(default_factory=list)


def create_app(settings: Settings | None = None, engine=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        app.state.settings.ensure_dirs()
        db = Database(app.state.settings.db_path)
        face_engine = app.state.preload_engine
        if face_engine is None:
            face_engine = FaceEngine(app.state.settings.model_path, app.state.settings.face_engine)
            log.info("loading face engine (%s)", app.state.settings.face_engine)
            face_engine.load()
        db.attach_index(face_engine.dim, face_engine.name)
        drive = GoogleDriveClient(
            api_key=app.state.settings.google_drive_api_key,
            max_photos=app.state.settings.max_photos,
        )
        app.state.db = db
        app.state.engine = face_engine
        app.state.drive = drive
        app.state.indexer = DriveFolderIndexer(db, face_engine, drive, app.state.settings)
        app.state.refs = RefStore()
        log.info(
            "ready: engine=%s dim=%s drive=%s",
            face_engine.name,
            face_engine.dim,
            "api" if app.state.settings.google_drive_api_key else "public",
        )
        try:
            yield
        finally:
            await drive.aclose()
            db.close()

    app = FastAPI(title="FaceScan Pro", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.preload_engine = engine

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    def health():
        face_engine = app.state.engine
        db: Database = app.state.db
        return {
            "status": "ok" if face_engine.ready else "loading",
            "engine": face_engine.name,
            "engine_label": face_engine.label,
            "dim": face_engine.dim,
            "accurate": bool(face_engine.accurate),
            "drive_mode": "api" if app.state.settings.google_drive_api_key else "public",
            "default_threshold": face_engine.default_threshold,
            "threshold_min": face_engine.threshold_min,
            "threshold_max": face_engine.threshold_max,
            "faces_indexed": db.count_faces(face_engine.name),
        }

    @app.get("/api/folders")
    def list_folders():
        rows = app.state.db.list_folders()
        return {"folders": [folder_view(app, row["drive_id"]) for row in rows]}

    @app.post("/api/folders")
    async def create_folder(body: FolderRequest):
        try:
            drive_id = parse_folder_id(body.url)
        except FolderLinkError as exc:
            raise HTTPException(400, str(exc)) from exc
        app.state.db.upsert_folder(drive_id, url=body.url.strip())
        app.state.indexer.start(drive_id)
        return folder_view(app, drive_id)

    @app.get("/api/folders/{drive_id}")
    def get_folder(drive_id: str):
        _require_id(drive_id)
        payload = folder_view(app, drive_id)
        if payload is None:
            raise HTTPException(404, "Folder not found.")
        return payload

    @app.post("/api/folders/{drive_id}/sync")
    async def sync_folder(drive_id: str):
        _require_id(drive_id)
        if app.state.db.get_folder(drive_id) is None:
            raise HTTPException(404, "Folder not found.")
        app.state.indexer.start(drive_id)
        return folder_view(app, drive_id)

    @app.delete("/api/folders/{drive_id}")
    async def remove_folder(drive_id: str):
        _require_id(drive_id)
        await app.state.indexer.stop(drive_id)
        removed = app.state.db.delete_folder(drive_id)
        if removed is None:
            app.state.indexer.live.pop(drive_id, None)
            raise HTTPException(404, "Folder not found.")
        still_used = app.state.db.existing_file_ids(removed)
        cache = app.state.settings.cache_path
        for file_id in removed:
            if file_id in still_used:
                continue
            preview = cache / f"{file_id}.jpg"
            preview.unlink(missing_ok=True)
        app.state.indexer.live.pop(drive_id, None)
        return {"ok": True}

    @app.post("/api/detect")
    async def detect(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Use a photo under 12 MB.")
        face_engine = app.state.engine
        try:
            image = await asyncio.to_thread(decode_image, data, app.state.settings.max_image_edge)
            faces = await asyncio.to_thread(face_engine.analyze, image)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("detect failed")
            raise HTTPException(500, "Couldn't read faces from that photo.") from exc
        if not faces:
            raise HTTPException(
                422, "No face found. Use a clearer photo where the face is visible."
            )
        token = app.state.refs.put([face.embedding for face in faces])
        ref_dir = app.state.settings.ref_path
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / f"{token}.jpg").write_bytes(preview_jpeg(image, app.state.settings.preview_edge))
        payload_faces = []
        for index, face in enumerate(faces):
            (ref_dir / f"{token}_{index}.jpg").write_bytes(crop_jpeg(image, face.bbox))
            x, y, w, h = face.bbox
            payload_faces.append(
                {
                    "index": index,
                    "score": face.confidence,
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "crop_url": f"/api/ref/{token}/{index}",
                }
            )
        return {
            "token": token,
            "preview_url": f"/api/ref/{token}",
            "faces": payload_faces,
        }

    @app.post("/api/search")
    def search(body: SearchRequest):
        if not TOKEN_RE.match(body.token or ""):
            raise HTTPException(404, "That reference photo expired. Add it again.")
        _require_id(body.folder_id)
        session = app.state.refs.get(body.token)
        if session is None:
            raise HTTPException(404, "That reference photo expired. Add it again.")
        if body.face_index < 0 or body.face_index >= len(session.embeddings):
            raise HTTPException(400, "Pick one of the detected faces.")
        folder = app.state.db.get_folder(body.folder_id)
        if folder is None:
            raise HTTPException(404, "Index that folder before searching.")
        threshold = min(1.0, max(0.0, float(body.threshold)))
        limit = min(500, max(1, int(body.limit)))
        try:
            matches = app.state.db.search(
                session.embeddings[body.face_index],
                threshold,
                int(folder["id"]),
                limit,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "count": len(matches),
            "searched_faces": app.state.db.index.count_folder(int(folder["id"]))
            if app.state.db.index is not None
            else 0,
            "matches": matches,
        }

    @app.post("/api/download")
    async def download(body: DownloadRequest):
        file_ids: list[str] = []
        seen: set[str] = set()
        for file_id in body.file_ids:
            if file_id in seen or not ID_RE.match(file_id or ""):
                continue
            seen.add(file_id)
            file_ids.append(file_id)
            if len(file_ids) > MAX_ZIP_FILES:
                raise HTTPException(400, "Download up to 400 photos at a time.")
        rows = app.state.db.known_files(file_ids)
        if not rows:
            raise HTTPException(404, "None of those photos are in the index.")
        spool = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
        wrote = 0
        used: set[str] = set()
        with zipfile.ZipFile(spool, "w", compression=zipfile.ZIP_STORED) as archive:
            for row in rows:
                try:
                    data = await app.state.drive.download_original(row["file_id"])
                except Exception:
                    log.warning("zip skipped %s", row["file_id"], exc_info=True)
                    continue
                if not data:
                    continue
                archive.writestr(safe_zip_name(row["name"], row["file_id"], data, used), data)
                wrote += 1
        if wrote == 0:
            spool.close()
            raise HTTPException(502, "Couldn't download those photos from Google Drive.")
        spool.seek(0)

        def chunks():
            try:
                while True:
                    block = spool.read(256 * 1024)
                    if not block:
                        break
                    yield block
            finally:
                spool.close()

        return StreamingResponse(
            chunks(),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=facescan-matches.zip",
                "X-Facescan-Count": str(wrote),
            },
        )

    @app.get("/api/preview/{file_id}")
    def preview(file_id: str):
        path = _cache_file(app.state.settings.cache_path, file_id)
        if not path.is_file():
            raise HTTPException(404, "Preview not ready.")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/ref/{token}")
    def reference_preview(token: str):
        path = _ref_file(app.state.settings.ref_path, token, None)
        if app.state.refs.get(token) is None or not path.is_file():
            raise HTTPException(404, "That reference photo expired. Add it again.")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/ref/{token}/{index}")
    def reference_face(token: str, index: int):
        if index < 0 or index > 20:
            raise HTTPException(404, "Face not found.")
        path = _ref_file(app.state.settings.ref_path, token, index)
        if app.state.refs.get(token) is None or not path.is_file():
            raise HTTPException(404, "That reference photo expired. Add it again.")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def folder_view(app: FastAPI, drive_id: str) -> dict | None:
    row = app.state.db.get_folder(drive_id)
    if row is None:
        return None
    live = app.state.indexer.live.get(drive_id) or {}
    data = {
        "id": row["drive_id"],
        "name": row["name"] or "Photo folder",
        "url": row["url"],
        "status": row["status"],
        "total_photos": row["total_photos"],
        "indexed_photos": row["indexed_photos"],
        "face_count": row["face_count"],
        "failed": row["failed"],
        "message": row["message"] or "",
        "error": row["error"],
        "done": row["indexed_photos"],
        "current_file": "",
        "updated_at": row["updated_at"],
    }
    for key in ("status", "name", "total_photos", "failed", "message", "done", "current_file"):
        if key in live and live[key] is not None:
            data[key] = live[key]
    if "error" in live:
        data["error"] = live["error"]
    return data


def safe_zip_name(name: str, file_id: str, data: bytes, used: set[str]) -> str:
    base = Path(name or "").name
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" ._")
    base = base[:120]
    if not base:
        base = file_id + sniff_ext(data)
    if Path(base).suffix == "":
        base += sniff_ext(data)
    stem = Path(base).stem or file_id
    ext = Path(base).suffix or sniff_ext(data)
    candidate = f"{stem}{ext}"
    number = 2
    while candidate.lower() in used:
        candidate = f"{stem}-{number}{ext}"
        number += 1
    used.add(candidate.lower())
    return candidate


def _require_id(value: str) -> str:
    if not ID_RE.match(value or ""):
        raise HTTPException(400, "Unknown folder.")
    return value


def _cache_file(cache: Path, file_id: str) -> Path:
    if not ID_RE.match(file_id or ""):
        raise HTTPException(404, "Preview not ready.")
    path = (cache / f"{file_id}.jpg").resolve()
    if path.parent != cache.resolve():
        raise HTTPException(404, "Preview not ready.")
    return path


def _ref_file(ref_dir: Path, token: str, index: int | None) -> Path:
    if not TOKEN_RE.match(token or ""):
        raise HTTPException(404, "That reference photo expired. Add it again.")
    name = f"{token}.jpg" if index is None else f"{token}_{index}.jpg"
    path = (ref_dir / name).resolve()
    if path.parent != ref_dir.resolve():
        raise HTTPException(404, "That reference photo expired. Add it again.")
    return path


app = create_app()
