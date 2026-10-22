"""SQLite (WAL) storage and an exact cosine face index.

FAISS ``IndexFlatIP`` is used when installed. Inner product on L2-normalized
vectors is cosine similarity. NumPy is the fallback and is exact as well.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("facescan.db")

_FOLDER_FIELDS = {
    "name",
    "url",
    "status",
    "total_photos",
    "indexed_photos",
    "face_count",
    "failed",
    "message",
    "error",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        raise ValueError("Empty face embedding")
    return values / norm


@dataclass
class StoredFace:
    face_id: int
    photo_id: int
    folder_pk: int
    file_id: str
    name: str
    width: int
    height: int
    embedding: np.ndarray
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class FaceMeta:
    face_id: int
    photo_id: int
    file_id: str
    name: str
    width: int
    height: int
    bbox: tuple[float, float, float, float]
    confidence: float


def _load_faiss():
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    return faiss


class _Bucket:
    def __init__(self, dim: int, use_faiss: bool) -> None:
        self.dim = dim
        self.metas: list[FaceMeta] = []
        self.vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self.faiss = None
        if use_faiss:
            faiss = _load_faiss()
            if faiss is not None:
                self.faiss = faiss.IndexFlatIP(dim)

    def add(self, meta: FaceMeta, vector: np.ndarray) -> None:
        self.metas.append(meta)
        self.vectors.append(vector)
        self._matrix = None
        if self.faiss is not None:
            self.faiss.add(vector.reshape(1, -1))

    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self.vectors:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            else:
                self._matrix = np.vstack(self.vectors)
        return self._matrix

    def rebuild_faiss(self) -> None:
        self._matrix = None
        if self.faiss is None:
            return
        self.faiss.reset()
        if self.vectors:
            self.faiss.add(np.vstack(self.vectors))

    def scores(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.faiss is not None and self.faiss.ntotal > 0:
            sims, indices = self.faiss.search(query.reshape(1, -1), int(self.faiss.ntotal))
            return sims[0], indices[0]
        matrix = self.matrix()
        if matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
        return matrix @ query, np.arange(matrix.shape[0])


class FaceIndex:
    """Per-folder exact cosine index. Search keeps the best face per photo."""

    def __init__(self, dim: int, model: str, use_faiss: bool | None = None) -> None:
        self.dim = dim
        self.model = model
        self._lock = threading.Lock()
        available = _load_faiss() is not None
        self.use_faiss = available if use_faiss is None else bool(use_faiss and available)
        self._buckets: dict[int, _Bucket] = {}
        if self.use_faiss:
            faiss = _load_faiss()
            if faiss is not None:
                try:
                    faiss.omp_set_num_threads(max(1, min(4, os.cpu_count() or 2)))
                except Exception:
                    pass

    def rebuild(self, faces: list[StoredFace]) -> None:
        with self._lock:
            self._buckets.clear()
            for face in faces:
                if face.embedding.size != self.dim:
                    log.warning("skip face %s with dim %s", face.face_id, face.embedding.size)
                    continue
                self._add_unlocked(face)

    def add(self, face: StoredFace) -> None:
        with self._lock:
            self._add_unlocked(face)

    def _add_unlocked(self, face: StoredFace) -> None:
        vector = l2_normalize(face.embedding)
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Embedding size {vector.shape[0]} does not match the {self.dim}-d index"
            )
        bucket = self._buckets.get(face.folder_pk)
        if bucket is None:
            bucket = _Bucket(self.dim, self.use_faiss)
            self._buckets[face.folder_pk] = bucket
        bucket.add(
            FaceMeta(
                face_id=face.face_id,
                photo_id=face.photo_id,
                file_id=face.file_id,
                name=face.name,
                width=face.width,
                height=face.height,
                bbox=face.bbox,
                confidence=face.confidence,
            ),
            vector,
        )

    def remove_photo(self, folder_pk: int, photo_id: int) -> None:
        with self._lock:
            bucket = self._buckets.get(folder_pk)
            if bucket is None:
                return
            kept = [
                (meta, vector)
                for meta, vector in zip(bucket.metas, bucket.vectors)
                if meta.photo_id != photo_id
            ]
            bucket.metas = [meta for meta, _ in kept]
            bucket.vectors = [vector for _, vector in kept]
            bucket.rebuild_faiss()
            if not bucket.metas:
                self._buckets.pop(folder_pk, None)

    def drop_folder(self, folder_pk: int) -> None:
        with self._lock:
            self._buckets.pop(folder_pk, None)

    def count(self) -> int:
        with self._lock:
            return sum(len(bucket.metas) for bucket in self._buckets.values())

    def count_folder(self, folder_pk: int) -> int:
        with self._lock:
            bucket = self._buckets.get(folder_pk)
            return 0 if bucket is None else len(bucket.metas)

    def search(
        self,
        embedding: np.ndarray,
        threshold: float,
        folder_pk: int,
        limit: int,
    ) -> list[dict]:
        query = l2_normalize(embedding)
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Embedding size {query.shape[0]} does not match the {self.dim}-d index"
            )
        with self._lock:
            bucket = self._buckets.get(folder_pk)
            if bucket is None or not bucket.metas:
                return []
            scores, indices = bucket.scores(query)
            best: dict[int, tuple[float, int]] = {}
            for score, index in zip(scores, indices):
                score_f = float(score)
                index_i = int(index)
                if index_i < 0 or index_i >= len(bucket.metas) or score_f < threshold:
                    continue
                meta = bucket.metas[index_i]
                previous = best.get(meta.photo_id)
                if previous is None or score_f > previous[0]:
                    best[meta.photo_id] = (score_f, index_i)
            ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)[:limit]
            matches = []
            for score_f, index_i in ranked:
                meta = bucket.metas[index_i]
                x, y, w, h = meta.bbox
                matches.append(
                    {
                        "file_id": meta.file_id,
                        "name": meta.name,
                        "photo_id": meta.photo_id,
                        "width": meta.width,
                        "height": meta.height,
                        "similarity": score_f,
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "confidence": meta.confidence,
                    }
                )
            return matches


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.index: FaceIndex | None = None
        self.model = ""
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA temp_store=MEMORY")

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY,
                drive_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                total_photos INTEGER NOT NULL DEFAULT 0,
                indexed_photos INTEGER NOT NULL DEFAULT 0,
                face_count INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY,
                folder_pk INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                file_id TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT '',
                indexed_at TEXT NOT NULL,
                UNIQUE(folder_pk, file_id)
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                folder_pk INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                w REAL NOT NULL,
                h REAL NOT NULL,
                confidence REAL NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_photos_folder ON photos(folder_pk);
            CREATE INDEX IF NOT EXISTS idx_faces_folder_model ON faces(folder_pk, model);
            CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
            """
        )
        self._conn.commit()

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        self._conn.close()

    def attach_index(self, dim: int, model: str, use_faiss: bool | None = None) -> None:
        self.model = model
        self.index = FaceIndex(dim, model, use_faiss=use_faiss)
        self.index.rebuild(self._all_faces(model, dim))

    def _folder_dict(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}

    def upsert_folder(self, drive_id: str, url: str, name: str = "") -> dict:
        now = utcnow()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM folders WHERE drive_id=?", (drive_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO folders (drive_id, name, url, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (drive_id, name or "Photo folder", url, now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE folders
                    SET url=?, name=CASE WHEN ? != '' THEN ? ELSE name END, updated_at=?
                    WHERE drive_id=?
                    """,
                    (url or row["url"], name, name, now, drive_id),
                )
            saved = conn.execute(
                "SELECT * FROM folders WHERE drive_id=?", (drive_id,)
            ).fetchone()
        folder = self._folder_dict(saved)
        assert folder is not None
        return folder

    def update_folder(self, drive_id: str, **fields) -> dict | None:
        assignments = []
        params: list = []
        for key, value in fields.items():
            if key not in _FOLDER_FIELDS:
                continue
            assignments.append(f"{key}=?")
            params.append(value)
        if not assignments:
            return self.get_folder(drive_id)
        assignments.append("updated_at=?")
        params.append(utcnow())
        params.append(drive_id)
        with self._tx() as conn:
            conn.execute(
                f"UPDATE folders SET {', '.join(assignments)} WHERE drive_id=?",
                params,
            )
            row = conn.execute(
                "SELECT * FROM folders WHERE drive_id=?", (drive_id,)
            ).fetchone()
        return self._folder_dict(row)

    def get_folder(self, drive_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM folders WHERE drive_id=?", (drive_id,)
            ).fetchone()
        return self._folder_dict(row)

    def list_folders(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM folders ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [self._folder_dict(row) for row in rows if row is not None]

    def delete_folder(self, drive_id: str) -> list[str] | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT id FROM folders WHERE drive_id=?", (drive_id,)
            ).fetchone()
            if row is None:
                return None
            folder_pk = int(row["id"])
            files = [
                item["file_id"]
                for item in conn.execute(
                    "SELECT file_id FROM photos WHERE folder_pk=?", (folder_pk,)
                )
            ]
            conn.execute("DELETE FROM faces WHERE folder_pk=?", (folder_pk,))
            conn.execute("DELETE FROM photos WHERE folder_pk=?", (folder_pk,))
            conn.execute("DELETE FROM folders WHERE id=?", (folder_pk,))
        if self.index is not None:
            self.index.drop_folder(folder_pk)
        return files

    def existing_file_ids(self, file_ids: list[str]) -> set[str]:
        if not file_ids:
            return set()
        found: set[str] = set()
        with self._lock:
            for offset in range(0, len(file_ids), 400):
                chunk = file_ids[offset : offset + 400]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT file_id FROM photos WHERE file_id IN ({marks})",
                    chunk,
                ).fetchall()
                found.update(row["file_id"] for row in rows)
        return found

    def processed_file_ids(self, folder_pk: int, model: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id FROM photos WHERE folder_pk=? AND model=?",
                (folder_pk, model),
            ).fetchall()
        return {row["file_id"] for row in rows}

    def save_photo(
        self,
        *,
        folder_pk: int,
        file_id: str,
        name: str,
        width: int,
        height: int,
        model: str,
        faces: list,
    ) -> int:
        now = utcnow()
        stored: list[StoredFace] = []
        with self._tx() as conn:
            row = conn.execute(
                "SELECT id FROM photos WHERE folder_pk=? AND file_id=?",
                (folder_pk, file_id),
            ).fetchone()
            replaced = row is not None
            if replaced:
                photo_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE photos
                    SET name=?, width=?, height=?, model=?, indexed_at=?
                    WHERE id=?
                    """,
                    (name, width, height, model, now, photo_id),
                )
                conn.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO photos
                        (folder_pk, file_id, name, width, height, model, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (folder_pk, file_id, name, width, height, model, now),
                )
                photo_id = int(cursor.lastrowid)
            for face in faces:
                embedding = np.ascontiguousarray(face.embedding, dtype=np.float32).reshape(-1)
                bbox = tuple(float(value) for value in face.bbox)
                cursor = conn.execute(
                    """
                    INSERT INTO faces
                        (photo_id, folder_pk, model, embedding, x, y, w, h, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        photo_id,
                        folder_pk,
                        model,
                        embedding.tobytes(),
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        float(face.confidence),
                    ),
                )
                stored.append(
                    StoredFace(
                        face_id=int(cursor.lastrowid),
                        photo_id=photo_id,
                        folder_pk=folder_pk,
                        file_id=file_id,
                        name=name,
                        width=width,
                        height=height,
                        embedding=embedding,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        confidence=float(face.confidence),
                    )
                )
            indexed = conn.execute(
                "SELECT COUNT(*) AS n FROM photos WHERE folder_pk=?",
                (folder_pk,),
            ).fetchone()["n"]
            face_count = conn.execute(
                "SELECT COUNT(*) AS n FROM faces WHERE folder_pk=? AND model=?",
                (folder_pk, model),
            ).fetchone()["n"]
            conn.execute(
                """
                UPDATE folders
                SET indexed_photos=?, face_count=?, updated_at=?
                WHERE id=?
                """,
                (indexed, face_count, now, folder_pk),
            )
        if self.index is not None:
            if replaced:
                self.index.remove_photo(folder_pk, photo_id)
            for item in stored:
                self.index.add(item)
        return photo_id

    def known_files(self, file_ids: list[str]) -> list[dict]:
        if not file_ids:
            return []
        found: dict[str, dict] = {}
        with self._lock:
            for offset in range(0, len(file_ids), 400):
                chunk = file_ids[offset : offset + 400]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT file_id, name FROM photos WHERE file_id IN ({marks})",
                    chunk,
                ).fetchall()
                for row in rows:
                    found[row["file_id"]] = {
                        "file_id": row["file_id"],
                        "name": row["name"],
                    }
        return [found[file_id] for file_id in file_ids if file_id in found]

    def count_faces(self, model: str | None = None) -> int:
        if self.index is not None and (model is None or model == self.model):
            return self.index.count()
        with self._lock:
            if model:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM faces WHERE model=?", (model,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()
        return int(row["n"])

    def search(
        self,
        embedding: np.ndarray,
        threshold: float,
        folder_pk: int,
        limit: int,
    ) -> list[dict]:
        if self.index is None:
            return []
        return self.index.search(embedding, threshold, folder_pk, limit)

    def _all_faces(self, model: str, dim: int) -> list[StoredFace]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT f.id, f.photo_id, f.folder_pk, f.embedding, f.x, f.y, f.w, f.h,
                       f.confidence, p.file_id, p.name, p.width, p.height
                FROM faces f
                JOIN photos p ON p.id = f.photo_id
                WHERE f.model=?
                """,
                (model,),
            ).fetchall()
        faces: list[StoredFace] = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if embedding.size != dim:
                log.warning("skip stored face %s with dim %s", row["id"], embedding.size)
                continue
            faces.append(
                StoredFace(
                    face_id=int(row["id"]),
                    photo_id=int(row["photo_id"]),
                    folder_pk=int(row["folder_pk"]),
                    file_id=row["file_id"],
                    name=row["name"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                    embedding=embedding,
                    bbox=(float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])),
                    confidence=float(row["confidence"]),
                )
            )
        return faces
