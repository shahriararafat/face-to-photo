"""Core tests that do not download face models or call Google."""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from app.config import Settings
from app.database import Database, FaceIndex, StoredFace
from app.drive_client import (
    DriveError,
    DriveFile,
    FolderLinkError,
    Listing,
    listing_kind,
    parse_drive_html,
    parse_folder_id,
)
from app.face_engine import DetectedFace, FaceEngine, decode_image
from app.indexer import DriveFolderIndexer
from app.main import create_app, safe_zip_name


FOLDER = "1AbC-def_1234567890"
PNG_ID = "photoFileId0001"
BAD_ID = "photoFileId0002"


def _vec(first: float = 1.0, dim: int = 8) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[0] = float(first)
    vector[1] = 1.0
    return vector / np.linalg.norm(vector)


def _face(photo_id: int, file_id: str, value: float, face_id: int, folder_pk: int = 1) -> StoredFace:
    return StoredFace(
        face_id=face_id,
        photo_id=photo_id,
        folder_pk=folder_pk,
        file_id=file_id,
        name=f"{file_id}.jpg",
        width=80,
        height=60,
        embedding=_vec(value),
        bbox=(0.1, 0.2, 0.3, 0.4),
        confidence=0.9,
    )


def _png(width: int = 64, height: int = 48, color=(220, 180, 160)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    name = "fake"
    label = "Test engine"
    dim = 8
    ready = True
    accurate = False
    default_threshold = 0.5
    threshold_min = 0.2
    threshold_max = 0.9

    def analyze(self, _bgr):
        return [DetectedFace(_vec(), (0.2, 0.25, 0.3, 0.4), 0.99)]


class FakeDrive:
    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs

    async def list_images(self, _folder_id: str) -> Listing:
        return Listing(
            files=[DriveFile(file_id, f"{file_id}.jpg") for file_id in self.blobs],
            truncated=False,
        )

    async def get_folder_name(self, _folder_id: str) -> str:
        return "Family weekend"

    async def download_for_index(self, file_id: str) -> bytes:
        if file_id not in self.blobs:
            raise DriveError("missing")
        return self.blobs[file_id]

    async def download_original(self, file_id: str) -> bytes:
        return self.blobs[file_id]


def test_parse_folder_id_accepts_common_links():
    assert parse_folder_id(f"https://drive.google.com/drive/folders/{FOLDER}?usp=sharing") == FOLDER
    assert parse_folder_id(f"https://drive.google.com/drive/u/1/folders/{FOLDER}") == FOLDER
    assert parse_folder_id(f"https://drive.google.com/folderview?id={FOLDER}") == FOLDER
    assert parse_folder_id(FOLDER) == FOLDER


def test_parse_folder_id_rejects_file_links_and_junk():
    with pytest.raises(FolderLinkError, match="single file"):
        parse_folder_id("https://drive.google.com/file/d/1AbC-def_1234567890/view")
    with pytest.raises(FolderLinkError):
        parse_folder_id("https://example.com/not-drive")
    with pytest.raises(FolderLinkError):
        parse_folder_id("")


def test_parse_embedded_listing_reads_images_and_subfolders():
    html = """
    <html><body>
      <div class="flip-entries">
        <div class="flip-entry" id="entry-photoFileId0001">
          <a href="https://drive.google.com/file/d/photoFileId0001/view">Day.jpg</a>
        </div>
        <div class="flip-entry" id="entry-folderId00000002">
          <a href="https://drive.google.com/drive/folders/folderId00000002">Day 2</a>
        </div>
        <a href="https://drive.google.com/file/d/notesFileId00003/view">notes.pdf</a>
      </div>
    </body></html>
    """
    images, folders = parse_drive_html(html, FOLDER)
    assert listing_kind(html) == "embedded"
    assert [image.file_id for image in images] == ["photoFileId0001"]
    assert images[0].name == "Day.jpg"
    assert folders == ["folderId00000002"]


def test_face_index_dedupes_photos_and_respects_folder():
    index = FaceIndex(dim=8, model="fake", use_faiss=False)
    strong = _face(1, "aaaFileId000001", 1.0, 1)
    weak = _face(1, "aaaFileId000001", 0.2, 2)
    other = _face(2, "bbbFileId000002", 1.0, 3)
    elsewhere = _face(3, "cccFileId000003", 1.0, 4, folder_pk=9)
    index.rebuild([strong, weak, other, elsewhere])
    matches = index.search(_vec(1.0), threshold=0.5, folder_pk=1, limit=10)
    assert [match["file_id"] for match in matches] == ["aaaFileId000001", "bbbFileId000002"]
    assert matches[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert index.search(_vec(1.0), threshold=0.5, folder_pk=9, limit=10)[0]["file_id"] == "cccFileId000003"
    assert index.search(_vec(-1.0), threshold=0.8, folder_pk=1, limit=10) == []


def test_faiss_matches_numpy_when_installed():
    pytest.importorskip("faiss")
    faces = [_face(index, f"fileId{index:012d}", index / 7, index) for index in range(1, 8)]
    numpy_index = FaceIndex(8, "fake", use_faiss=False)
    faiss_index = FaceIndex(8, "fake", use_faiss=True)
    assert faiss_index.use_faiss is True
    numpy_index.rebuild(faces)
    faiss_index.rebuild(faces)
    query = _vec(1.0)
    numpy_matches = numpy_index.search(query, 0.3, 1, 10)
    faiss_matches = faiss_index.search(query, 0.3, 1, 10)
    assert [match["file_id"] for match in numpy_matches] == [match["file_id"] for match in faiss_matches]
    for left, right in zip(numpy_matches, faiss_matches):
        assert left["similarity"] == pytest.approx(right["similarity"], abs=1e-5)


def test_database_roundtrip_and_cascade(tmp_path):
    db = Database(tmp_path / "t.db")
    db.attach_index(8, "fake", use_faiss=False)
    folder = db.upsert_folder(FOLDER, url="https://drive.google.com/drive/folders/" + FOLDER, name="Album")
    face = DetectedFace(_vec(), (0.1, 0.2, 0.3, 0.4), 0.8)
    db.save_photo(
        folder_pk=folder["id"],
        file_id=PNG_ID,
        name="one.jpg",
        width=64,
        height=48,
        model="fake",
        faces=[face],
    )
    assert PNG_ID in db.processed_file_ids(folder["id"], "fake")
    matches = db.search(_vec(), 0.4, folder["id"], 5)
    assert matches[0]["name"] == "one.jpg"
    removed = db.delete_folder(FOLDER)
    assert removed == [PNG_ID]
    assert db.get_folder(FOLDER) is None
    assert db.count_faces("fake") == 0
    assert db.search(_vec(), 0.1, folder["id"], 5) == []
    db.close()


def test_decode_image_fixes_exif_orientation():
    image = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    raw = Image.open(io.BytesIO(buffer.getvalue()))
    assert raw.size == (40, 20)
    turned = ImageOps.exif_transpose(raw)
    bgr = decode_image(buffer.getvalue(), max_edge=1600)
    assert bgr.shape[0] == turned.height
    assert bgr.shape[1] == turned.width
    assert bgr.shape[2] == 3


def test_haar_engine_loads_without_models(tmp_path):
    engine = FaceEngine(tmp_path, prefer="haar")
    engine.load()
    assert engine.name == "haar"
    assert engine.dim == 256
    blank = np.zeros((80, 80, 3), dtype=np.uint8)
    assert engine.analyze(blank) == []


def test_safe_zip_name_dedupes_and_adds_extension():
    used: set[str] = set()
    png = _png()
    first = safe_zip_name("My Photo!!.png", PNG_ID, png, used)
    second = safe_zip_name("My Photo!!.png", BAD_ID, png, used)
    bare = safe_zip_name("portrait", "fileIdBare00001", png, used)
    assert first == "My Photo_.png"
    assert second.startswith("My Photo_-")
    assert bare.endswith(".png")


def test_indexer_skips_failures_and_is_searchable(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        google_drive_api_key="",
        index_concurrency=2,
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    engine = FakeEngine()
    db.attach_index(engine.dim, engine.name, use_faiss=False)
    db.upsert_folder(FOLDER, url="https://drive.google.com/folders/" + FOLDER)
    drive = FakeDrive({PNG_ID: _png(), BAD_ID: _png()})
    calls = {"n": 0}

    async def download(file_id: str) -> bytes:
        calls["n"] += 1
        if file_id == BAD_ID:
            raise DriveError("broken")
        return _png()

    drive.download_for_index = download  # type: ignore[method-assign]

    async def run():
        indexer = DriveFolderIndexer(db, engine, drive, settings)
        indexer.start(FOLDER)
        await indexer._jobs[FOLDER]
        first_pass = calls["n"]
        indexer.start(FOLDER)
        await indexer._jobs[FOLDER]
        return indexer, first_pass

    indexer, first_pass = asyncio.run(run())
    folder = db.get_folder(FOLDER)
    assert folder["status"] == "completed"
    assert folder["indexed_photos"] == 1
    assert folder["face_count"] == 1
    assert folder["name"] == "Family weekend"
    assert indexer.live[FOLDER]["failed"] == 1
    assert first_pass == 2
    assert calls["n"] == 3  # the broken photo is retried; the good one is not
    assert (settings.cache_path / f"{PNG_ID}.jpg").is_file()
    assert len(db.search(_vec(), 0.5, folder["id"], 10)) == 1
    db.close()


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        google_drive_api_key="",
    )
    app = create_app(settings, engine=FakeEngine())
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_detect_search_and_zip(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["engine"] == "fake"
    assert body["drive_mode"] == "public"
    assert body["accurate"] is False

    rejected = client.post("/api/folders", json={"url": "https://drive.google.com/file/d/abc/view"})
    assert rejected.status_code == 400

    page = client.get("/")
    assert page.status_code == 200
    assert "Find one face" in page.text

    detected = client.post(
        "/api/detect",
        files={"file": ("face.png", _png(), "image/png")},
    )
    assert detected.status_code == 200, detected.text
    payload = detected.json()
    assert len(payload["faces"]) == 1
    crop = client.get(payload["faces"][0]["crop_url"])
    assert crop.status_code == 200
    assert crop.headers["content-type"].startswith("image/")

    missing = client.post("/api/detect", files={"file": ("x.txt", b"hello", "text/plain")})
    assert missing.status_code == 400

    app = client.app
    folder = app.state.db.upsert_folder(FOLDER, url="https://drive.google.com/drive/folders/" + FOLDER)
    app.state.db.save_photo(
        folder_pk=folder["id"],
        file_id=PNG_ID,
        name="match.jpg",
        width=64,
        height=48,
        model="fake",
        faces=[DetectedFace(_vec(), (0.2, 0.25, 0.3, 0.4), 0.99)],
    )
    # The in-memory index was built before this insert, and save_photo updates it.
    found = client.post(
        "/api/search",
        json={
            "token": payload["token"],
            "face_index": 0,
            "folder_id": FOLDER,
            "threshold": 0.2,
            "limit": 10,
        },
    )
    assert found.status_code == 200, found.text
    assert found.json()["count"] == 1
    assert found.json()["matches"][0]["similarity"] == pytest.approx(1.0, abs=1e-5)

    class StubDrive:
        async def download_original(self, file_id: str) -> bytes:
            assert file_id == PNG_ID
            return _png()

    app.state.drive = StubDrive()
    zipped = client.post("/api/download", json={"file_ids": [PNG_ID, "not-a-real"]})
    assert zipped.status_code == 200
    assert zipped.headers["content-type"] == "application/zip"
    assert zipped.content[:2] == b"PK"

    blocked = client.post("/api/download", json={"file_ids": ["someOtherFile999"]})
    assert blocked.status_code == 404

    removed = client.delete(f"/api/folders/{FOLDER}")
    assert removed.status_code == 200
    assert client.get(f"/api/folders/{FOLDER}").status_code == 404
