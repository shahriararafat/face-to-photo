"""Environment-backed settings. The app runs with an empty .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

_ENGINES = {"auto", "insightface", "sface", "haar"}


class Settings(BaseSettings):
    google_drive_api_key: str = ""
    face_engine: str = "auto"
    data_dir: str = "data"
    model_dir: str = "models"
    max_image_edge: int = 1600
    preview_edge: int = 720
    index_concurrency: int = 3
    max_photos: int = 10000

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    @model_validator(mode="after")
    def _normalize(self) -> "Settings":
        self.google_drive_api_key = (self.google_drive_api_key or "").strip()
        engine = (self.face_engine or "auto").strip().lower()
        self.face_engine = engine if engine in _ENGINES else "auto"
        self.index_concurrency = max(1, min(int(self.index_concurrency), 8))
        self.max_photos = max(1, min(int(self.max_photos), 50000))
        self.max_image_edge = max(640, min(int(self.max_image_edge), 3200))
        self.preview_edge = max(320, min(int(self.preview_edge), self.max_image_edge))
        return self

    def _abs(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        return path

    @property
    def data_path(self) -> Path:
        return self._abs(self.data_dir)

    @property
    def model_path(self) -> Path:
        return self._abs(self.model_dir)

    @property
    def cache_path(self) -> Path:
        return self.data_path / "cache"

    @property
    def ref_path(self) -> Path:
        return self.cache_path / "refs"

    @property
    def db_path(self) -> Path:
        return self.data_path / "facescan.db"

    def ensure_dirs(self) -> None:
        for path in (self.data_path, self.cache_path, self.ref_path, self.model_path):
            path.mkdir(parents=True, exist_ok=True)
