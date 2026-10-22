"""CPU face detection and embeddings, with a three-tier fallback.

1. InsightFace buffalo_l (ArcFace 512-d) when the package is installed.
2. OpenCV YuNet + SFace. This is the default on a normal VPS.
3. A Haar cascade and a normalized spatial descriptor if no model can load.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger("facescan.engine")

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"
YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
SFACE_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)


@dataclass
class DetectedFace:
    embedding: np.ndarray
    bbox: tuple[float, float, float, float]
    confidence: float


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        raise ValueError("Empty face embedding")
    return values / norm


def clamp_bbox(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    x = min(1.0, max(0.0, float(x)))
    y = min(1.0, max(0.0, float(y)))
    w = min(1.0 - x, max(0.0, float(w)))
    h = min(1.0 - y, max(0.0, float(h)))
    return (x, y, w, h)


def decode_image(data: bytes, max_edge: int = 1600) -> np.ndarray:
    if not data or len(data) < 16:
        raise ValueError("That file isn't an image we can read.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ValueError("That file isn't an image we can read.") from exc
    image = ImageOps.exif_transpose(image) or image
    if getattr(image, "is_animated", False):
        image.seek(0)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    if image.width < 16 or image.height < 16:
        raise ValueError("That photo is too small to find a face.")
    edge = max(image.size)
    if edge > max_edge:
        scale = max_edge / float(edge)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    rgb = np.asarray(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preview_jpeg(bgr: np.ndarray, max_edge: int, quality: int = 82) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def crop_jpeg(bgr: np.ndarray, bbox: tuple[float, float, float, float], pad: float = 0.45) -> bytes:
    height, width = bgr.shape[:2]
    x, y, w, h = bbox
    cx = (x + w / 2.0) * width
    cy = (y + h / 2.0) * height
    side = max(w * width, h * height) * (1.0 + pad)
    side = max(side, 32.0)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right - left < 8 or bottom - top < 8:
        crop = bgr
    else:
        crop = bgr[top:bottom, left:right]
    return preview_jpeg(crop, 360, quality=86)


def _download(urls: tuple[str, ...], dest: Path, min_bytes: int) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        partial = dest.with_suffix(dest.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FaceScanPro/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
            if partial.stat().st_size < min_bytes:
                raise OSError(f"download too small ({partial.stat().st_size} bytes)")
            partial.replace(dest)
            log.info("saved model %s", dest.name)
            return
        except Exception as exc:
            last_error = exc
            log.warning("could not download %s: %s", dest.name, exc)
            partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {dest.name}") from last_error


class FaceEngine:
    def __init__(self, model_dir: Path, prefer: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        self.prefer = (prefer or "auto").lower()
        self.name = "unloaded"
        self.label = "Starting"
        self.dim = 0
        self.default_threshold = 0.5
        self.threshold_min = 0.35
        self.threshold_max = 0.75
        self.ready = False
        self._lock = threading.Lock()
        self._app = None
        self._detector = None
        self._recognizer = None
        self._cascade = None

    @property
    def accurate(self) -> bool:
        return self.name in {"insightface", "sface"}

    def load(self) -> None:
        cv2.setNumThreads(max(1, (os.cpu_count() or 2) // 2))
        prefer = self.prefer
        if prefer in {"auto", "insightface"} and self._try_insightface():
            log.info("face engine ready: %s (%sd)", self.name, self.dim)
            return
        if prefer in {"auto", "insightface", "sface"} and self._try_sface():
            log.info("face engine ready: %s (%sd)", self.name, self.dim)
            return
        self._load_haar()
        log.info("face engine ready: %s (%sd)", self.name, self.dim)

    def analyze(self, bgr: np.ndarray) -> list[DetectedFace]:
        if bgr is None or bgr.size == 0:
            return []
        image = bgr
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            return []
        with self._lock:
            if self.name == "insightface":
                faces = self._analyze_insight(image)
            elif self.name == "sface":
                faces = self._analyze_sface(image)
            else:
                faces = self._analyze_haar(image)
        kept = [face for face in faces if face.bbox[2] > 0.012 and face.bbox[3] > 0.012]
        kept.sort(key=lambda face: face.bbox[2] * face.bbox[3], reverse=True)
        return kept[:12]

    def _set_profile(
        self,
        name: str,
        label: str,
        dim: int,
        default: float,
        low: float,
        high: float,
    ) -> None:
        self.name = name
        self.label = label
        self.dim = dim
        self.default_threshold = default
        self.threshold_min = low
        self.threshold_max = high
        self.ready = True

    def _try_insightface(self) -> bool:
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            log.info("InsightFace not installed (%s)", exc)
            return False
        try:
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(self.model_dir),
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            log.warning("InsightFace failed to start", exc_info=True)
            return False
        self._app = app
        self._set_profile("insightface", "ArcFace · CPU", 512, 0.50, 0.35, 0.75)
        return True

    def _try_sface(self) -> bool:
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            log.info("This OpenCV build has no YuNet/SFace")
            return False
        try:
            _download(YUNET_URLS, self.model_dir / YUNET_NAME, 100_000)
            _download(SFACE_URLS, self.model_dir / SFACE_NAME, 1_000_000)
            detector = cv2.FaceDetectorYN.create(
                str(self.model_dir / YUNET_NAME),
                "",
                (320, 320),
                0.5,
                0.3,
                5000,
            )
            recognizer = cv2.FaceRecognizerSF.create(str(self.model_dir / SFACE_NAME), "")
        except Exception:
            log.warning("SFace failed to load", exc_info=True)
            return False
        self._detector = detector
        self._recognizer = recognizer
        # OpenCV's reference cosine threshold for SFace is 0.363.
        self._set_profile("sface", "SFace · CPU", 128, 0.40, 0.30, 0.80)
        return True

    def _load_haar(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Haar cascade missing at {cascade_path}")
        self._cascade = cascade
        self._set_profile("haar", "Basic fallback", 256, 0.88, 0.70, 0.98)

    def _analyze_insight(self, bgr: np.ndarray) -> list[DetectedFace]:
        height, width = bgr.shape[:2]
        found = []
        for face in self._app.get(bgr):
            embedding = l2_normalize(face.normed_embedding)
            x1, y1, x2, y2 = (float(value) for value in face.bbox)
            bbox = clamp_bbox(x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height)
            found.append(DetectedFace(embedding, bbox, float(getattr(face, "det_score", 1.0))))
        return found

    def _analyze_sface(self, bgr: np.ndarray) -> list[DetectedFace]:
        height, width = bgr.shape[:2]
        self._detector.setInputSize((width, height))
        detected = self._detector.detect(bgr)
        faces = detected[1] if isinstance(detected, tuple) else detected
        if faces is None:
            return []
        found = []
        for face in faces:
            try:
                x, y, fw, fh = (float(value) for value in face[:4])
                score = float(face[-1])
                aligned = self._recognizer.alignCrop(bgr, face)
                feature = self._recognizer.feature(aligned)
                embedding = l2_normalize(feature)
            except Exception:
                log.debug("skipped a face that could not be aligned", exc_info=True)
                continue
            bbox = clamp_bbox(x / width, y / height, fw / width, fh / height)
            found.append(DetectedFace(embedding, bbox, score))
        return found

    def _analyze_haar(self, bgr: np.ndarray) -> list[DetectedFace]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        height, width = gray.shape[:2]
        found = []
        for (x, y, fw, fh) in rects:
            crop = gray[y : y + fh, x : x + fw]
            if crop.size == 0:
                continue
            embedding = _spatial_descriptor(crop)
            bbox = clamp_bbox(x / width, y / height, fw / width, fh / height)
            found.append(DetectedFace(embedding, bbox, 0.5))
        return found


def _spatial_descriptor(gray_face: np.ndarray) -> np.ndarray:
    face = cv2.resize(gray_face, (16, 16), interpolation=cv2.INTER_AREA)
    face = cv2.equalizeHist(face)
    vector = face.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())
    return l2_normalize(vector)
