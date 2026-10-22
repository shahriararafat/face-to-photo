"""List and download images from a public Google Drive folder.

Uses Drive API v3 pagination when ``GOOGLE_DRIVE_API_KEY`` is set, and the
public folder HTML otherwise. No OAuth and no private-folder access.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger("facescan.drive")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_FOLDER_IN_PATH = re.compile(r"/folders/([a-zA-Z0-9_-]{10,})")
_ID_PARAM = re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})")
_BARE_ID = re.compile(r"^[a-zA-Z0-9_-]{10,}$")
_FILE_HREF = re.compile(r"/file/d/([a-zA-Z0-9_-]{10,})")
_FOLDER_HREF = re.compile(r"/folders/([a-zA-Z0-9_-]{10,})")
_CONFIRM = re.compile(r"confirm=([0-9A-Za-z_\-]+)")
_CONFIRM_INPUT = re.compile(r'name="confirm"\s+value="([^"]+)"')

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}

_DENIED_MARKERS = (
    "you need access",
    "request access",
    "access denied",
    "sorry, the file you have requested does not exist",
)


class DriveError(Exception):
    """A problem the user can act on."""


class FolderLinkError(ValueError):
    """The pasted link is not a Drive folder."""


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    mime_type: str = "image/jpeg"


@dataclass
class Listing:
    files: list[DriveFile] = field(default_factory=list)
    truncated: bool = False


def parse_folder_id(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise FolderLinkError("Paste a public Google Drive folder link.")
    if "/file/d/" in raw or "/file/d/" in raw.replace("\\", "/"):
        raise FolderLinkError(
            "That link is a single file. Paste the shared folder link instead."
        )
    bare = _BARE_ID.match(raw)
    if bare is None and not re.search(r"drive\.google\.com|docs\.google\.com", raw):
        raise FolderLinkError("Paste a public Google Drive folder link.")
    match = _FOLDER_IN_PATH.search(raw) or _ID_PARAM.search(raw)
    if match:
        return match.group(1)
    if bare:
        return raw
    raise FolderLinkError("Couldn't find a folder id in that link.")


def is_image_name(name: str) -> bool:
    suffix = Path(name or "").suffix.lower()
    if suffix == "":
        return True
    return suffix in IMAGE_EXTS


def listing_kind(html: str) -> str:
    if "flip-entry" in html or "flip-entries" in html:
        return "embedded"
    lowered = html.lower()
    if "no files in this folder" in lowered or "this folder is empty" in lowered:
        return "empty"
    return "unknown"


def access_denied(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in _DENIED_MARKERS)


def parse_title(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return ""
    soup = BeautifulSoup(html, "lxml")
    if not soup.title or not soup.title.string:
        return ""
    title = re.sub(r"\s*-\s*Google Drive\s*$", "", soup.title.string.strip(), flags=re.I)
    if not title or title.lower() in {"google drive", "drive"}:
        return ""
    return title


def parse_drive_html(html: str, parent_id: str) -> tuple[list[DriveFile], list[str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    images: dict[str, DriveFile] = {}
    folders: list[str] = []
    seen_folders = {parent_id}

    def add_folder(folder_id: str) -> None:
        if folder_id and folder_id not in seen_folders:
            seen_folders.add(folder_id)
            folders.append(folder_id)

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        name = anchor.get_text(strip=True) or ""
        file_match = _FILE_HREF.search(href)
        if file_match:
            file_id = file_match.group(1)
            if is_image_name(name):
                images.setdefault(
                    file_id,
                    DriveFile(file_id, name or f"{file_id}.jpg", "image/jpeg"),
                )
            continue
        folder_match = _FOLDER_HREF.search(href)
        if folder_match:
            add_folder(folder_match.group(1))

    if not images:
        for entry in soup.select("[id^=entry-]"):
            entry_id = (entry.get("id") or "")[6:]
            if not re.fullmatch(r"[a-zA-Z0-9_-]{10,}", entry_id):
                continue
            link = entry.find("a", href=True)
            href = link["href"] if link else ""
            name = link.get_text(strip=True) if link else ""
            if "/folders/" in href:
                add_folder(entry_id)
            elif is_image_name(name):
                images.setdefault(
                    entry_id,
                    DriveFile(entry_id, name or f"{entry_id}.jpg", "image/jpeg"),
                )

    return list(images.values()), folders


def looks_like_image(data: bytes, content_type: str = "") -> bool:
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return True
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    if data[:2] == b"BM":
        return True
    return content_type.lower().startswith("image/")


def sniff_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return ".gif"
    return ".jpg"


class GoogleDriveClient:
    def __init__(
        self,
        api_key: str = "",
        max_photos: int = 10000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.max_photos = max_photos
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def list_images(self, folder_id: str) -> Listing:
        api_error: DriveError | None = None
        if self.api_key:
            try:
                return await self._list_api(folder_id)
            except DriveError as exc:
                api_error = exc
                log.warning("Drive API listing failed, trying the public page: %s", exc)
        try:
            listing = await self._list_public(folder_id)
        except DriveError:
            if api_error:
                raise api_error
            raise
        if not listing.files and api_error:
            raise api_error
        return listing

    async def get_folder_name(self, folder_id: str) -> str:
        if self.api_key:
            response = await self._request(
                "GET",
                f"https://www.googleapis.com/drive/v3/files/{folder_id}",
                params={"fields": "name", "key": self.api_key},
            )
            if response.status_code == 200:
                name = (response.json() or {}).get("name") or ""
                if name:
                    return name
        response = await self._request(
            "GET", f"https://drive.google.com/drive/folders/{folder_id}"
        )
        if response.status_code == 200:
            title = parse_title(response.text)
            if title:
                return title
        return "Photo folder"

    async def download_for_index(self, file_id: str) -> bytes:
        for url in (
            f"https://drive.google.com/thumbnail?id={file_id}&sz=w1600",
            f"https://lh3.googleusercontent.com/d/{file_id}=w1600",
        ):
            data = await self._read_image(url, max_bytes=25 * 1024 * 1024)
            if data and len(data) > 2000:
                return data
        return await self.download_original(file_id)

    async def download_original(self, file_id: str) -> bytes:
        if self.api_key:
            data = await self._read_image(
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                params={"alt": "media", "key": self.api_key},
                max_bytes=80 * 1024 * 1024,
            )
            if data:
                return data
        data = await self._uc_download(file_id)
        if data:
            return data
        raise DriveError("Google Drive didn't return that photo.")

    async def _list_api(self, root_id: str) -> Listing:
        found: list[DriveFile] = []
        seen_files: set[str] = set()
        seen_folders: set[str] = set()
        queue = [root_id]
        truncated = False
        while queue and len(found) < self.max_photos:
            folder_id = queue.pop(0)
            if folder_id in seen_folders:
                continue
            seen_folders.add(folder_id)
            page_token = None
            while len(found) < self.max_photos:
                params = {
                    "q": f"'{folder_id}' in parents and trashed = false",
                    "pageSize": 1000,
                    "fields": "nextPageToken,files(id,name,mimeType)",
                    "key": self.api_key,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = await self._request(
                    "GET", "https://www.googleapis.com/drive/v3/files", params=params
                )
                if response.status_code != 200:
                    raise DriveError(_api_error(response))
                payload = response.json()
                for item in payload.get("files") or []:
                    mime = item.get("mimeType") or ""
                    file_id = item.get("id") or ""
                    name = item.get("name") or file_id
                    if not file_id:
                        continue
                    if mime == "application/vnd.google-apps.folder":
                        queue.append(file_id)
                        continue
                    if not _is_image(mime, name) or file_id in seen_files:
                        continue
                    seen_files.add(file_id)
                    found.append(DriveFile(file_id, name, mime or "image/jpeg"))
                    if len(found) >= self.max_photos:
                        truncated = True
                        break
                page_token = payload.get("nextPageToken")
                if not page_token or len(found) >= self.max_photos:
                    if page_token and len(found) >= self.max_photos:
                        truncated = True
                    break
        return Listing(found, truncated=truncated)

    async def _list_public(self, root_id: str) -> Listing:
        found: dict[str, DriveFile] = {}
        seen_folders: set[str] = set()
        queue: list[tuple[str, int]] = [(root_id, 0)]
        truncated = False
        saw_listing = False
        while queue and len(found) < self.max_photos:
            folder_id, depth = queue.pop(0)
            if folder_id in seen_folders:
                continue
            seen_folders.add(folder_id)
            images, subs, listed = await self._scrape_folder(folder_id)
            saw_listing = saw_listing or listed
            for image in images:
                if image.file_id in found:
                    continue
                found[image.file_id] = image
                if len(found) >= self.max_photos:
                    truncated = True
                    break
            if depth < 4 and len(found) < self.max_photos:
                for sub in subs:
                    if sub not in seen_folders:
                        queue.append((sub, depth + 1))
        if not found and saw_listing:
            raise DriveError("This folder has no photos.")
        if not found:
            raise DriveError(
                "Couldn't list that folder. Share it as Anyone with the link. "
                "Folders with more than about 100 photos need a free Google Drive API key."
            )
        return Listing(list(found.values()), truncated=truncated)

    async def _scrape_folder(self, folder_id: str) -> tuple[list[DriveFile], list[str], bool]:
        images: dict[str, DriveFile] = {}
        folders: list[str] = []
        saw_listing = False
        denials = 0
        attempts = 0
        urls = (
            f"https://drive.google.com/embeddedfolderview?id={folder_id}",
            f"https://drive.google.com/drive/folders/{folder_id}",
        )
        network_errors = 0
        for url in urls:
            attempts += 1
            try:
                response = await self._request("GET", url)
            except DriveError:
                network_errors += 1
                continue
            if response.status_code in {401, 403, 404} or access_denied(response.text):
                denials += 1
                continue
            if response.status_code != 200:
                continue
            kind = listing_kind(response.text)
            parsed_images, parsed_folders = parse_drive_html(response.text, folder_id)
            if kind in {"embedded", "empty"} or parsed_images or parsed_folders:
                saw_listing = True
            for image in parsed_images:
                images.setdefault(image.file_id, image)
            folders.extend(parsed_folders)
        if not images and not folders and attempts and denials == attempts:
            raise DriveError(
                "This folder isn't public. In Google Drive, set sharing to Anyone with the link."
            )
        if not images and not folders and attempts and network_errors == attempts:
            raise DriveError("Couldn't reach Google Drive. Try again in a moment.")
        unique_folders: list[str] = []
        seen = {folder_id}
        for folder in folders:
            if folder not in seen:
                seen.add(folder)
                unique_folders.append(folder)
        return list(images.values()), unique_folders, saw_listing

    async def _uc_download(self, file_id: str) -> bytes | None:
        response = await self._request(
            "GET",
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id},
        )
        data = response.content
        if looks_like_image(data, response.headers.get("content-type", "")):
            return data
        text = data.decode("utf-8", errors="ignore")
        token_match = _CONFIRM.search(text) or _CONFIRM_INPUT.search(text)
        if not token_match:
            view = await self._read_image(
                "https://drive.google.com/uc",
                params={"export": "view", "id": file_id},
                max_bytes=80 * 1024 * 1024,
            )
            return view
        confirmed = await self._read_image(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id, "confirm": token_match.group(1)},
            max_bytes=80 * 1024 * 1024,
        )
        return confirmed

    async def _read_image(
        self,
        url: str,
        params: dict | None = None,
        max_bytes: int = 25 * 1024 * 1024,
    ) -> bytes | None:
        try:
            response = await self._request("GET", url, params=params)
        except DriveError:
            return None
        if response.status_code != 200:
            return None
        data = response.content
        if len(data) > max_bytes:
            return None
        if not looks_like_image(data, response.headers.get("content-type", "")):
            return None
        return data

    async def _request(self, method: str, url: str, params: dict | None = None) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.client.request(method, url, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                await _nap(attempt)
                continue
            if response.status_code in {429, 500, 502, 503, 504}:
                await _nap(attempt)
                continue
            return response
        raise DriveError("Couldn't reach Google Drive. Try again in a moment.") from last_error


def _is_image(mime: str, name: str) -> bool:
    if mime.startswith("image/"):
        return True
    if mime.startswith("application/vnd.google-apps"):
        return False
    suffix = Path(name).suffix.lower()
    return suffix in IMAGE_EXTS


def _api_error(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
    except Exception:
        message = (response.text or "").strip()[:180] or "request failed"
    if response.status_code in {401, 403}:
        return (
            "Google Drive refused the API key. Enable the Drive API for that key "
            f"and make sure the folder is public. ({message})"
        )
    if response.status_code == 404:
        return "That folder was not found. Check the link and sharing."
    return f"Google Drive API error: {message}"


async def _nap(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(0.35 * (attempt + 1))
