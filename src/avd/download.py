"""Single-video download and reliable postprocessed path resolution."""

from pathlib import Path
from urllib.parse import urlparse

from .errors import DubbingError


def download_video(url: str, directory: Path, height: int = 720) -> Path:
    host = (urlparse(url).hostname or "").lower()
    if urlparse(url).scheme not in {"http", "https"} or not (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    ):
        raise DubbingError("Provide a valid YouTube video URL.")
    import yt_dlp

    directory.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []

    def finished(filename: str) -> None:
        completed.append(Path(filename))

    options = {
        "format": f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
        "outtmpl": str(directory / "%(id)s.%(ext)s"),
        "merge_output_format": "mkv", "noplaylist": True,
        "restrictfilenames": True, "post_hooks": [finished],
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            candidates = list(reversed(completed))
            if info and info.get("filepath"):
                candidates.append(Path(info["filepath"]))
            if info:
                candidates.append(Path(downloader.prepare_filename(info)))
            for path in candidates:
                if path.is_file() and path.stat().st_size > 0:
                    return path.resolve()
    except yt_dlp.utils.DownloadError as exc:
        raise DubbingError(f"YouTube download failed: {exc}") from exc
    raise DubbingError("Download finished without a confirmed final file path.")
