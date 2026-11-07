"""Native media for Gemini, request side: inline audio/video/full-context PDF
parts, the Google Files API with a cross-turn cache, and YouTube links found
in user messages (SPEC section 10).

A file is sent to Gemini directly instead of through OpenWebUI's text RAG when
Gemini understands it natively. Each file stays anchored to the user message
it was uploaded with on every later turn; the invisible media markers in the
assistant message remember the anchor.
"""
from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)


class _Media:
    _INLINE_TOTAL_MAX_BYTES = 90 * 1024 * 1024
    _PDF_INLINE_MAX_BYTES = 50 * 1024 * 1024
    _MEDIA_POLL_INTERVAL = 2.0
    _MEDIA_PROCESS_TIMEOUT = 300.0
    _MAX_YOUTUBE_LINKS = 10

    _PATTERN_YOUTUBE = re.compile(
        r"(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)"
        r"([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])"
    )

    async def _collect_native_media(
        self, ctx, client, messages, files, prior_entries
    ) -> dict:
        """The native media for this request.

        Returns ``{"parts": {user_idx: [Part, ...]}, "native_names":
        [filename, ...], "entries": ["<msgnum>:media:<value>", ...]}`` where
        ``user_idx`` is the 0-based index among the user messages and each
        entry is a media marker payload ready for ``_create_metadata_marker``.
        Parts are anchored to the message the file was uploaded with, so they
        stay in the same place of the history on follow-up turns.
        """
        parts_by_user_idx: Dict[int, List] = {}
        native_names: List[str] = []
        entries: List[str] = []

        candidates = self._media_candidates(messages, files, prior_entries)
        if getattr(ctx.user_valves, "USE_NATIVE_FILES", True):
            force_pdf = bool(
                getattr(self.valves, "NATIVE_PDF_DEFAULT", False)
                or (ctx.metadata or {}).get("gemini_native_pdf")
            )
            inline_total = 0
            ordered = sorted(
                candidates, key=lambda fid: (candidates[fid]["anchor"], fid)
            )
            for file_id in ordered:
                info = candidates[file_id]
                hint = info["mime_hint"]
                if hint is not None and not self._is_native_media(
                    hint, info["context"], info["prior"], force_pdf
                ):
                    continue
                name = info["name"]
                await ctx.status.activity(f"📎 Attaching {name}…")
                record, data = await self._read_media_file(file_id)
                if data is None:
                    logger.warning(f"native media: could not read file {file_id} ({name})")
                    continue
                mime = self._media_mime(record, info)
                if not self._is_native_media(mime, info["context"], info["prior"], force_pdf):
                    continue

                part: Optional[Any] = None
                size = len(data)
                files_api = str(getattr(self.valves, "FILES_API", "auto") or "auto")
                if (
                    files_api != "always"
                    and size <= self._inline_cap(mime)
                    and inline_total + size <= self._INLINE_TOTAL_MAX_BYTES
                ):
                    part = gtypes.Part.from_bytes(data=data, mime_type=mime)
                    inline_total += size
                elif files_api != "never":
                    try:
                        gfile = await self._ensure_gemini_file(
                            ctx, client, file_id, record, data, mime, name
                        )
                        part = gtypes.Part(
                            file_data=gtypes.FileData(file_uri=gfile.uri, mime_type=mime)
                        )
                    except Exception as e:
                        logger.warning(f"native media: Files API failed for {name}: {e}")
                        await ctx.status.notification(str(e))
                else:
                    await ctx.status.notification(
                        f"{name} is too large to send inline and the Files API is disabled"
                    )
                if part is None:
                    continue

                parts_by_user_idx.setdefault(info["anchor"], []).append(part)
                native_names.append(name)
                entries.append(f'{info["anchor"]}:media:{quote(f"{file_id}:{name}", safe="")}')

        self._collect_youtube_parts(messages, parts_by_user_idx)
        return {"parts": parts_by_user_idx, "native_names": native_names, "entries": entries}

    def _media_candidates(
        self, messages, files, prior_entries
    ) -> Dict[str, dict]:
        """The file candidates keyed by file id.

        A candidate is a file item of ``__files__``, of a user message's
        ``files`` list, or a prior ``media`` marker (OpenWebUI does not
        reliably re-send full-context files on follow-up turns, so those must
        be re-included from the marker). Each carries its name, context, mime
        hint and anchor: a prior marker's msgnum, else the index of the user
        message whose ``files`` list owns it, else the latest user message.
        """
        user_msgs = [
            m for m in (messages or []) if isinstance(m, dict) and m.get("role") == "user"
        ]
        latest = max(len(user_msgs) - 1, 0)

        prior_anchor: Dict[str, int] = {}
        prior_name: Dict[str, str] = {}
        for entry in prior_entries or []:
            parts = str(entry).split(":", 2)
            if len(parts) < 3 or parts[1] != "media":
                continue
            try:
                msg_idx = int(parts[0])
            except ValueError:
                continue
            decoded = unquote(parts[2])
            file_id, _, fname = decoded.partition(":")
            if file_id:
                prior_anchor[file_id] = msg_idx
                if fname:
                    prior_name[file_id] = fname

        owning_idx: Dict[str, int] = {}
        user_idx = -1
        for message in messages or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            user_idx += 1
            for item in message.get("files") or []:
                file_id = self._media_item_id(item)
                if file_id and file_id not in owning_idx:
                    owning_idx[file_id] = user_idx

        candidates: Dict[str, dict] = {}
        items = list(files or [])
        for message in user_msgs:
            items.extend(message.get("files") or [])
        for item in items:
            file_id = self._media_item_id(item)
            if not file_id or file_id in candidates:
                continue
            name = (item or {}).get("name") or prior_name.get(file_id, file_id)
            candidates[file_id] = {
                "name": name,
                "context": (item or {}).get("context"),
                "mime_hint": self._media_item_mime(item)
                or (mimetypes.guess_type(name)[0] if name else None),
                "anchor": prior_anchor.get(file_id, owning_idx.get(file_id, latest)),
                "prior": file_id in prior_anchor,
            }

        for file_id, msg_idx in prior_anchor.items():
            if file_id in candidates:
                continue
            name = prior_name.get(file_id, file_id)
            candidates[file_id] = {
                "name": name,
                "context": None,
                "mime_hint": mimetypes.guess_type(name)[0] if name else None,
                "anchor": msg_idx,
                "prior": True,
            }
        return candidates

    @staticmethod
    def _media_item_id(item) -> Optional[str]:
        if not isinstance(item, dict) or item.get("type") not in (None, "file"):
            return None
        file_id = item.get("id")
        return file_id if isinstance(file_id, str) and file_id else None

    @staticmethod
    def _media_item_mime(item) -> Optional[str]:
        if not isinstance(item, dict):
            return None
        file = item.get("file")
        if isinstance(file, dict):
            meta = file.get("meta")
            if isinstance(meta, dict) and meta.get("content_type"):
                return str(meta["content_type"])
        return None

    @staticmethod
    def _media_mime(record, info) -> Optional[str]:
        meta = getattr(record, "meta", None)
        if isinstance(meta, dict) and meta.get("content_type"):
            return str(meta["content_type"])
        return info.get("mime_hint")

    @staticmethod
    def _is_native_media(
        mime: Optional[str], context, has_prior_marker: bool, force_pdf: bool = False
    ) -> bool:
        """Whether Gemini takes this file natively: audio and video always,
        PDF when full-context, already sent on a prior turn or forced (the
        Native PDFs toggle / NATIVE_PDF_DEFAULT), else RAG."""
        if not mime:
            return False
        mime = str(mime).lower()
        if mime.startswith("audio/") or mime.startswith("video/"):
            return True
        if mime == "application/pdf":
            return context == "full" or has_prior_marker or force_pdf
        return False

    def _inline_cap(self, mime: Optional[str]) -> int:
        cap = max(int(getattr(self.valves, "INLINE_FILE_MAX_MB", 0) or 0), 0) * 1024 * 1024
        if str(mime or "").lower() == "application/pdf":
            cap = min(cap, self._PDF_INLINE_MAX_BYTES)
        return cap

    @staticmethod
    async def _read_media_file(file_id: str) -> Tuple[Optional[Any], Optional[bytes]]:
        """(record, bytes) behind an OpenWebUI file id, or (record, None) when
        it cannot be read (module missing, unknown id, unreadable path)."""
        if not FILES_AVAILABLE:
            return None, None
        try:
            record = await Files.get_file_by_id(id=file_id)
        except Exception as e:
            logger.warning(f"native media: could not look up file {file_id}: {e}")
            return None, None
        if record is None:
            return None, None
        location = getattr(record, "path", None)
        if not location:
            return None, None
        try:
            path = Storage.get_file(location)
            with open(path, "rb") as handle:
                return record, handle.read()
        except Exception as e:
            logger.warning(f"native media: could not read file {file_id}: {e}")
            return None, None

    async def _ensure_gemini_file(
        self, ctx, client, file_id, record, data, mime, name
    ) -> gtypes.File:
        """An ACTIVE Files API file for the bytes: a cached entry confirmed
        with files.get, or a fresh upload polled to ACTIVE.

        The cache is two-tier: the Pipe-level dict for this process, and the
        file's own meta (survives restarts), both keyed by API-key fingerprint, file id and content hash. Raises
        RuntimeError when Google fails or never finishes processing.
        """
        cache_key = (
            f"{self._api_key_fingerprint(ctx.api_key)}:{file_id}:"
            f"{getattr(record, 'hash', '')}"
        )
        entry = Pipe._gemini_file_cache.get(cache_key)
        if entry is None:
            meta = getattr(record, "meta", None)
            if isinstance(meta, dict):
                entry = (meta.get("gemini_files") or {}).get(cache_key)
        if entry:
            try:
                expires = float(entry.get("expires") or 0)
            except (TypeError, ValueError):
                expires = 0.0
            if expires - time.time() > 3600:
                gfile = None
                try:
                    gfile = await client.aio.files.get(name=entry["name"])
                except Exception as e:
                    logger.warning(f"Files API: could not confirm {entry['name']}: {e}")
                if gfile is not None and str(getattr(gfile.state, "name", "")) == "ACTIVE":
                    return gfile

        gfile = await client.aio.files.upload(
            file=io.BytesIO(data),
            config=gtypes.UploadFileConfig(mime_type=mime, display_name=name),
        )
        state = str(getattr(gfile.state, "name", ""))
        if state == "FAILED":
            raise RuntimeError(f"Google rejected {name} for processing")
        deadline = time.monotonic() + self._MEDIA_PROCESS_TIMEOUT
        while state == "PROCESSING":
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Google took too long to process {name}")
            await ctx.status.activity(f"⏳ Google is processing {name}…")
            await asyncio.sleep(self._MEDIA_POLL_INTERVAL)
            gfile = await client.aio.files.get(name=gfile.name)
            state = str(getattr(gfile.state, "name", ""))
            if state == "FAILED":
                raise RuntimeError(f"Google failed to process {name}")
        if state != "ACTIVE":
            raise RuntimeError(f"Google could not process {name} (state {state})")

        stored = {
            "name": gfile.name,
            "uri": gfile.uri,
            "mime": mime,
            "expires": (
                gfile.expiration_time.timestamp()
                if getattr(gfile, "expiration_time", None) is not None
                else 0
            ),
        }
        Pipe._gemini_file_cache[cache_key] = stored
        if FILES_AVAILABLE:
            try:
                meta = dict(getattr(record, "meta", None) or {})
                gemini_files = dict(meta.get("gemini_files") or {})
                gemini_files[cache_key] = stored
                await Files.update_file_metadata_by_id(
                    record.id, {"gemini_files": gemini_files}
                )
            except Exception as e:
                logger.warning(f"could not store the Files API entry on file {file_id}: {e}")
        return gfile

    def _collect_youtube_parts(self, messages, parts_by_user_idx) -> None:
        """YouTube links in user messages become video file_data parts at their
        message's index (the links stay in the text as well)."""
        policy = str(getattr(self.valves, "YOUTUBE_LINKS", "most_recent") or "off")
        if policy == "off":
            return
        user_msgs = [
            m for m in (messages or []) if isinstance(m, dict) and m.get("role") == "user"
        ]
        if not user_msgs:
            return

        if policy == "latest_message":
            indexes = [len(user_msgs) - 1]
        elif policy == "most_recent":
            found = [
                i
                for i in range(len(user_msgs) - 1, -1, -1)
                if self._youtube_links(self._block_text(user_msgs[i].get("content")))
            ]
            indexes = found[:1]
        else:  # "all"
            indexes = list(range(len(user_msgs) - 1, -1, -1))

        seen = set()
        for idx in indexes:
            for video_id in self._youtube_links(
                self._block_text(user_msgs[idx].get("content"))
            ):
                if video_id in seen:
                    continue
                seen.add(video_id)
                parts_by_user_idx.setdefault(idx, []).append(
                    gtypes.Part(
                        file_data=gtypes.FileData(
                            file_uri=f"https://www.youtube.com/watch?v={video_id}"
                        )
                    )
                )
                if len(seen) >= self._MAX_YOUTUBE_LINKS:
                    return

    @classmethod
    def _youtube_links(cls, text: str) -> List[str]:
        """The video ids of every YouTube link (watch, youtu.be, shorts, live)
        in the text, in order of appearance."""
        if not text:
            return []
        return cls._PATTERN_YOUTUBE.findall(text)
