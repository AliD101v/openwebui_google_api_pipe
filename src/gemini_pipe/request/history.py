"""Chat history -> Gemini Contents (text and images), request side.

The assistant's persisted text is reduced to its replayable prose: everything
this pipe rendered into it (reasoning, tool calls, search blocks, citations,
notices) is stripped, so a continued conversation carries no UI artefacts.
"""
from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class _History:
    _IMAGE_SEND_AS_IS = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    )
    _IMAGE_CONVERT_TO_PNG = ("image/gif", "image/bmp", "image/tiff", "image/avif")

    _PATTERN_METADATA_MARKER = re.compile(PATTERN_METADATA_MARKER)
    _PATTERN_REASONING_DETAILS = re.compile(
        r'<details\b[^>]*\btype="reasoning"[^>]*>.*?</details>', flags=re.DOTALL
    )
    _PATTERN_BUILTIN_DETAILS = re.compile(
        r'<details\b[^>]*\bdata-gemini-builtin="1"[^>]*>.*?</details>',
        flags=re.DOTALL,
    )
    _PATTERN_TOOL_CALLS_DETAILS = re.compile(
        r'<details\b[^>]*\btype="tool_calls"[^>]*>.*?</details>', flags=re.DOTALL
    )
    _PATTERN_FILES_URL = re.compile(r"/api/v1/files/([0-9a-fA-F-]+)/content")
    _PATTERN_DATA_URL = re.compile(r"^data:[^,]*;base64,(.*)$", flags=re.DOTALL)
    _PATTERN_CITATION_RUN = re.compile(r"(?:\[\d+\])+")
    _PATTERN_CODE_SPANS = re.compile(r"```.*?```|`[^`\n]*`", flags=re.DOTALL)
    _SUGGESTION_LINE_PREFIX = "🔎 **Google Search suggestions:**"
    _NOTICE_LINE_PREFIX = "> ⚠️"

    async def _convert_history(
        self, messages, info, ctx
    ) -> Tuple[Optional[str], List[gtypes.Content]]:
        """The system instruction and the Gemini Contents for one request.

        Memories are extracted out of the system prompt and appended to the
        last user message instead: they are re-ranked per request, so they
        must not churn the (cacheable) system prefix.
        """
        messages = messages or []
        last_user_index = -1
        for i, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user_index = i

        system_parts: List[str] = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                text = self._block_text(message.get("content", ""))
                if text.strip():
                    system_parts.append(text.strip())
        system_instruction = "\n\n".join(system_parts).strip() or None

        memories: Optional[str] = None
        if system_instruction:
            system_instruction, memories = self._extract_and_remove_memories(
                system_instruction
            )
            if not system_instruction:
                system_instruction = None

        contents: List[gtypes.Content] = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                parts = await self._user_parts(message.get("content", ""), ctx)
                if memories is not None and i == last_user_index:
                    parts = self._append_memories(parts, memories)
                if parts:
                    contents.append(gtypes.Content(role="user", parts=parts))
            elif role == "assistant":
                text = self._assistant_text(message.get("content", ""))
                if text:
                    contents.append(
                        gtypes.Content(role="model", parts=[gtypes.Part(text=text)])
                    )

        merged: List[gtypes.Content] = []
        for content in contents:
            if not content.parts:
                continue
            if merged and merged[-1].role == content.role:
                merged[-1].parts.extend(content.parts)
            else:
                merged.append(content)
        return system_instruction, merged

    @staticmethod
    def _block_text(content: Any) -> str:
        """The prose of a message content: the string itself, or its text blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    async def _user_parts(self, content: Any, ctx) -> List[gtypes.Part]:
        """The parts of one user message: text as-is, images as inline_data."""
        parts: List[gtypes.Part] = []
        if isinstance(content, str):
            if content:
                parts.append(gtypes.Part(text=content))
            return parts
        if not isinstance(content, list):
            return parts
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text") or ""
                if text:
                    parts.append(gtypes.Part(text=text))
            elif block_type == "image_url":
                url = (block.get("image_url") or {}).get("url") or ""
                part = await self._image_part(url)
                if part is not None:
                    parts.append(part)
        return parts

    def _append_memories(
        self, parts: List[gtypes.Part], memories: str
    ) -> List[gtypes.Part]:
        """Append the relocated memories under the standard header, after the
        last text part of the message (or as a new part when there is none)."""
        suffix = MEMORY_CONTEXT_APPENDIX_HEADER + memories
        for index in range(len(parts) - 1, -1, -1):
            if parts[index].text:
                parts[index] = gtypes.Part(text=parts[index].text + suffix)
                return parts
        parts.append(gtypes.Part(text=suffix))
        return parts

    async def _image_part(self, url: str) -> Optional[gtypes.Part]:
        """The inline_data part for one image URL, or None when it cannot be
        sent (fetch failure, unrecognised type, no converter)."""
        if not url:
            return None
        file_match = self._PATTERN_FILES_URL.search(url)
        if url.startswith("data:"):
            data = self._decode_data_url(url)
        elif FILES_AVAILABLE and file_match:
            data = await self._read_file_bytes(file_match.group(1))
        elif url.startswith(("http://", "https://")):
            data = await self._fetch_url_image(url)
        else:
            return None
        if data is None:
            return None
        prepared = self._prepare_image_bytes(data, self._sniff_image_media_type(data))
        if prepared is None:
            return None
        image_bytes, mime_type = prepared
        return gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    @classmethod
    def _decode_data_url(cls, url: str) -> Optional[bytes]:
        match = cls._PATTERN_DATA_URL.match(url)
        if not match:
            logger.debug("skipping image: not a base64 data URL")
            return None
        try:
            return base64.b64decode(match.group(1))
        except Exception as e:
            logger.debug(f"skipping image: invalid base64 data URL: {e}")
            return None

    @staticmethod
    async def _read_file_bytes(file_id: str) -> Optional[bytes]:
        """The bytes behind an /api/v1/files/<uuid>/content URL, via OpenWebUI's
        Files registry and storage provider."""
        try:
            file = await Files.get_file_by_id(id=file_id)
        except Exception as e:
            logger.warning(f"could not look up file {file_id}: {e}")
            return None
        if file is None:
            logger.debug(f"skipping image: unknown file id {file_id}")
            return None
        location = getattr(file, "path", None)
        if not location:
            logger.debug(f"skipping image: file {file_id} has no storage location")
            return None
        try:
            path = Storage.get_file(location)
            with open(path, "rb") as handle:
                return handle.read()
        except Exception as e:
            logger.warning(f"could not read file {file_id}: {e}")
            return None

    @staticmethod
    async def _fetch_url_image(url: str) -> Optional[bytes]:
        """The bytes of an http(s) image URL, through OpenWebUI's SSRF-safe
        fetcher; None when the fetcher is absent or the fetch fails."""
        if not IMAGE_URL_FETCH_AVAILABLE:
            logger.debug("skipping image URL: the OpenWebUI fetcher is not available")
            return None
        try:
            encoded = await get_image_base64_from_url(url)
        except Exception as e:
            logger.warning(f"could not fetch image URL {url}: {e}")
            return None
        if not encoded:
            logger.debug(f"skipping image URL: empty fetch result for {url}")
            return None
        try:
            return base64.b64decode(encoded)
        except Exception as e:
            logger.debug(f"skipping image URL: invalid base64 from {url}: {e}")
            return None

    def _prepare_image_bytes(
        self, data: bytes, mime_type: Optional[str]
    ) -> Optional[Tuple[bytes, str]]:
        """(bytes, mime) ready to send, or None: types the API takes are sent
        as-is, GIF/BMP/TIFF/AVIF are converted to PNG when Pillow is
        installed, and anything else is skipped."""
        if not mime_type:
            logger.debug("skipping image: unrecognised image type")
            return None
        if mime_type in self._IMAGE_SEND_AS_IS:
            return data, mime_type
        if mime_type in self._IMAGE_CONVERT_TO_PNG:
            if not PIL_AVAILABLE:
                logger.warning(
                    f"skipping {mime_type} image: Pillow is not installed, "
                    "so it cannot be converted to PNG"
                )
                return None
            try:
                image = PILImage.open(io.BytesIO(data))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue(), "image/png"
            except Exception as e:
                logger.warning(f"skipping {mime_type} image: conversion to PNG failed: {e}")
                return None
        logger.debug(f"skipping image: unsupported type {mime_type}")
        return None

    @staticmethod
    def _sniff_image_media_type(raw: bytes) -> Optional[str]:
        """Identify an image from its leading bytes, ignoring any declared type.

        Necessary because OpenWebUI's label is wrong in two different ways
        (both in MessageInput.svelte):

          * Its HEIC branch tests `file.type === 'image/heic'` exactly, so
            `image/heif`, the `*-sequence` variants, and the very common case
            of an empty `file.type` skip conversion entirely.
          * When conversion DOES run, the resulting JPEG is re-wrapped with
            `new File([blob], name, { type: file.type })` -- the ORIGINAL type.
            So a successfully converted image still arrives labelled HEIC.

        The bytes are the only reliable source.
        """
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
        # ISO-BMFF: size, then "ftyp", then the brand.
        if raw[4:8] == b"ftyp":
            brand = raw[8:12]
            if brand in (b"avif", b"avis"):
                return "image/avif"
            if brand in (b"heic", b"heix", b"heim", b"heis", b"heif", b"mif1"):
                return "image/heic"
        return None

    def _assistant_text(self, content: Any) -> str:
        """The replayable prose of an assistant message: its persisted text
        minus everything this pipe rendered into it."""
        text = self._block_text(content)
        if not text:
            return ""
        had_search_block = 'name="google_search"' in text
        text = self._PATTERN_METADATA_MARKER.sub("", text)
        text = self._PATTERN_REASONING_DETAILS.sub("", text)
        text = self._PATTERN_BUILTIN_DETAILS.sub("", text)
        text = self._PATTERN_TOOL_CALLS_DETAILS.sub("", text)
        text = self._strip_decorative_lines(text)
        if had_search_block:
            # Citation numbers are only valid within the call that produced
            # them; replayed text would point at the wrong sources, so the
            # markers go away with the search block.
            text = self._strip_citation_runs(text)
        return text.strip()

    def _strip_decorative_lines(self, text: str) -> str:
        kept = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith(self._SUGGESTION_LINE_PREFIX):
                continue
            if stripped.startswith(self._NOTICE_LINE_PREFIX):
                continue
            kept.append(line)
        return "\n".join(kept)

    def _strip_citation_runs(self, text: str) -> str:
        """Drop the [n] citation runs outside inline code and fenced blocks."""
        parts = self._PATTERN_CODE_SPANS.split(text)
        return "".join(
            self._PATTERN_CITATION_RUN.sub("", seg) if idx % 2 == 0 else seg
            for idx, seg in enumerate(parts)
        )
