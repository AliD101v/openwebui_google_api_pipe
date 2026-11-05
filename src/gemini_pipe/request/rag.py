"""Memory extraction and the inline metadata markers, request side.

Memories arrive in the system prompt but OpenWebUI re-retrieves and re-ranks
them per request, so they are relocated to the last user message instead of
churn the system prefix. The invisible [](gemini:...) markers carry
per-message state (file ids, media anchors) through OpenWebUI storage.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class _Rag:
    _METADATA_MARKER = re.compile(PATTERN_METADATA_MARKER)

    def _extract_and_remove_memories(self, text: str) -> tuple[str, Optional[str]]:
        """Extract the OpenWebUI Memory System's memories out of the system
        prompt and remove them from it.

        Two injection formats are recognised, because OpenWebUI changed shape:
          * ``<memory_context>...</memory_context>`` (current, utils/memory.py)
            - can sit anywhere in the system message.
          * ``\\nUser Context:\\n...`` (legacy) - runs to the end of the string.

        Both are re-retrieved and re-ranked per request, so they are never
        stable across turns. Leaving them in the system prompt costs a full
        prefix rewrite every turn; the caller relocates the return value to
        the last user message instead.

        Returns:
            tuple[str, Optional[str]]: (cleaned_text, extracted_context)
            - cleaned_text: original text with all memory blocks removed (stripped)
            - extracted_context: the extracted memories, or None if none found
        """
        # Fast path: two substring scans are far cheaper than two regex scans,
        # and presence of the marker is the ground truth. Deliberately *not*
        # gated on a config flag - OpenWebUI decides to inject based on the
        # request's features.memory plus an admin-level config, neither of
        # which the pipe can observe reliably.
        if "<memory_context>" not in text and "User Context:" not in text:
            return text.strip(), None

        extracted_parts: List[str] = []

        # <memory_context> may appear anywhere; strip every occurrence.
        def _take_memory_context(match) -> str:
            content = match.group(1).strip()
            if content:
                extracted_parts.append(content)
            return ""

        cleaned_text = PATTERN_MEMORY_CONTEXT.sub(_take_memory_context, text)

        # Legacy tail form.
        match = PATTERN_USER_CONTEXT.search(cleaned_text)
        if match:
            context_content = match.group(1).strip()
            if context_content:
                extracted_parts.append(f"User Context:\n{context_content}")
            # Remove "\nUser Context:\n" and everything after it.
            cleaned_text = cleaned_text[: match.start()]

        extracted_context = "\n\n".join(extracted_parts) if extracted_parts else None
        return cleaned_text.strip(), extracted_context

    def _create_metadata_marker(self, id: str, value: str, messagenum: int = 0) -> str:
        """Build a URL-encoded inline metadata marker for embedding in
        assistant text; it renders as nothing but round-trips the payload."""
        encoded_value = quote(value, safe="")
        return f" [](gemini:{messagenum}:{id}:{encoded_value}) "

    def _remove_sources_from_rag(self, rag_content: str, filenames_to_remove: List[str]) -> str:
        """Remove the <source> tags whose name attribute matches one of the
        filenames from a RAG template.

        Returns the modified template, or "" when every source was removed
        (an empty <context></context> or no sources at all are left): the
        caller then drops the whole RAG block.
        """
        if not filenames_to_remove:
            return rag_content

        modified = rag_content
        for filename in filenames_to_remove:
            pattern = re.compile(
                rf'<source[^>]*name="{re.escape(filename)}"[^>]*>.*?</source>\s*',
                re.DOTALL,
            )
            modified = pattern.sub("", modified)

        if PATTERN_EMPTY_CONTEXT.search(modified) or not PATTERN_SOURCE_TAGS.search(modified):
            logger.debug("RAG: all sources removed, clearing entire RAG message")
            return ""

        logger.debug(
            f"RAG: removed {len(filenames_to_remove)} source(s) from RAG content"
        )
        return modified

    def _remove_specific_sources_from_rag_message(
        self, processed_messages: List[dict], filenames_to_remove: List[str]
    ) -> None:
        """Remove the sources named in ``filenames_to_remove`` from the RAG
        template of the latest message (system or user) that carries one.

        Files sent natively to Gemini must not also arrive as RAG text. Other
        sources are kept; when all sources match, the whole RAG template goes.
        The messages are modified in place.
        """
        if not filenames_to_remove:
            return

        for i in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[i]
            if not isinstance(msg, dict) or msg.get("role") not in ("system", "user"):
                continue

            content = msg.get("content")
            if isinstance(content, str):
                match = PATTERN_RAG_MESSAGE.search(content)
                if not match:
                    continue
                modified_rag = self._remove_sources_from_rag(
                    match.group(0), filenames_to_remove
                )
                start, end = match.span()
                if not modified_rag:
                    new_text = content[:start] + content[end:]
                else:
                    new_text = content[:start] + modified_rag + content[end:]
                msg["content"] = new_text.strip()
                return

            if not isinstance(content, list):
                continue

            modified = False
            new_content: List[dict] = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    new_content.append(block)
                    continue

                text = block.get("text", "")
                match = PATTERN_RAG_MESSAGE.search(text)
                if not match:
                    new_content.append(block)
                    continue

                modified_rag = self._remove_sources_from_rag(
                    match.group(0), filenames_to_remove
                )
                start, end = match.span()
                if not modified_rag:
                    new_text = text[:start] + text[end:]
                else:
                    new_text = text[:start] + modified_rag + text[end:]
                new_text = new_text.strip()
                if new_text:
                    new_block = dict(block)
                    new_block["text"] = new_text
                    new_content.append(new_block)
                modified = True

            if modified:
                msg["content"] = new_content
                return

    def _extract_metadata_marker_from_message(self, message) -> List[str]:
        """Extract the metadata payloads from the last assistant message and
        remove the markers from it in place."""
        metadata: List[str] = []
        if not isinstance(message, dict):
            return metadata
        if message.get("role") != "assistant":
            return metadata
        content = message.get("content")
        if isinstance(content, list):
            # Join all text blocks for searching, but also update blocks in-place.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block_text = block.get("text", "")
                    metadata.extend(self._METADATA_MARKER.findall(block_text))
                    block["text"] = self._METADATA_MARKER.sub("", block_text)
        elif isinstance(content, str):
            metadata.extend(self._METADATA_MARKER.findall(content))
            message["content"] = self._METADATA_MARKER.sub("", content)
        return metadata
