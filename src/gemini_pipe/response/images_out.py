"""Generated-image output: render a placeholder for each streamed image, store
it in OpenWebUI's file storage, and replace the placeholder with the inline
markdown link (a short warning line when it could not be saved).
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import mimetypes
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class _ImagesOut:
    async def _start_image_output(self, ctx, call, part) -> None:
        """Render a "saving image" placeholder and start storing one generated
        image; the store result replaces the placeholder once it finishes."""
        key = f"img:{call.index}:{len(call.image_tasks)}"
        await ctx.upsert_block(key, "\n_🖼️ Saving image…_\n", call.index)
        await ctx.status.activity("🖼️ Saving image…")
        inline = _get(part, "inline_data")
        data = _get(inline, "data") or b""
        mime = _get(inline, "mime_type") or "image/png"

        async def _save() -> None:
            url = await self._store_generated_image(ctx, data, mime)
            if url:
                block = f"\n![Generated image]({url})\n"
            else:
                block = self._format_notice("Generated image could not be saved")
            await ctx.upsert_block(key, block, call.index)

        call.image_tasks.append(asyncio.create_task(_save()))

    async def _store_generated_image(self, ctx, data: bytes, mime: str) -> Optional[str]:
        """Persist one generated image in OpenWebUI and return its /files
        content URL; None when no storage is available or the store fails."""
        metadata = {
            "chat_id": (ctx.metadata or {}).get("chat_id"),
            "message_id": (ctx.metadata or {}).get("message_id"),
        }
        user_id = (ctx.user or {}).get("id")
        try:
            if UPLOAD_IMAGE_AVAILABLE and USERS_AVAILABLE and ctx.request:
                user = await Users.get_user_by_id(user_id)
                file_item, _ = await upload_image(
                    ctx.request, data, mime, metadata, user
                )
                return f"/api/v1/files/{file_item.id}/content"
            if FILES_AVAILABLE:
                extension = (
                    mimetypes.guess_extension(mime)
                    or mimetypes.guess_extension(mime.split(";")[0])
                    or ".png"
                )
                file_id = str(uuid.uuid4())
                filename = f"generated-image{extension}"
                _, file_path = Storage.upload_file(io.BytesIO(data), filename, {})
                await Files.insert_new_file(
                    user_id=user_id,
                    form_data=FileForm(
                        id=file_id,
                        hash=hashlib.sha256(data).hexdigest(),
                        filename=filename,
                        path=file_path,
                        data={},
                        meta={
                            "content_type": mime,
                            "size": len(data),
                            "source": "gemini_image_generation",
                        },
                    ),
                )
                return f"/api/v1/files/{file_id}/content"
            logger.warning(
                "no OpenWebUI file storage available for a generated image"
            )
            return None
        except Exception as e:
            logger.warning(f"could not store generated image: {e}")
            return None
