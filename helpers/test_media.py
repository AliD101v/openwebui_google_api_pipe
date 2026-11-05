"""Checks for native media: inline audio/video/PDF, Files API with cache,
YouTube links and the marker persistence across turns (task 04)."""
from __future__ import annotations

import asyncio

from google.genai import types as gtypes

from _testlib import FakeClient, FakeEmitter, chunk, fake_owui_files, make_pipe, pipe_module

BODY = {
    "model": "gemini_pipe.gemini/gemini-4-argon",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}


def make_user(pipe, **user_valves):
    return {"id": "u1", "name": "T", "valves": pipe.UserValves(**user_valves)}


def answer_script(text="Done."):
    return [
        chunk(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        )
    ]


def gemini_file(name="files/abc", mime="video/mp4", state="ACTIVE"):
    """A real google.genai types.File for the fake Files API."""
    return gtypes.File.model_validate(
        {
            "name": name,
            "uri": f"https://generativelanguage.googleapis.com/v1beta/{name}",
            "mimeType": mime,
            "state": state,
            "expirationTime": "2099-01-01T00:00:00Z",
        }
    )


def run_pipe(pipe, fake, *, messages=None, files=None, user=None):
    """Run pipe() against the fake client and return (result, emitter)."""
    pipe._build_genai_client = lambda api_key, timeout_s=None: fake
    emitter = FakeEmitter()
    body = dict(BODY, messages=list(messages or BODY["messages"]))
    result = asyncio.run(
        pipe.pipe(
            body,
            __user__=user or make_user(pipe),
            __event_emitter__=emitter,
            __metadata__={"chat_id": "c1", "message_id": "m1"},
            __files__=files,
        )
    )
    return result, emitter


def user_contents(fake, call=0):
    return [c for c in fake.stream_calls[call]["contents"] if c.role == "user"]


def test_audio_inline_and_rag_source_removed(tmp_path):
    # criterion 1
    pipe = make_pipe(GEMINI_API_KEY="k")
    mod = pipe_module(pipe)
    audio = b"m" * 1024
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": "f1",
                "filename": "memo.mp3",
                "path": "files/memo.mp3",
                "hash": "h1",
                "content_type": "audio/mpeg",
                "data": audio,
            }
        ],
    )
    rag = (
        "### Task: Summarise\n<context>\n"
        '<source id="1" name="memo.mp3">transcript</source>\n'
        '<source id="2" name="notes.txt">x</source>\n'
        "</context>"
    )
    messages = [
        {"role": "system", "content": rag},
        {"role": "user", "content": "Summarise"},
    ]
    files = [
        {
            "type": "file",
            "id": "f1",
            "name": "memo.mp3",
            "file": {"meta": {"content_type": "audio/mpeg"}},
        }
    ]
    fake = FakeClient(scripts=[answer_script()])
    result, emitter = run_pipe(pipe, fake, messages=messages, files=files)

    user = user_contents(fake)
    assert len(user) == 1
    parts = user[0].parts
    assert len(parts) == 2
    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "audio/mpeg"
    assert parts[0].inline_data.data == audio
    assert parts[1].text == "Summarise"
    system_instruction = fake.stream_calls[0]["config"].system_instruction or ""
    assert "memo.mp3" not in system_instruction
    assert "notes.txt" in system_instruction
    assert "[](gemini:0:media:" in result


def test_follow_up_turn_anchored_to_first_user_message(tmp_path):
    # criterion 2
    pipe = make_pipe(GEMINI_API_KEY="k")
    mod = pipe_module(pipe)
    audio = b"m" * 1024
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": "f1",
                "filename": "memo.mp3",
                "path": "files/memo.mp3",
                "hash": "h1",
                "content_type": "audio/mpeg",
                "data": audio,
            }
        ],
    )
    marker = " [](gemini:0:media:f1%3Amemo.mp3) "
    messages = [
        {"role": "user", "content": "Summarise"},
        {"role": "assistant", "content": "A summary." + marker},
        {"role": "user", "content": "More detail"},
    ]
    fake = FakeClient(scripts=[answer_script()])
    result, emitter = run_pipe(pipe, fake, messages=messages, files=None)

    user = user_contents(fake)
    assert len(user) == 2
    assert user[0].parts[0].inline_data is not None
    assert user[0].parts[0].inline_data.mime_type == "audio/mpeg"
    assert user[1].parts[0].text == "More detail"
    assert not any(p.inline_data is not None for p in user[1].parts)
    assert "[](gemini:0:media:" in result


def test_files_api_upload_then_cache_reuse(tmp_path):
    # criterion 3
    pipe = make_pipe(GEMINI_API_KEY="k", INLINE_FILE_MAX_MB=1)
    pipe._MEDIA_POLL_INTERVAL = 0
    mod = pipe_module(pipe)
    video = b"v" * (2 * 1024 * 1024)
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": "f2",
                "filename": "clip.mp4",
                "path": "files/clip.mp4",
                "hash": "h2",
                "content_type": "video/mp4",
                "data": video,
            }
        ],
    )
    uploaded = gemini_file(state="PROCESSING")
    active = gemini_file(state="ACTIVE")
    fake = FakeClient(
        scripts=[answer_script()], files_upload=[uploaded], files_get=[active]
    )
    files = [
        {
            "type": "file",
            "id": "f2",
            "name": "clip.mp4",
            "file": {"meta": {"content_type": "video/mp4"}},
        }
    ]
    messages = [{"role": "user", "content": "Summarise the clip"}]
    result, emitter = run_pipe(pipe, fake, messages=messages, files=files)

    parts = [p for c in user_contents(fake) for p in c.parts]
    file_data_parts = [p for p in parts if p.file_data is not None]
    assert len(file_data_parts) == 1
    assert (
        file_data_parts[0].file_data.file_uri
        == "https://generativelanguage.googleapis.com/v1beta/files/abc"
    )
    assert file_data_parts[0].file_data.mime_type == "video/mp4"
    assert len(fake.upload_calls) == 1

    fake2 = FakeClient(scripts=[answer_script()], files_get=[active])
    run_pipe(pipe, fake2, messages=messages, files=files)
    assert not fake2.upload_calls, "cached file must not be uploaded again"
    assert len(fake.upload_calls) == 1


def test_pdf_inline_only_with_full_context(tmp_path):
    # criterion 4
    pipe = make_pipe(GEMINI_API_KEY="k")
    mod = pipe_module(pipe)
    pdf = b"%PDF-1.4 tiny"
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": "f3",
                "filename": "report.pdf",
                "path": "files/report.pdf",
                "hash": "h3",
                "content_type": "application/pdf",
                "data": pdf,
            }
        ],
    )
    messages = [{"role": "user", "content": "Summarise"}]
    full = [
        {
            "type": "file",
            "id": "f3",
            "name": "report.pdf",
            "context": "full",
            "file": {"meta": {"content_type": "application/pdf"}},
        }
    ]
    fake = FakeClient(scripts=[answer_script()])
    run_pipe(pipe, fake, messages=messages, files=full)
    parts = [p for c in user_contents(fake) for p in c.parts]
    inline = [p for p in parts if p.inline_data is not None]
    assert len(inline) == 1
    assert inline[0].inline_data.mime_type == "application/pdf"
    assert inline[0].inline_data.data == pdf

    plain = [
        {
            "type": "file",
            "id": "f3",
            "name": "report.pdf",
            "file": {"meta": {"content_type": "application/pdf"}},
        }
    ]
    fake2 = FakeClient(scripts=[answer_script()])
    run_pipe(pipe, fake2, messages=messages, files=plain)
    parts2 = [p for c in user_contents(fake2) for p in c.parts]
    assert not any(p.inline_data is not None for p in parts2)


def test_youtube_links_most_recent_and_off():
    # criterion 5
    pipe = make_pipe(GEMINI_API_KEY="k", YOUTUBE_LINKS="off")
    messages = [
        {"role": "user", "content": "Summarise https://youtu.be/abc123def45"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "And the ending?"},
    ]
    fake_off = FakeClient(scripts=[answer_script()])
    run_pipe(pipe, fake_off, messages=messages)
    off_parts = [p for c in user_contents(fake_off) for p in c.parts]
    assert not any(p.file_data is not None for p in off_parts)

    pipe.valves = pipe.Valves(GEMINI_API_KEY="k", YOUTUBE_LINKS="most_recent")
    fake = FakeClient(scripts=[answer_script()])
    run_pipe(pipe, fake, messages=messages)
    user = user_contents(fake)
    assert len(user) == 2
    first = [p for p in user[0].parts if p.file_data is not None]
    assert len(first) == 1
    assert first[0].file_data.file_uri == "https://www.youtube.com/watch?v=abc123def45"
    assert not any(p.file_data is not None for p in user[1].parts)


def test_files_api_disabled_too_large_notifies(tmp_path):
    # criterion 6
    pipe = make_pipe(GEMINI_API_KEY="k", FILES_API="never", INLINE_FILE_MAX_MB=1)
    mod = pipe_module(pipe)
    video = b"v" * (2 * 1024 * 1024)
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": "f4",
                "filename": "big.mp4",
                "path": "files/big.mp4",
                "hash": "h4",
                "content_type": "video/mp4",
                "data": video,
            }
        ],
    )
    files = [
        {
            "type": "file",
            "id": "f4",
            "name": "big.mp4",
            "file": {"meta": {"content_type": "video/mp4"}},
        }
    ]
    messages = [{"role": "user", "content": "Summarise"}]
    fake = FakeClient(scripts=[answer_script()])
    result, emitter = run_pipe(pipe, fake, messages=messages, files=files)

    parts = [p for c in user_contents(fake) for p in c.parts]
    assert not any(p.inline_data is not None for p in parts)
    assert not any(p.file_data is not None for p in parts)
    assert not fake.upload_calls
    notifications = emitter.of_type("notification")
    assert any("too large" in e["data"]["content"] for e in notifications)


def test_native_files_off_keeps_rag(tmp_path):
    # criterion 7
    pipe = make_pipe(GEMINI_API_KEY="k")
    mod = pipe_module(pipe)
    audio = b"m" * 1024
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": "f1",
                "filename": "memo.mp3",
                "path": "files/memo.mp3",
                "hash": "h1",
                "content_type": "audio/mpeg",
                "data": audio,
            }
        ],
    )
    rag = (
        "### Task: Summarise\n<context>\n"
        '<source id="1" name="memo.mp3">transcript</source>\n'
        "</context>"
    )
    messages = [
        {"role": "system", "content": rag},
        {"role": "user", "content": "Summarise"},
    ]
    files = [
        {
            "type": "file",
            "id": "f1",
            "name": "memo.mp3",
            "file": {"meta": {"content_type": "audio/mpeg"}},
        }
    ]
    fake = FakeClient(scripts=[answer_script()])
    result, emitter = run_pipe(
        pipe,
        fake,
        messages=messages,
        files=files,
        user=make_user(pipe, USE_NATIVE_FILES=False),
    )
    parts = [p for c in user_contents(fake) for p in c.parts]
    assert not any(p.inline_data is not None for p in parts)
    system_instruction = fake.stream_calls[0]["config"].system_instruction or ""
    assert "memo.mp3" in system_instruction
