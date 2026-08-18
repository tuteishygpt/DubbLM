"""Safe, framework-independent media persistence."""

from io import BytesIO
from pathlib import Path

import pytest

from dubbing.web.contracts import MediaStore
from dubbing.web.storage import (
    DEFAULT_MAX_UPLOAD_BYTES,
    FileMediaStore,
    MediaConflictError,
    MediaNotFoundError,
    MediaValidationError,
    MediaWriteError,
)


class _MemoryMediaStore:
    """A non-filesystem adapter used to prove the consumer boundary."""

    def save(self, *args, **kwargs):
        return "saved"

    def get(self, *args, **kwargs):
        return "found"

    def delete(self, *args, **kwargs):
        return None

    def register(self, *args, **kwargs):
        return "registered"

    def release_job_materialization(self, *args, **kwargs):
        return None


def _consumer(store: MediaStore) -> tuple[object, object, object, object]:
    return (
        store.save("owner", "clip.mp4", BytesIO(b"video")),
        store.get("owner", "media-id"),
        store.delete("owner", "media-id"),
        store.register("owner", Path("result.srt")),
    )


def test_media_consumer_depends_only_on_replaceable_protocol():
    assert _consumer(_MemoryMediaStore()) == ("saved", "found", None, "registered")


def test_default_upload_limit_is_twenty_gibibytes():
    assert DEFAULT_MAX_UPLOAD_BYTES == 20 * 1024**3


@pytest.mark.parametrize(
    ("filename", "kind"),
    [
        ("clip.mp4", "video"),
        ("clip.mov", "video"),
        ("clip.mkv", "video"),
        ("clip.webm", "video"),
        ("clip.avi", "video"),
        ("voice.wav", "audio"),
        ("voice.mp3", "audio"),
        ("voice.m4a", "audio"),
        ("voice.flac", "audio"),
        ("voice.ogg", "audio"),
    ],
)
def test_save_streams_each_supported_media_extension(tmp_path, filename, kind):
    probed: list[tuple[str, str]] = []

    def probe(path: Path, media_kind: str) -> bool:
        probed.append((path.suffix, media_kind))
        return path.read_bytes() == b"payload"

    store = FileMediaStore(tmp_path, probe=probe)
    saved = store.save("alice", filename, BytesIO(b"payload"), chunk_size=2)

    assert saved.name == filename
    assert saved.kind == kind
    assert saved.size == 7
    assert saved.id not in filename
    assert saved.path.is_file()
    assert probed == [(Path(filename).suffix, kind)]


def test_save_rejects_disallowed_extension_before_writing(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)

    with pytest.raises(MediaValidationError, match="extension"):
        store.save("alice", "notes.txt", BytesIO(b"payload"))

    assert list(tmp_path.rglob("*")) == []


def test_save_rejects_failed_probe_and_removes_partial_media(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: False)

    with pytest.raises(MediaValidationError, match="media"):
        store.save("alice", "fake.mp4", BytesIO(b"not video"))

    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_save_stops_stream_at_limit_and_removes_partial_media(tmp_path):
    stream = BytesIO(b"too-large")
    store = FileMediaStore(tmp_path, max_upload_bytes=4, probe=lambda *_: True)

    with pytest.raises(MediaValidationError, match="size"):
        store.save("alice", "clip.mp4", stream, chunk_size=3)

    assert stream.tell() == 6
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_get_masks_other_owners_and_rejects_traversal(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    saved = store.save("alice", "clip.mp4", BytesIO(b"video"))

    with pytest.raises(MediaNotFoundError):
        store.get("bob", saved.id)
    with pytest.raises(MediaValidationError):
        store.get("../alice", saved.id)
    with pytest.raises(MediaValidationError):
        store.get("alice", "../metadata.json")


def test_register_copies_only_an_explicit_file_into_managed_storage(tmp_path):
    generated = tmp_path / "pipeline" / "captions.srt"
    generated.parent.mkdir()
    generated.write_text("captions", encoding="utf-8")
    store = FileMediaStore(tmp_path / "server")

    registered = store.register("alice", generated, kind="subtitle")
    generated.unlink()
    looked_up = store.get("alice", registered.id)

    assert looked_up.registered is True
    assert looked_up.name == "captions.srt"
    assert looked_up.kind == "subtitle"
    assert looked_up.path.read_text(encoding="utf-8") == "captions"
    with pytest.raises(MediaNotFoundError):
        store.get("bob", registered.id)


def test_reference_service_keyword_aliases_still_use_safe_managed_storage(tmp_path):
    source = tmp_path / "legacy.wav"
    source.write_bytes(b"legacy")
    probed: list[str] = []
    store = FileMediaStore(
        tmp_path / "server",
        probe=lambda _path, kind: probed.append(kind) is None,
    )

    saved = store.save(
        owner_id="alice",
        source=BytesIO(b"reference"),
        name="voice.wav",
        kind="reference",
    )
    registered = store.register(
        owner_id="alice",
        path=source,
        name="legacy.wav",
        kind="reference",
    )

    assert saved.kind == "reference"
    assert saved.path.read_bytes() == b"reference"
    assert registered.path.read_bytes() == b"legacy"
    assert probed == ["audio"]


def test_delete_rejects_a_materialized_upload_referenced_by_a_job(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    saved = store.save("alice", "clip.mp4", BytesIO(b"video"))
    store.materialize_for_job("alice", saved.id, "f5c1481f-47b1-440a-a08d-c67ce33f31d8")

    with pytest.raises(MediaConflictError, match="referenced"):
        store.delete("alice", saved.id)

    assert store.get("alice", saved.id).id == saved.id


def test_delete_removes_an_unreferenced_upload_and_masks_repeat_delete(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    saved = store.save("alice", "clip.mp4", BytesIO(b"video"))

    store.delete("alice", saved.id)

    with pytest.raises(MediaNotFoundError):
        store.get("alice", saved.id)


def test_job_materialization_preserves_sanitized_original_basename_and_stem(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    saved = store.save("alice", "../My unsafe clip (final).MP4", BytesIO(b"video"))

    materialized = store.materialize_for_job(
        "alice", saved.id, "f5c1481f-47b1-440a-a08d-c67ce33f31d8"
    )

    assert materialized.basename == "My_unsafe_clip_final.MP4"
    assert materialized.stem == "My_unsafe_clip_final"
    assert materialized.path.name == materialized.basename
    assert materialized.path.read_bytes() == b"video"
    assert materialized.path.resolve().is_relative_to(tmp_path.resolve())


def test_save_preserves_unicode_and_cyrillic_audio_filenames(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    saved = store.save("alice", "жыгамонт_цмокі.mp3", BytesIO(b"audio"))

    assert saved.name == "жыгамонт_цмокі.mp3"
    assert saved.path.name == "жыгамонт_цмокі.mp3"
    assert saved.path.is_file()

    retrieved = store.get("alice", saved.id)
    assert retrieved.name == "жыгамонт_цмокі.mp3"
    assert retrieved.path.name == "жыгамонт_цмокі.mp3"


def test_materialization_rejects_traversal_job_reference(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    saved = store.save("alice", "clip.mp4", BytesIO(b"video"))

    with pytest.raises(MediaValidationError):
        store.materialize_for_job("alice", saved.id, "../escape")


def test_failed_materialization_rolls_back_copy_and_retry_records_reference(tmp_path, monkeypatch):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    saved = store.save("alice", "clip.mp4", BytesIO(b"video"))
    job_id = "f5c1481f-47b1-440a-a08d-c67ce33f31d8"
    original = store._atomic_json
    failed = False

    def fail_media_reference(path, value):
        nonlocal failed
        if path.name == "metadata.json" and value.get("references") and not failed:
            failed = True
            raise MediaWriteError("disk error")
        original(path, value)

    monkeypatch.setattr(store, "_atomic_json", fail_media_reference)

    with pytest.raises(MediaWriteError):
        store.materialize_for_job("alice", saved.id, job_id)
    materialized = store.materialize_for_job("alice", saved.id, job_id)

    assert materialized.path.read_bytes() == b"video"
    with pytest.raises(MediaConflictError):
        store.delete("alice", saved.id)


def test_release_job_materialization_removes_copy_and_upload_reference(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    first = store.save("alice", "clip.mp4", BytesIO(b"video"))
    second = store.save("alice", "voice.wav", BytesIO(b"audio"))
    job_id = "f5c1481f-47b1-440a-a08d-c67ce33f31d8"
    materialized = [
        store.materialize_for_job("alice", media.id, job_id)
        for media in (first, second)
    ]

    store.release_job_materialization("alice", job_id, [first.id, second.id])

    assert not any(item.path.exists() for item in materialized)
    store.delete("alice", first.id)
    store.delete("alice", second.id)


def test_release_job_materialization_is_safe_for_partially_materialized_set(tmp_path):
    store = FileMediaStore(tmp_path, probe=lambda *_: True)
    first = store.save("alice", "clip.mp4", BytesIO(b"video"))
    second = store.save("alice", "voice.wav", BytesIO(b"audio"))
    job_id = "f5c1481f-47b1-440a-a08d-c67ce33f31d8"
    store.materialize_for_job("alice", first.id, job_id)

    store.release_job_materialization("alice", job_id, [first.id, second.id])

    store.delete("alice", first.id)
    store.delete("alice", second.id)
