from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from dubbing.web.references import (
    ReferenceConflictError,
    ReferenceLibraryService,
    ReferenceValidationError,
    ReferenceWriteError,
)


@dataclass
class StoredMedia:
    id: str
    name: str
    url: str
    path: Path


class FakeMediaStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[tuple[str, str], StoredMedia] = {}
        self.deleted: list[tuple[str, str]] = []
        self.next_id = 1
        self.delete_error: Exception | None = None

    def save(self, *, owner_id: str, source: object, name: str, kind: str) -> StoredMedia:
        assert kind == "reference"
        media_id = f"media-{self.next_id}"
        self.next_id += 1
        path = self.root / owner_id / media_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, (str, Path)):
            path.write_bytes(Path(source).read_bytes())
        else:
            path.write_bytes(source.read())
        record = StoredMedia(media_id, name, f"/media/{media_id}", path)
        self.records[(owner_id, media_id)] = record
        return record

    def register(self, *, owner_id: str, path: str | Path, name: str, kind: str) -> StoredMedia:
        assert kind == "reference"
        media_id = f"legacy-{self.next_id}"
        self.next_id += 1
        record = StoredMedia(media_id, name, f"/media/{media_id}", Path(path))
        self.records[(owner_id, media_id)] = record
        return record

    def get(self, *, owner_id: str, media_id: str) -> StoredMedia:
        return self.records[(owner_id, media_id)]

    def delete(self, *, owner_id: str, media_id: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append((owner_id, media_id))
        self.records.pop((owner_id, media_id), None)


class ProcessMediaStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, *, owner_id: str, source: object, name: str, kind: str) -> StoredMedia:
        time.sleep(0.4)
        media_id = f"media-{Path(str(source)).stem}"
        path = self.root / owner_id / media_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(str(source)).read_bytes())
        return StoredMedia(media_id, name, f"/media/{media_id}", path)

    def get(self, *, owner_id: str, media_id: str) -> StoredMedia:
        path = next((self.root / owner_id / media_id).iterdir())
        return StoredMedia(media_id, path.name, f"/media/{media_id}", path)

    def delete(self, *, owner_id: str, media_id: str) -> None:
        return None


def _concurrent_reference_save(
    library: str, media: str, source: str, speaker: str,
    revision: str, start: object, results: object,
) -> None:
    service = ReferenceLibraryService(Path(library), ProcessMediaStore(Path(media)))
    start.wait()
    try:
        service.save(
            owner_id="alice", speaker_id=speaker, source_audio=source,
            reference_text=speaker, revision=revision,
        )
    except Exception as exc:
        results.put(type(exc).__name__)
    else:
        results.put("success")


class FakeSettings:
    def __init__(self) -> None:
        self.received: dict[str, object] = {}

    def assign_reference(self, speaker_id: str, **kwargs: object) -> object:
        self.received = {"speaker_id": speaker_id, **kwargs}
        return type("Snapshot", (), {"revision": "settings-next"})()


def test_reference_save_lists_registered_audio_without_exposing_paths(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    source.write_bytes(b"first voice")
    store = FakeMediaStore(tmp_path / "media")
    service = ReferenceLibraryService(tmp_path / "library", store)

    initial = service.list(owner_id="alice")
    saved = service.save(
        owner_id="alice",
        speaker_id="Warm Narrator",
        source_audio=source,
        reference_text="  Welcome home.  ",
        revision=initial.revision,
    )

    assert saved.revision != initial.revision
    assert len(saved.entries) == 1
    entry = saved.entries[0]
    assert entry.speaker_id == "Warm Narrator"
    assert entry.reference_text == "Welcome home."
    assert entry.audio.id == "media-1"
    assert entry.audio.url == "/media/media-1"
    assert "path" not in vars(entry.audio)
    assert str(tmp_path) not in repr(saved)


def test_reference_update_rejects_stale_revision_and_preserves_current_entry(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    store = FakeMediaStore(tmp_path / "media")
    service = ReferenceLibraryService(tmp_path / "library", store)
    empty_revision = service.list(owner_id="alice").revision
    current = service.save(
        owner_id="alice", speaker_id="Guide", source_audio=first,
        reference_text="first", revision=empty_revision,
    )

    with pytest.raises(ReferenceConflictError):
        service.save(
            owner_id="alice", speaker_id="Guide", source_audio=second,
            reference_text="second", revision=empty_revision,
        )

    assert service.list(owner_id="alice") == current
    assert store.next_id == 2


def test_reference_metadata_replace_failure_preserves_previous_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    store = FakeMediaStore(tmp_path / "media")
    service = ReferenceLibraryService(tmp_path / "library", store)
    current = service.save(
        owner_id="alice", speaker_id="Guide", source_audio=first,
        reference_text="first", revision=service.list(owner_id="alice").revision,
    )
    metadata_path = next((tmp_path / "library" / "alice").glob("*/meta.yml"))
    previous = metadata_path.read_bytes()

    monkeypatch.setattr("dubbing.web.references.os.replace", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(ReferenceWriteError, match="disk full"):
        service.save(
            owner_id="alice", speaker_id="Guide", source_audio=second,
            reference_text="second", revision=current.revision,
        )

    assert metadata_path.read_bytes() == previous
    assert ("alice", "media-2") in store.deleted


def test_reference_delete_is_revisioned_and_owner_scoped(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    source.write_bytes(b"voice")
    store = FakeMediaStore(tmp_path / "media")
    service = ReferenceLibraryService(tmp_path / "library", store)
    alice = service.save(
        owner_id="alice", speaker_id="Guide", source_audio=source,
        reference_text="alice", revision=service.list(owner_id="alice").revision,
    )
    service.save(
        owner_id="bob", speaker_id="Guide", source_audio=source,
        reference_text="bob", revision=service.list(owner_id="bob").revision,
    )

    deleted = service.delete(owner_id="alice", speaker_id="Guide", revision=alice.revision)

    assert deleted.entries == ()
    assert [entry.reference_text for entry in service.list(owner_id="bob").entries] == ["bob"]
    assert ("alice", "media-1") in store.deleted


def test_legacy_metadata_path_is_registered_and_assignment_uses_internal_path(tmp_path: Path) -> None:
    audio = tmp_path / "legacy.wav"
    audio.write_bytes(b"legacy")
    speaker_dir = tmp_path / "library" / "alice" / "Legacy_Label"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "meta.yml").write_text(
        yaml.safe_dump(
            {
                "speaker_id": "Legacy Label",
                "reference_audio_path": str(audio),
                "reference_text": "old words",
            }
        ),
        encoding="utf-8",
    )
    store = FakeMediaStore(tmp_path / "media")
    settings = FakeSettings()
    service = ReferenceLibraryService(tmp_path / "library", store)

    snapshot = service.list(owner_id="alice")
    assignment = service.assign(
        owner_id="alice",
        library_speaker_id="Legacy Label",
        profile_speaker_id="SPEAKER_00",
        settings=settings,
        settings_revision="settings-old",
    )

    assert snapshot.entries[0].audio.id.startswith("legacy-")
    assert settings.received == {
        "speaker_id": "SPEAKER_00",
        "reference_audio": str(audio),
        "reference_text": "old words",
        "revision": "settings-old",
    }
    assert assignment.settings_revision == "settings-next"
    assert assignment.audio_id == snapshot.entries[0].audio.id
    assert str(audio) not in repr(assignment)


def test_reference_record_without_url_gets_public_audio_route(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    source.write_bytes(b"voice")
    store = FakeMediaStore(tmp_path / "media")
    service = ReferenceLibraryService(tmp_path / "library", store)
    saved = service.save(
        owner_id="alice", speaker_id="Warm Guide", source_audio=source,
        reference_text="words", revision=service.list(owner_id="alice").revision,
    )
    store.records[("alice", saved.entries[0].audio.id)].url = ""

    loaded = service.list(owner_id="alice")

    assert loaded.entries[0].audio.url == "/api/reference-library/Warm%20Guide/audio"


def test_reference_directory_identity_is_collision_free_for_display_labels(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    source.write_bytes(b"voice")
    store = FakeMediaStore(tmp_path / "media")
    service = ReferenceLibraryService(tmp_path / "library", store)
    first = service.save(
        owner_id="alice", speaker_id="a/b", source_audio=source,
        reference_text="slash", revision=service.list(owner_id="alice").revision,
    )
    second = service.save(
        owner_id="alice", speaker_id="a_b", source_audio=source,
        reference_text="underscore", revision=first.revision,
    )

    assert [(entry.speaker_id, entry.reference_text) for entry in second.entries] == [
        ("a/b", "slash"), ("a_b", "underscore")
    ]
    directories = [path.name for path in (tmp_path / "library").glob("*/*") if path.is_dir()]
    assert len(set(directories)) == 2


@pytest.mark.parametrize("owner", ["../alice", "a/b", "..", " alice"])
def test_reference_rejects_unsafe_owner_identity(tmp_path: Path, owner: str) -> None:
    service = ReferenceLibraryService(tmp_path / "library", FakeMediaStore(tmp_path / "media"))

    with pytest.raises(ReferenceValidationError, match="owner"):
        service.list(owner_id=owner)


def test_reference_delete_restores_metadata_when_media_delete_fails(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    source.write_bytes(b"voice")
    store = FakeMediaStore(tmp_path / "media")
    service = ReferenceLibraryService(tmp_path / "library", store)
    saved = service.save(
        owner_id="alice", speaker_id="Guide", source_audio=source,
        reference_text="words", revision=service.list(owner_id="alice").revision,
    )
    store.delete_error = OSError("media unavailable")

    with pytest.raises(ReferenceWriteError, match="media unavailable"):
        service.delete(owner_id="alice", speaker_id="Guide", revision=saved.revision)

    assert service.list(owner_id="alice").entries[0].speaker_id == "Guide"


def test_same_reference_revision_cannot_succeed_in_two_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    library = tmp_path / "library"
    media = tmp_path / "media"
    source_one = tmp_path / "one.wav"
    source_two = tmp_path / "two.wav"
    source_one.write_bytes(b"one")
    source_two.write_bytes(b"two")
    revision = ReferenceLibraryService(library, ProcessMediaStore(media)).list(
        owner_id="alice"
    ).revision
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_reference_save,
            args=(str(library), str(media), str(source), speaker, revision, start, results),
        )
        for source, speaker in ((source_one, "ONE"), (source_two, "TWO"))
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=10) for _process in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert outcomes.count("success") == 1
    assert outcomes.count("ReferenceConflictError") == 1
