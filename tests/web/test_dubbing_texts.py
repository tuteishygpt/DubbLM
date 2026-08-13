from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from dubbing.web.dubbing_texts import (
    DubbingTextConflictError,
    DubbingTextContext,
    DubbingTextService,
    DubbingTextValidationError,
    DubbingTextWriteError,
)


@dataclass
class StoredMedia:
    id: str
    name: str
    url: str
    path: Path


class FakeMediaStore:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], StoredMedia] = {}
        self.registered: list[tuple[str, str, str | None]] = []

    def register(
        self, *, owner_id: str, path: str | Path, name: str, kind: str,
        job_id: str | None = None,
    ) -> StoredMedia:
        assert kind == "dubbing_segment"
        key = (owner_id, str(Path(path).resolve()))
        media_id = f"audio-{len(self.records) + 1}"
        record = self.records.setdefault(
            key,
            StoredMedia(media_id, name, f"/jobs/{job_id}/files/{media_id}", Path(path)),
        )
        self.registered.append((owner_id, str(Path(path)), job_id))
        return record


class ConcreteShapeMediaStore(FakeMediaStore):
    def register(
        self, owner_id: str, source_path: str | Path | None = None, *,
        path: str | Path | None = None, name: str | None = None,
        kind: str = "result",
    ) -> object:
        source = Path(source_path or path or "")
        record = super().register(
            owner_id=owner_id, path=source, name=name or source.name, kind=kind,
        )
        return type(
            "MediaRecord",
            (),
            {"id": record.id, "name": record.name, "path": record.path},
        )()


class FakeDubber:
    def __init__(self, config: object) -> None:
        self.config = config
        self.calls: list[tuple[int, str | None]] = []

    def resynthesize_one_segment(
        self, segments: list[dict[str, object]], segment_index: int,
        override_text: str | None = None,
    ) -> dict[str, object]:
        self.calls.append((segment_index, override_text))
        segment = segments[segment_index]
        output = Path(str(self.config["generated_audio"]))
        output.write_bytes(b"new audio")
        segment["synthesized_speech_file"] = str(output)
        segment["synthesized_text"] = override_text
        return segment


@dataclass
class Fixture:
    service: DubbingTextService
    context: DubbingTextContext
    store: FakeMediaStore
    dubbers: list[FakeDubber]
    config: dict[str, object]


@pytest.fixture
def text_fixture(tmp_path: Path) -> Fixture:
    cache_path = tmp_path / "cache" / "translation" / "key.pkl"
    snapshot_path = tmp_path / "cache" / "dubbing_texts" / "snapshot.pkl"
    artifact_path = tmp_path / "prj" / "video" / "artifacts" / "dubbing_texts.tsv"
    audio_path = tmp_path / "prj" / "video" / "artifacts" / "audio" / "source.wav"
    transcription_path = tmp_path / "prj" / "video" / "artifacts" / "transcription.txt"
    generated_audio = tmp_path / "prj" / "video" / "artifacts" / "audio_chunks" / "0.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"source")
    generated_audio.parent.mkdir(parents=True)
    config: dict[str, object] = {
        "transcription_path": str(transcription_path),
        "isolated_tracks": None,
        "semantic_split_enabled": True,
        "generated_audio": str(generated_audio),
    }
    context = DubbingTextContext(
        config=config,
        cache_path=cache_path,
        snapshot_path=snapshot_path,
        artifact_path=artifact_path,
        audio_path=audio_path,
    )
    store = FakeMediaStore()
    dubbers: list[FakeDubber] = []

    def make_dubber(value: object) -> FakeDubber:
        dubber = FakeDubber(value)
        dubbers.append(dubber)
        return dubber

    service = DubbingTextService(
        store,
        context_builder=lambda _config: context,
        dubber_factory=make_dubber,
    )
    return Fixture(service, context, store, dubbers, config)


def _segment(**updates: object) -> dict[str, object]:
    segment: dict[str, object] = {
        "speaker": "SPEAKER_00",
        "start": 1.234,
        "end": 2.345,
        "text": "Original",
        "translation": "Translated",
        "short_translation": "Translated",
        "very_short_translation": "Translated",
        "long_translation": "Translated",
        "synthesized_text": "Spoken",
        "style_prompt": "calm",
    }
    segment.update(updates)
    return segment


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))


def test_load_prefers_translation_cache_to_transcription_and_persists_segment_id(
    text_fixture: Fixture,
) -> None:
    fixture = text_fixture
    _write_pickle(fixture.context.cache_path, [_segment(translation="from cache")])
    Path(str(fixture.config["transcription_path"])).write_text(
        "[00.00.00.111-00.00.01.222] SPEAKER_01: from transcription\n",
        encoding="utf-8",
    )

    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)
    reloaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)

    assert loaded.source == "translation"
    assert loaded.segments[0].translation == "from cache"
    assert uuid.UUID(loaded.segments[0].segment_id).version == 4
    assert reloaded.segments[0].segment_id == loaded.segments[0].segment_id
    assert reloaded.revision == loaded.revision


def test_load_prefers_latest_snapshot_and_preserves_its_cache_key(text_fixture: Fixture) -> None:
    fixture = text_fixture
    alternate_cache = fixture.context.cache_path.parent / "active-key.pkl"
    _write_pickle(fixture.context.cache_path, [_segment(translation="old cache")])
    _write_pickle(alternate_cache, [_segment(translation="active cache")])
    _write_pickle(
        fixture.context.snapshot_path,
        {
            "version": 1,
            "segments": [_segment(translation="latest snapshot")],
            "translation_cache_reusable": True,
            "translation_cache_key": "active-key",
        },
    )

    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)

    assert loaded.source == "snapshot"
    assert loaded.segments[0].translation == "latest snapshot"
    payload = pickle.loads(fixture.context.snapshot_path.read_bytes())
    assert payload["translation_cache_key"] == "active-key"


def test_load_seeds_transcription_with_millisecond_precision(text_fixture: Fixture) -> None:
    fixture = text_fixture
    Path(str(fixture.config["transcription_path"])).write_text(
        "[00.00.01.007-00.00.02.345] SPEAKER_07: precise words\n",
        encoding="utf-8",
    )

    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)

    segment = loaded.segments[0]
    assert loaded.source == "transcription"
    assert segment.start == 1.007
    assert segment.end == 2.345
    assert segment.text == "precise words"
    assert segment.translation == "precise words"


def test_load_registers_synthesized_audio_without_returning_a_path(text_fixture: Fixture) -> None:
    fixture = text_fixture
    generated = Path(str(fixture.config["generated_audio"]))
    generated.write_bytes(b"audio")
    _write_pickle(
        fixture.context.cache_path,
        [_segment(synthesized_speech_file=str(generated))],
    )

    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)

    audio = loaded.segments[0].audio
    assert audio is not None
    assert audio.id == "audio-1"
    assert audio.url == "/jobs/job-1/files/audio-1"
    assert "path" not in vars(audio)
    assert str(generated) not in repr(loaded)


def test_load_supports_concrete_store_record_without_job_or_url(tmp_path: Path) -> None:
    cache = tmp_path / "translation.pkl"
    snapshot = tmp_path / "snapshot.pkl"
    artifact = tmp_path / "dubbing_texts.tsv"
    source = tmp_path / "source.wav"
    generated = tmp_path / "0.wav"
    source.write_bytes(b"source")
    generated.write_bytes(b"audio")
    _write_pickle(cache, [_segment(synthesized_speech_file=str(generated))])
    context = DubbingTextContext({}, cache, snapshot, artifact, source)
    service = DubbingTextService(
        ConcreteShapeMediaStore(), context_builder=lambda _config: context,
        dubber_factory=lambda config: FakeDubber(config),
    )

    loaded = service.load(owner_id="alice", job_id="job-1", config={})

    assert loaded.segments[0].audio is not None
    assert loaded.segments[0].audio.url == "/api/jobs/job-1/files/audio-1"


def test_save_rejects_row_count_mismatch(text_fixture: Fixture) -> None:
    fixture = text_fixture
    _write_pickle(fixture.context.cache_path, [_segment(), _segment(speaker="SPEAKER_01")])
    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)

    with pytest.raises(DubbingTextValidationError, match="row count"):
        fixture.service.save(
            owner_id="alice", job_id="job-1", config=fixture.config,
            segments=loaded.segments[:1], revision=loaded.revision,
        )


def test_save_rejects_stale_revision_and_updates_tts_translation_cache(
    text_fixture: Fixture,
) -> None:
    fixture = text_fixture
    _write_pickle(fixture.context.cache_path, [_segment()])
    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)
    edited = replace(
        loaded.segments[0], speaker="SPEAKER_09", start=1.111, end=3.999,
        translation="Edited translation",
    )
    saved = fixture.service.save(
        owner_id="alice", job_id="job-1", config=fixture.config,
        segments=[edited], revision=loaded.revision,
    )

    with pytest.raises(DubbingTextConflictError):
        fixture.service.save(
            owner_id="alice", job_id="job-1", config=fixture.config,
            segments=[edited], revision=loaded.revision,
        )

    cached = pickle.loads(fixture.context.cache_path.read_bytes())
    assert cached[0]["translation"] == "Edited translation"
    assert cached[0]["short_translation"] == "Edited translation"
    assert saved.segments[0].segment_id == loaded.segments[0].segment_id
    assert saved.segments[0].start == 1.111
    assert saved.segments[0].end == 3.999


def test_save_replace_failure_preserves_cache_snapshot_and_tsv(
    text_fixture: Fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = text_fixture
    _write_pickle(fixture.context.cache_path, [_segment()])
    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)
    fixture.context.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.context.artifact_path.write_text("old tsv", encoding="utf-8")
    previous = {
        fixture.context.cache_path: fixture.context.cache_path.read_bytes(),
        fixture.context.snapshot_path: fixture.context.snapshot_path.read_bytes(),
        fixture.context.artifact_path: fixture.context.artifact_path.read_bytes(),
    }
    edited = replace(loaded.segments[0], translation="new")
    monkeypatch.setattr(
        "dubbing.web.dubbing_texts.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(DubbingTextWriteError, match="replace failed"):
        fixture.service.save(
            owner_id="alice", job_id="job-1", config=fixture.config,
            segments=[edited], revision=loaded.revision,
        )

    assert {path: path.read_bytes() for path in previous} == previous


def test_regenerate_targets_segment_id_registers_audio_and_returns_new_revision(
    text_fixture: Fixture,
) -> None:
    fixture = text_fixture
    _write_pickle(fixture.context.cache_path, [_segment()])
    loaded = fixture.service.load(owner_id="alice", job_id="job-1", config=fixture.config)

    result = fixture.service.regenerate(
        owner_id="alice",
        job_id="job-1",
        config=fixture.config,
        segment_id=loaded.segments[0].segment_id,
        revision=loaded.revision,
        synthesized_text="Fresh words",
    )

    assert fixture.dubbers[0].calls == [(0, "Fresh words")]
    assert result.revision != loaded.revision
    assert result.segment.segment_id == loaded.segments[0].segment_id
    assert result.segment.synthesized_text == "Fresh words"
    assert result.segment.audio is not None
    assert result.segment.audio.url.startswith("/jobs/job-1/files/")
    assert str(fixture.config["generated_audio"]) not in repr(result)
