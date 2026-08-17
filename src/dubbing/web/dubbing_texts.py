"""Framework-independent loading, editing, and regeneration of dubbing text.

Pipeline cache dictionaries stay private to this module.  Public results are
typed and expose synthesized audio only through registered media references.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import os
import pickle
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote

from .contracts import MediaStore

try:  # Windows locking backend
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None

try:  # POSIX locking backend
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None


TRANSLATION_FIELDS = (
    "translation",
    "short_translation",
    "very_short_translation",
    "long_translation",
)


class DubbingTextError(Exception):
    """Base class for dubbing-text domain failures."""


class DubbingTextConflictError(DubbingTextError):
    """A write was based on an obsolete content revision."""


class DubbingTextValidationError(DubbingTextError):
    """A requested edit cannot be represented by pipeline segment data."""


class DubbingTextNotFoundError(DubbingTextError):
    """The project or requested segment does not exist."""


class DubbingTextWriteError(DubbingTextError):
    """One or more persisted text artifacts could not be replaced."""


@dataclass(frozen=True)
class DubbingTextContext:
    """Resolved paths for the existing per-video project and cache layout."""

    config: object
    cache_path: Path
    snapshot_path: Path
    artifact_path: Path
    audio_path: Path


@dataclass(frozen=True)
class SegmentAudio:
    id: str
    name: str
    url: str


@dataclass(frozen=True)
class DubbingTextSegment:
    segment_id: str
    speaker: str
    start: float
    end: float
    text: str
    translation: str
    synthesized_text: str
    style_prompt: str
    audio: SegmentAudio | None = None


@dataclass(frozen=True)
class DubbingTextSnapshot:
    revision: str
    segments: tuple[DubbingTextSegment, ...]
    source: str


@dataclass(frozen=True)
class DubbingTextRegeneration:
    revision: str
    segment: DubbingTextSegment


@dataclass
class _LoadedState:
    context: DubbingTextContext
    segments: list[dict[str, Any]]
    source: str
    reusable: bool
    active_cache_path: Path
    revision: str


class DubbingTextService:
    """Edit existing per-video translation artifacts with revision checks."""

    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(
        self,
        media_store: MediaStore,
        *,
        job_repository: object | None = None,
        context_builder: Callable[[object], DubbingTextContext] | None = None,
        dubber_factory: Callable[[object], object] | None = None,
    ) -> None:
        self._media_store = media_store
        self._job_repository = job_repository
        self._context_builder = context_builder or self._build_context
        self._dubber_factory = dubber_factory or self._build_dubber

    def load(
        self, *, owner_id: str, job_id: str, config: object
    ) -> DubbingTextSnapshot:
        owner = self._validate_identity(owner_id, "Owner ID")
        job = self._validate_identity(job_id, "Job ID")
        context = self._context_builder(config)
        with self._lock_for(context.snapshot_path):
            with self._interprocess_lock(context.snapshot_path):
                state = self._load_state(context, persist_ids=True)
                self._ensure_audio_refs(owner, job, state)
                return self._public_snapshot(owner, job, state)

    def save(
        self,
        *,
        owner_id: str,
        job_id: str,
        config: object,
        segments: Sequence[DubbingTextSegment],
        revision: str,
    ) -> DubbingTextSnapshot:
        owner = self._validate_identity(owner_id, "Owner ID")
        job = self._validate_identity(job_id, "Job ID")
        context = self._context_builder(config)
        with self._lock_for(context.snapshot_path):
            with self._interprocess_lock(context.snapshot_path):
                state = self._load_state(context, persist_ids=True)
                self._ensure_audio_refs(owner, job, state)
                self._ensure_revision(revision, state.revision)
                self._apply_edits(state.segments, segments)
                state.revision = self._persist_state(state)
                state.source = "snapshot"
                return self._public_snapshot(owner, job, state)

    def regenerate(
        self,
        *,
        owner_id: str,
        job_id: str,
        config: object,
        segment_id: str,
        revision: str,
        synthesized_text: str | None = None,
    ) -> DubbingTextRegeneration:
        owner = self._validate_identity(owner_id, "Owner ID")
        job = self._validate_identity(job_id, "Job ID")
        requested_id = self._validate_identity(segment_id, "Segment ID")
        context = self._context_builder(config)
        with self._lock_for(context.snapshot_path):
            with self._interprocess_lock(context.snapshot_path):
                state = self._load_state(context, persist_ids=True)
                self._ensure_audio_refs(owner, job, state)
                self._ensure_revision(revision, state.revision)
                row_index = next(
                    (
                        index
                        for index, segment in enumerate(state.segments)
                        if str(segment.get("segment_id") or "") == requested_id
                    ),
                    -1,
                )
                if row_index < 0:
                    raise DubbingTextNotFoundError(f"Dubbing segment not found: {requested_id}.")
                segment = state.segments[row_index]
                override = (
                    str(synthesized_text).strip()
                    if synthesized_text is not None
                    else str(segment.get("synthesized_text") or segment.get("translation") or "").strip()
                )
                if not override:
                    raise DubbingTextValidationError("Cannot regenerate a segment with empty text.")
                previous_segment = copy.deepcopy(segment)
                previous_path_value = str(
                    segment.get("synthesized_speech_file") or ""
                ).strip()
                previous_path = Path(previous_path_value) if previous_path_value else None
                previous_audio = (
                    previous_path.read_bytes()
                    if previous_path is not None and previous_path.is_file()
                    else None
                )
                dubber = self._dubber_factory(context.config)
                try:
                    dubber.resynthesize_one_segment(
                        segments=state.segments,
                        segment_index=row_index,
                        override_text=override,
                    )
                    state.segments[row_index].pop("synthesized_audio_ref", None)
                    state.revision = self._persist_state(state)
                    self._ensure_audio_refs(owner, job, state)
                except Exception as exc:
                    rollback_error = self._restore_regenerated_audio(
                        state.segments[row_index], previous_path, previous_audio
                    )
                    state.segments[row_index] = previous_segment
                    if rollback_error is not None:
                        raise DubbingTextWriteError(
                            f"Regeneration failed ({exc}); audio rollback failed: {rollback_error}"
                        ) from exc
                    raise
                public = self._public_segment(owner, job, state.segments[row_index])
                return DubbingTextRegeneration(revision=state.revision, segment=public)

    def _load_state(
        self, context: DubbingTextContext, *, persist_ids: bool
    ) -> _LoadedState:
        snapshot_payload: object | None = None
        snapshot_raw = b""
        if context.snapshot_path.is_file():
            snapshot_raw = self._read_bytes(context.snapshot_path)
            snapshot_payload = self._unpickle(snapshot_raw, context.snapshot_path)

        if snapshot_payload is not None:
            if isinstance(snapshot_payload, list):
                segments = self._validate_segment_list(snapshot_payload, context.snapshot_path)
                reusable = False
                active_cache = context.cache_path
            elif isinstance(snapshot_payload, dict) and snapshot_payload.get("version") == 1:
                segments = self._validate_segment_list(
                    snapshot_payload.get("segments"), context.snapshot_path
                )
                reusable = snapshot_payload.get("translation_cache_reusable") is True
                cache_key = snapshot_payload.get("translation_cache_key")
                active_cache = (
                    context.cache_path.parent / f"{cache_key}.pkl"
                    if reusable and isinstance(cache_key, str) and cache_key
                    else context.cache_path
                )
                if reusable and not (isinstance(cache_key, str) and cache_key):
                    reusable = False
            else:
                raise DubbingTextValidationError(
                    f"Unexpected Dubbing Texts snapshot in {context.snapshot_path}."
                )
            source = "snapshot"
        elif context.cache_path.is_file():
            segments = self._validate_segment_list(
                self._unpickle(self._read_bytes(context.cache_path), context.cache_path),
                context.cache_path,
            )
            source = "translation"
            reusable = True
            active_cache = context.cache_path
        elif context.artifact_path.is_file():
            segments = self._seed_from_dubbing_texts_tsv(context.artifact_path)
            self._attach_audio_chunks(segments, context.config)
            source = "dubbing_texts_tsv"
            reusable = True
            active_cache = context.cache_path
        elif self._has_debug_translations(context.config):
            segments = self._seed_from_translations_and_transcription(context.config)
            self._attach_audio_chunks(segments, context.config)
            source = "translations_tsv"
            reusable = True
            active_cache = context.cache_path
        else:
            segments = self._seed_from_transcription(context.config)
            self._attach_audio_chunks(segments, context.config)
            source = "transcription"
            reusable = not (
                self._config_get(context.config, "isolated_tracks")
                and self._config_get(context.config, "semantic_split_enabled", True)
            )
            active_cache = context.cache_path

        assigned_ids = self._ensure_segment_ids(segments)
        state = _LoadedState(
            context=context,
            segments=segments,
            source=source,
            reusable=reusable,
            active_cache_path=active_cache,
            revision=self._revision(snapshot_raw) if snapshot_raw else "",
        )
        if not snapshot_raw or (assigned_ids and persist_ids):
            state.revision = self._persist_snapshot(state)
        return state

    def _public_snapshot(
        self, owner: str, job_id: str, state: _LoadedState
    ) -> DubbingTextSnapshot:
        return DubbingTextSnapshot(
            revision=state.revision,
            segments=tuple(
                self._public_segment(owner, job_id, segment) for segment in state.segments
            ),
            source=state.source,
        )

    def _public_segment(
        self, owner: str, job_id: str, segment: Mapping[str, Any]
    ) -> DubbingTextSegment:
        audio: SegmentAudio | None = None
        media_id = str(segment.get("synthesized_audio_ref") or "").strip()
        if media_id:
            try:
                record = self._media_store.get(owner_id=owner, media_id=media_id)
            except Exception:
                record = None
            if record is not None:
                media_id = self._record_value(record, "id", "media_id")
                url = self._record_value(record, "url")
                if not url:
                    if self._job_repository is None:
                        raise DubbingTextValidationError(
                            "A job repository is required to authorize synthesized audio."
                        )
                    url = (
                        f"/api/jobs/{quote(job_id, safe='')}/files/"
                        f"{quote(media_id, safe='')}"
                    )
                audio = SegmentAudio(
                    id=media_id,
                    name=self._record_value(record, "name") or "segment audio",
                    url=url,
                )
        return DubbingTextSegment(
            segment_id=str(segment.get("segment_id") or ""),
            speaker=str(segment.get("speaker") or ""),
            start=self._parse_seconds(segment.get("start")),
            end=self._parse_seconds(segment.get("end")),
            text=str(segment.get("text") or ""),
            translation=str(segment.get("translation") or ""),
            synthesized_text=str(segment.get("synthesized_text") or ""),
            style_prompt=str(segment.get("style_prompt") or ""),
            audio=audio,
        )

    def _ensure_audio_refs(
        self, owner: str, job_id: str, state: _LoadedState
    ) -> None:
        self._attach_audio_chunks(state.segments, state.context.config)
        changed = False
        records: list[object] = []
        for segment in state.segments:
            audio_path = str(segment.get("synthesized_speech_file") or "").strip()
            if not audio_path or not Path(audio_path).is_file():
                continue
            media_id = str(segment.get("synthesized_audio_ref") or "").strip()
            record = None
            if media_id:
                try:
                    record = self._media_store.get(owner_id=owner, media_id=media_id)
                except Exception:
                    record = None
            if record is None:
                record = self._media_store.register(
                    owner_id=owner,
                    path=audio_path,
                    name=Path(audio_path).name,
                    kind="dubbing_segment",
                )
                media_id = self._record_value(record, "id", "media_id")
                if not media_id:
                    raise DubbingTextValidationError(
                        "MediaStore returned no synthesized-audio identifier."
                    )
                segment["synthesized_audio_ref"] = media_id
                changed = True
            records.append(record)
        if changed:
            state.revision = self._persist_snapshot(state)
        for record in records:
            self._authorize_job_audio(owner, job_id, record)

    def _authorize_job_audio(self, owner: str, job_id: str, record: object) -> None:
        if self._job_repository is None:
            if not self._record_value(record, "url"):
                raise DubbingTextValidationError(
                    "A job repository is required to authorize synthesized audio."
                )
            return
        job = self._job_repository.get(owner, job_id)
        media_id = self._record_value(record, "id", "media_id")
        files = [dict(item) for item in job.files]
        if any(str(item.get("id") or "") == media_id for item in files):
            return
        files.append(
            {
                "id": media_id,
                "name": self._record_value(record, "name") or "segment audio",
                "kind": self._record_value(record, "kind") or "dubbing_segment",
                "size": int(self._record_value(record, "size") or 0),
            }
        )
        self._job_repository.update(owner, job_id, files=files)

    @staticmethod
    def _apply_edits(
        raw_segments: list[dict[str, Any]], edits: Sequence[DubbingTextSegment]
    ) -> None:
        if len(edits) != len(raw_segments):
            raise DubbingTextValidationError(
                f"Edited row count ({len(edits)}) does not match cached segment row count "
                f"({len(raw_segments)})."
            )
        raw_by_id = {
            str(segment.get("segment_id") or ""): segment for segment in raw_segments
        }
        edit_ids = [str(edit.segment_id or "") for edit in edits]
        if len(set(edit_ids)) != len(edit_ids) or set(edit_ids) != set(raw_by_id):
            raise DubbingTextValidationError("Edited segments do not match the persisted segment IDs.")
        for edit in edits:
            translation = str(edit.translation or "").strip()
            if not translation:
                raise DubbingTextValidationError("Translation text cannot be empty.")
            segment = raw_by_id[edit.segment_id]
            segment["speaker"] = str(edit.speaker or "").strip()
            segment["start"] = DubbingTextService._parse_seconds(edit.start)
            segment["end"] = DubbingTextService._parse_seconds(edit.end)
            segment["text"] = str(edit.text or "")
            for field in TRANSLATION_FIELDS:
                segment[field] = translation
            if edit.synthesized_text:
                segment["synthesized_text"] = str(edit.synthesized_text)
            segment["style_prompt"] = str(edit.style_prompt or "")

    def _persist_snapshot(self, state: _LoadedState) -> str:
        content = self._snapshot_bytes(state)
        self._atomic_group_replace({state.context.snapshot_path: content})
        return self._revision(content)

    def _persist_state(self, state: _LoadedState) -> str:
        snapshot = self._snapshot_bytes(state)
        contents: dict[Path, bytes] = {
            state.context.snapshot_path: snapshot,
            state.context.artifact_path: self._tsv_bytes(state.segments),
        }
        if state.reusable:
            contents[state.active_cache_path] = pickle.dumps(
                state.segments, protocol=pickle.HIGHEST_PROTOCOL
            )
        self._atomic_group_replace(contents)
        return self._revision(snapshot)

    @staticmethod
    def _snapshot_bytes(state: _LoadedState) -> bytes:
        payload = {
            "version": 1,
            "segments": state.segments,
            "translation_cache_reusable": state.reusable,
            "translation_cache_key": (
                state.active_cache_path.stem if state.reusable else None
            ),
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _tsv_bytes(segments: Iterable[Mapping[str, Any]]) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, delimiter="\t")
        writer.writerow(
            [
                "segment_id", "speaker", "start", "end", "original",
                "translation", "synthesized_text", "style_prompt", "audio_file",
            ]
        )
        for segment in segments:
            writer.writerow(
                [
                    segment.get("segment_id", ""),
                    segment.get("speaker", ""),
                    f"{DubbingTextService._parse_seconds(segment.get('start')):.3f}",
                    f"{DubbingTextService._parse_seconds(segment.get('end')):.3f}",
                    segment.get("text", ""),
                    segment.get("translation", ""),
                    segment.get("synthesized_text", ""),
                    segment.get("style_prompt", ""),
                    Path(str(segment.get("synthesized_speech_file") or "")).name,
                ]
            )
        return buffer.getvalue().encode("utf-8")

    @staticmethod
    def _atomic_group_replace(contents: Mapping[Path, bytes]) -> None:
        temporary_paths: dict[Path, Path] = {}
        rollback_paths: dict[Path, Path | None] = {}
        replaced: list[Path] = []
        failed_rollback_paths: set[Path] = set()
        try:
            for path, content in contents.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(
                    mode="wb",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".new.tmp",
                    delete=False,
                ) as temporary:
                    temporary.write(content)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_paths[path] = Path(temporary.name)
                if path.is_file():
                    with NamedTemporaryFile(
                        mode="wb",
                        dir=path.parent,
                        prefix=f".{path.name}.",
                        suffix=".rollback.tmp",
                        delete=False,
                    ) as rollback:
                        rollback.write(path.read_bytes())
                        rollback.flush()
                        os.fsync(rollback.fileno())
                        rollback_paths[path] = Path(rollback.name)
                else:
                    rollback_paths[path] = None
            for path, temporary in temporary_paths.items():
                os.replace(temporary, path)
                replaced.append(path)
        except OSError as exc:
            rollback_errors: list[str] = []
            for path in reversed(replaced):
                try:
                    rollback = rollback_paths[path]
                    if rollback is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(rollback, path)
                except OSError as rollback_exc:
                    failed_rollback_paths.add(path)
                    rollback_errors.append(f"{path}: {rollback_exc}")
            if rollback_errors:
                raise DubbingTextWriteError(
                    f"Could not write dubbing text artifacts: {exc}; rollback failed: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise DubbingTextWriteError(f"Could not write dubbing text artifacts: {exc}") from exc
        finally:
            cleanup_paths = [*temporary_paths.values()]
            cleanup_paths.extend(
                rollback
                for path, rollback in rollback_paths.items()
                if path not in failed_rollback_paths
            )
            for temporary in cleanup_paths:
                if temporary is None:
                    continue
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _restore_regenerated_audio(
        current_segment: Mapping[str, Any],
        previous_path: Path | None,
        previous_audio: bytes | None,
    ) -> OSError | None:
        current_path_value = str(
            current_segment.get("synthesized_speech_file") or ""
        ).strip()
        current_path = Path(current_path_value) if current_path_value else None
        try:
            if current_path is not None and current_path != previous_path:
                current_path.unlink(missing_ok=True)
            if previous_audio is None:
                if previous_path is not None:
                    previous_path.unlink(missing_ok=True)
            else:
                if previous_path is None:  # pragma: no cover - defensive invariant
                    raise OSError("Previous audio bytes have no destination path.")
                previous_path.parent.mkdir(parents=True, exist_ok=True)
                with previous_path.open("wb") as restored:
                    restored.write(previous_audio)
                    restored.flush()
                    os.fsync(restored.fileno())
        except OSError as exc:
            return exc
        return None

    @staticmethod
    def _seed_from_dubbing_texts_tsv(path: Path) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        if not path.is_file():
            raise DubbingTextNotFoundError(f"Dubbing texts TSV not found: {path}.")
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                speaker = (row.get("speaker") or "SPEAKER_00").strip()
                start = DubbingTextService._parse_seconds(row.get("start", 0))
                end = DubbingTextService._parse_seconds(row.get("end", 0))
                orig = (row.get("original") or row.get("text") or "").strip()
                trans = (row.get("translation") or orig).strip()
                synth = (row.get("synthesized_text") or trans).strip()
                style = (row.get("style_prompt") or "").strip()
                audio_file = (row.get("audio_file") or "").strip()
                seg: dict[str, Any] = {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": orig,
                    "translation": trans,
                    "short_translation": trans,
                    "very_short_translation": trans,
                    "long_translation": trans,
                    "synthesized_text": synth,
                    "style_prompt": style,
                    "emotion": "Neutral",
                }
                if row.get("segment_id"):
                    seg["segment_id"] = str(row["segment_id"]).strip()
                if audio_file:
                    audio_path = Path(audio_file)
                    if not audio_path.is_file():
                        cand = path.parent / "audio_chunks" / audio_path.name
                        if cand.is_file():
                            audio_path = cand
                        else:
                            cand2 = path.parent / "su_audio_chunks" / audio_path.name
                            if cand2.is_file():
                                audio_path = cand2
                    if audio_path.is_file():
                        seg["synthesized_speech_file"] = str(audio_path)
                segments.append(seg)
        if not segments:
            raise DubbingTextValidationError(f"Dubbing texts TSV is empty: {path}.")
        return segments

    @staticmethod
    def _has_debug_translations(config: object) -> bool:
        debug_dir = DubbingTextService._config_get(config, "debug_dir", "")
        translations_file = Path(str(debug_dir or "")) / "translations.tsv"
        transcription_file = Path(str(DubbingTextService._config_get(config, "transcription_path", "") or ""))
        return translations_file.is_file() and transcription_file.is_file()

    @staticmethod
    def _seed_from_translations_and_transcription(config: object) -> list[dict[str, Any]]:
        debug_dir = DubbingTextService._config_get(config, "debug_dir", "")
        translations_file = Path(str(debug_dir or "")) / "translations.tsv"
        raw_segments = DubbingTextService._seed_from_transcription(config)
        translations: list[str] = []
        if translations_file.is_file():
            with translations_file.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    tr = row.get("translation") or row.get("translated") or ""
                    if tr:
                        translations.append(tr.strip())
        for idx, seg in enumerate(raw_segments):
            if idx < len(translations):
                tr = translations[idx]
                seg["translation"] = tr
                seg["short_translation"] = tr
                seg["very_short_translation"] = tr
                seg["long_translation"] = tr
                seg["synthesized_text"] = tr
        return raw_segments

    @staticmethod
    def _attach_audio_chunks(segments: list[dict[str, Any]], config: object) -> None:
        su_chunks_dir = Path(str(DubbingTextService._config_get(config, "su_audio_chunks_dir", "") or ""))
        chunks_dir = Path(str(DubbingTextService._config_get(config, "audio_chunks_dir", "") or ""))

        for idx, seg in enumerate(segments):
            if seg.get("synthesized_speech_file") and Path(str(seg["synthesized_speech_file"])).is_file():
                continue
            candidates = [
                su_chunks_dir / f"timed_{idx}.wav",
                su_chunks_dir / f"measure_{idx}.wav",
                su_chunks_dir / f"tempo_{idx}.wav",
                chunks_dir / f"{idx}.wav",
            ]
            for cand in candidates:
                if cand.is_file():
                    seg["synthesized_speech_file"] = str(cand)
                    break

    @staticmethod
    def _seed_from_transcription(config: object) -> list[dict[str, Any]]:
        path_value = DubbingTextService._config_get(config, "transcription_path", "")
        path = Path(str(path_value or ""))
        if not path.is_file():
            raise DubbingTextNotFoundError(
                f"Current transcription not found: {path}. Run transcribe_only first."
            )
        timestamp = r"\d{2}\.\d{2}\.\d{2}(?:\.\d{1,3})?"
        pattern = re.compile(
            rf"^\[(?P<start>{timestamp})-(?P<end>{timestamp})\]\s+"
            r"(?P<speaker>[^:]+):\s?(?P<text>.*)$"
        )
        segments: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            match = pattern.match(raw_line.strip())
            if match is None:
                raise DubbingTextValidationError(
                    f"Unexpected transcription format at {path}:{line_number}."
                )
            text = match.group("text").strip()
            segments.append(
                {
                    "speaker": match.group("speaker").strip() or "SPEAKER_00",
                    "start": DubbingTextService._parse_timestamp(match.group("start")),
                    "end": DubbingTextService._parse_timestamp(match.group("end")),
                    "text": text,
                    "translation": text,
                    "short_translation": text,
                    "very_short_translation": text,
                    "long_translation": text,
                    "emotion": "Neutral",
                    "style_prompt": "",
                }
            )
        if not segments:
            raise DubbingTextValidationError(f"Current transcription is empty: {path}.")
        return segments

    @staticmethod
    def _parse_timestamp(value: str) -> float:
        parts = value.split(".")
        hours, minutes, seconds = (int(part) for part in parts[:3])
        milliseconds = int(parts[3].ljust(3, "0")) if len(parts) == 4 else 0
        total_milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
        return total_milliseconds / 1000

    @staticmethod
    def _parse_seconds(value: object) -> float:
        try:
            return float(str(value).strip() or 0.0)
        except (TypeError, ValueError) as exc:
            raise DubbingTextValidationError(f"Invalid timestamp {value!r}.") from exc

    @staticmethod
    def _validate_segment_list(value: object, path: Path) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise DubbingTextValidationError(f"Unexpected segment payload in {path}.")
        return [dict(item) for item in value]

    @staticmethod
    def _ensure_segment_ids(segments: list[dict[str, Any]]) -> bool:
        changed = False
        seen: set[str] = set()
        for segment in segments:
            value = str(segment.get("segment_id") or "").strip()
            try:
                valid = bool(value) and uuid.UUID(value).version == 4 and value not in seen
            except ValueError:
                valid = False
            if not valid:
                value = str(uuid.uuid4())
                segment["segment_id"] = value
                changed = True
            seen.add(value)
        return changed

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DubbingTextValidationError(f"Could not read {path}: {exc}") from exc

    @staticmethod
    def _unpickle(raw: bytes, path: Path) -> object:
        try:
            return pickle.loads(raw)
        except (pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
            raise DubbingTextValidationError(f"Could not decode {path}: {exc}") from exc

    @staticmethod
    def _revision(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _ensure_revision(expected: str, actual: str) -> None:
        if expected != actual:
            raise DubbingTextConflictError("Dubbing texts were changed by another writer.")

    @staticmethod
    def _validate_identity(value: str, label: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise DubbingTextValidationError(f"{label} is required.")
        return normalized

    @classmethod
    def _lock_for(cls, path: Path) -> threading.RLock:
        key = str(path.resolve())
        with cls._locks_guard:
            return cls._locks.setdefault(key, threading.RLock())

    @staticmethod
    @contextmanager
    def _interprocess_lock(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+b") as lock_file:
            try:
                if _msvcrt is not None:
                    lock_file.seek(0)
                    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_LOCK, 1)
                elif _fcntl is not None:
                    _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
                else:  # pragma: no cover
                    raise OSError("No supported file-locking backend is available.")
            except OSError as exc:
                raise DubbingTextWriteError(
                    f"Could not lock dubbing texts for writing: {exc}"
                ) from exc
            try:
                yield
            finally:
                if _msvcrt is not None:
                    lock_file.seek(0)
                    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)
                else:
                    _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)

    @staticmethod
    def _record_value(record: object, *names: str) -> str:
        for name in names:
            if isinstance(record, dict) and name in record:
                return str(record[name] or "")
            if hasattr(record, name):
                return str(getattr(record, name) or "")
        return ""

    @staticmethod
    def _config_get(config: object, key: str, default: object = None) -> object:
        if isinstance(config, Mapping):
            return config.get(key, default)
        getter = getattr(config, "get", None)
        if callable(getter):
            value = getter(key)
            return default if value is None else value
        return default

    @staticmethod
    def _build_context(overrides: object) -> DubbingTextContext:
        from ..core.cache_manager import CacheManager
        from ..core.runner import build_config_from_overrides
        from ..core.smart_dubbing import SmartDubbing

        if not isinstance(overrides, Mapping):
            raise DubbingTextValidationError("Dubbing text configuration must be a mapping.")
        config = build_config_from_overrides(dict(overrides))

        # Dynamically merge speaker_map from project_metadata.json if available
        artifacts_dir = Path(str(config.get("artifacts_dir", "")))
        if artifacts_dir.is_dir():
            metadata_path = artifacts_dir / "project_metadata.json"
            if metadata_path.is_file():
                try:
                    import json, yaml
                    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    saved_cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
                    speaker_map = saved_cfg.get("speaker_map") if isinstance(saved_cfg.get("speaker_map"), dict) else None
                    if speaker_map:
                        config_file = Path(str(config.get("config", "dubbing_config.yml")))
                        global_voices: dict[str, Any] = {}
                        if config_file.is_file():
                            try:
                                loaded_yaml = yaml.safe_load(config_file.read_bytes().decode("utf-8")) or {}
                                global_voices = loaded_yaml.get("voices") or {}
                            except Exception:
                                pass
                        resolved_voices = dict(config.get("voices") or {})
                        for spk_id, prof_name in speaker_map.items():
                            if prof_name in global_voices:
                                resolved_voices[spk_id] = global_voices[prof_name]
                        if resolved_voices:
                            from dataclasses import asdict, is_dataclass
                            from ..core.voice_profiles import normalize_voices
                            raw_map = {}
                            for spk, v in resolved_voices.items():
                                if hasattr(v, "to_dict"):
                                    raw_map[spk] = v.to_dict()
                                elif is_dataclass(v) and not isinstance(v, type):
                                    raw_map[spk] = asdict(v)
                                elif isinstance(v, dict):
                                    raw_map[spk] = v
                            normalized = normalize_voices({"voices": raw_map})
                            if hasattr(config, "set"):
                                config.set("voices", normalized)
                            else:
                                config["voices"] = normalized
                except Exception:
                    pass

        audio_artifacts_dir = Path(str(config.get("audio_artifacts_dir", "")))
        audio_path = audio_artifacts_dir / "source.wav"
        if not audio_path.is_file():
            if (audio_artifacts_dir / "output.wav").is_file():
                audio_path = audio_artifacts_dir / "output.wav"
            elif (audio_artifacts_dir / "background.wav").is_file():
                audio_path = audio_artifacts_dir / "background.wav"
            elif Path(str(config.get("input", ""))).is_file():
                audio_path = Path(str(config.get("input")))
            else:
                raise DubbingTextNotFoundError(
                    f"Expected extracted source audio at {audio_path}. Run transcribe_only first."
                )
        cache_manager = CacheManager(use_cache=True, input_file=config.get("input"))
        key_builder = SmartDubbing.__new__(SmartDubbing)
        key_builder.config = config
        key_builder.cache_manager = cache_manager
        cache_key = key_builder._build_translation_cache_key(str(audio_path))
        snapshot_key = key_builder._build_dubbing_text_snapshot_key(str(audio_path))
        return DubbingTextContext(
            config=config,
            cache_path=cache_manager.get_cache_path("translation") / f"{cache_key}.pkl",
            snapshot_path=cache_manager.get_cache_path("dubbing_texts") / f"{snapshot_key}.pkl",
            artifact_path=Path(str(config.get("artifacts_dir"))) / "dubbing_texts.tsv",
            audio_path=audio_path,
        )

    @staticmethod
    def _build_dubber(config: object) -> object:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from ..core.smart_dubbing import SmartDubbing

        return SmartDubbing(config)

