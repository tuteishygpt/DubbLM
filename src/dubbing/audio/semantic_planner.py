"""Deterministic semantic segmentation for isolated-speaker transcripts.

The planner treats VAD as acoustic evidence and ASR word timestamps as the
placement authority.  Neither one is allowed to create a semantic boundary by
itself.  The public entry point deliberately has no dependency on the dubbing
or translation orchestrators so boundary selection can be tested and cached
independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence


SEMANTIC_PLANNER_VERSION = "semantic_planner_v2"
INCOMPLETE_TAIL_RULES_VERSION = "incomplete_tail_en_v1"
SEMANTIC_BOUNDARY_PROMPT_VERSION = "semantic_boundary_prompt_v1"

_SENTENCE_PUNCTUATION = (".", "!", "?", "…")
_SOFT_PUNCTUATION = (",", ";", ":", "—", "-")
_EN_INCOMPLETE_LAST_TOKENS = {
    "a", "an", "and", "as", "at", "because", "before", "but", "by",
    "despite", "during", "for", "from", "if", "in", "into", "nor", "of",
    "on", "or", "since", "so", "than", "that", "the", "though", "through",
    "to", "unless", "until", "when", "whenever", "where", "whereas",
    "wherever", "whether", "while", "with", "without", "yet",
}
_EN_INCOMPLETE_PHRASES = (
    ("i", "think", "that"),
    ("if", "we've", "laid", "off"),
    ("if", "we", "have", "laid", "off"),
)


class SemanticSegmentationError(ValueError):
    """Raised when the configured hard limit cannot be met losslessly."""


@dataclass(frozen=True)
class SemanticPlannerConfig:
    preferred_duration: float = 15.0
    hard_duration: float = 35.0
    search_window: float = 10.0
    classifier_timeout: float = 30.0
    classifier_batch_size: int = 50
    classifier_batch_characters: int = 12_000

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SemanticPlannerConfig":
        return cls(
            preferred_duration=float(config.get("tts_preferred_segment_duration", 15.0)),
            hard_duration=float(config.get("tts_hard_segment_duration", 35.0)),
            search_window=float(config.get("semantic_split_search_window", 10.0)),
        )


@dataclass(frozen=True)
class SemanticPlanResult:
    units: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    fingerprint: str
    cache_persistable: bool
    classifier_mode: str

    @property
    def semantic_plan_fingerprint(self) -> str:
        return self.fingerprint


def _finite_range(
    record: Mapping[str, Any], *, kind: str, provider_index: int, speaker: str
) -> tuple[float, float]:
    try:
        start = float(record["start"])
        end = float(record["end"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{speaker} {kind} {provider_index} has invalid timestamps") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
        raise ValueError(
            f"{speaker} {kind} {provider_index} has invalid timestamps: {start!r}..{end!r}"
        )
    return start, end


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _stable_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _normalize_regions(
    regions: Sequence[Any], speaker: str
) -> list[dict[str, Any]]:
    validated: list[tuple[float, float, int, Optional[str]]] = []
    for provider_index, region in enumerate(regions):
        if isinstance(region, Mapping):
            start, end = _finite_range(
                region, kind="VAD region", provider_index=provider_index, speaker=speaker
            )
            supplied_id = region.get("vad_region_id") or region.get("id")
        else:
            try:
                start, end = float(region[0]), float(region[1])
            except (TypeError, ValueError, IndexError, OverflowError) as exc:
                raise ValueError(
                    f"{speaker} VAD region {provider_index} has invalid timestamps"
                ) from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError(
                    f"{speaker} VAD region {provider_index} has invalid timestamps: "
                    f"{start!r}..{end!r}"
                )
            supplied_id = None
        validated.append((start, end, provider_index, str(supplied_id) if supplied_id else None))

    validated.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        {
            "start": start,
            "end": end,
            "provider_index": provider_index,
            "vad_region_id": supplied_id or f"{speaker}:v{index:06d}",
        }
        for index, (start, end, provider_index, supplied_id) in enumerate(validated)
    ]


def _normalize_foreign_activity(
    activity: Sequence[Any], speaker: str
) -> list[tuple[float, float]]:
    """Validate and coalesce foreign word activity into stable components."""
    validated: set[tuple[float, float]] = set()
    for provider_index, record in enumerate(activity):
        try:
            if isinstance(record, Mapping):
                start = float(record["start"])
                end = float(record["end"])
            else:
                start = float(record[0])
                end = float(record[1])
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError(
                f"{speaker} foreign activity {provider_index} has invalid timestamps"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
            raise ValueError(
                f"{speaker} foreign activity {provider_index} has invalid timestamps: "
                f"{start!r}..{end!r}"
            )
        if start != end:
            validated.add((start, end))

    coalesced: list[tuple[float, float]] = []
    for start, end in sorted(validated):
        if coalesced and start <= coalesced[-1][1]:
            previous_start, previous_end = coalesced[-1]
            coalesced[-1] = (previous_start, max(previous_end, end))
        else:
            coalesced.append((start, end))
    return coalesced


def _normalize_segments(
    segments: Sequence[Mapping[str, Any]], speaker: str
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for provider_index, source in enumerate(segments):
        if not isinstance(source, Mapping):
            raise ValueError(f"{speaker} segment {provider_index} is not a mapping")
        start, end = _finite_range(
            source, kind="segment", provider_index=provider_index, speaker=speaker
        )
        text = _normalized_text(source.get("text"))
        normalized = dict(source)
        normalized.update(
            start=start,
            end=end,
            text=text,
            speaker=str(source.get("speaker") or speaker),
            _provider_index=provider_index,
        )
        validated.append(normalized)

    validated.sort(
        key=lambda item: (
            item["start"], item["end"], item["text"].casefold(), item["_provider_index"]
        )
    )
    for item in validated:
        item["source_segment_id"] = str(
            item.get("source_segment_id")
            or item.get("asr_segment_id")
            or f"{speaker}:s{_stable_hash([item['start'], item['end'], item['text'].casefold()], 12)}"
        )
    return validated


def _assign_region(word: Mapping[str, Any], regions: Sequence[Mapping[str, Any]]) -> Optional[str]:
    candidates: list[tuple[float, float, str]] = []
    word_start, word_end = float(word["start"]), float(word["end"])
    word_center = (word_start + word_end) / 2.0
    for region in regions:
        overlap = max(0.0, min(word_end, region["end"]) - max(word_start, region["start"]))
        tolerance_intersection = word_end > region["start"] - 0.15 and word_start < region["end"] + 0.15
        if overlap <= 0.0 and not tolerance_intersection:
            continue
        center_distance = abs(word_center - ((region["start"] + region["end"]) / 2.0))
        candidates.append((-overlap, center_distance, region["vad_region_id"]))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _normalize_words(
    segments: Sequence[Mapping[str, Any]], regions: Sequence[Mapping[str, Any]], speaker: str
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    any_word_list = False
    for segment in segments:
        words = segment.get("words")
        if words is None:
            continue
        if not isinstance(words, list):
            raise ValueError(
                f"{speaker} segment {segment['_provider_index']} words must be a list"
            )
        any_word_list = any_word_list or bool(words)
        for local_index, source_word in enumerate(words):
            if not isinstance(source_word, Mapping):
                raise ValueError(
                    f"{speaker} word {segment['_provider_index']}:{local_index} is not a mapping"
                )
            provider_index = len(flattened)
            start, end = _finite_range(
                source_word, kind="word", provider_index=provider_index, speaker=speaker
            )
            text = _normalized_text(source_word.get("word") or source_word.get("text"))
            if not text:
                raise ValueError(f"{speaker} word {provider_index} has empty text")
            confidence = source_word.get("confidence")
            try:
                confidence_value = float(confidence) if confidence is not None else float("-inf")
            except (TypeError, ValueError, OverflowError):
                confidence_value = float("-inf")
            if not math.isfinite(confidence_value):
                confidence_value = float("-inf")
            flattened.append(
                {
                    "word": text,
                    "start": start,
                    "end": end,
                    "confidence": None if confidence_value == float("-inf") else confidence_value,
                    "_confidence_sort": confidence_value,
                    "_provider_index": provider_index,
                    "source_segment_id": segment["source_segment_id"],
                }
            )

    if not any_word_list:
        return []

    flattened.sort(
        key=lambda word: (
            word["start"], word["end"], word["word"].casefold(),
            -word["_confidence_sort"], word["_provider_index"],
        )
    )
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for word in flattened:
        key = (
            word["word"].casefold(), round(word["start"] * 1000), round(word["end"] * 1000)
        )
        if key in seen:
            continue
        seen.add(key)
        region_id = _assign_region(word, regions)
        if regions and region_id is None:
            continue
        clean = {key: value for key, value in word.items() if not key.startswith("_")}
        clean["vad_region_id"] = region_id
        clean["source_word_index"] = len(deduplicated)
        deduplicated.append(clean)
    return deduplicated


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)


def _has_incomplete_tail(text: str, source_language: str) -> bool:
    if not source_language.lower().startswith("en"):
        return False
    tokens = _word_tokens(text)
    if not tokens:
        return False
    if tokens[-1] in _EN_INCOMPLETE_LAST_TOKENS:
        return True
    return any(tuple(tokens[-len(phrase):]) == phrase for phrase in _EN_INCOMPLETE_PHRASES)


def _cross_boundary_is_incomplete(left_text: str, right_text: str, source_language: str) -> bool:
    """Detect punctuation that explicitly signals continuation across a VAD gap."""
    if not source_language.lower().startswith("en") or not left_text.rstrip().endswith(","):
        return False
    right_tokens = _word_tokens(right_text)
    if not right_tokens:
        return False
    first_raw = right_text.lstrip()[:1]
    return right_tokens[0] in {"and", "but", "nor", "or", "so", "yet"} or first_raw.islower()


def _context(items: Sequence[Mapping[str, Any]], index: int) -> tuple[str, str]:
    left = " ".join(str(item["text"]) for item in items[max(0, index - 19): index + 1])
    right = " ".join(str(item["text"]) for item in items[index + 1: index + 21])
    return left, right


def _build_candidates(
    items: Sequence[Mapping[str, Any]],
    speaker: str,
    source_language: str,
    foreign_activity: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index in range(len(items) - 1):
        left, right = items[index], items[index + 1]
        left_context, right_context = _context(items, index)
        pause = max(0.0, float(right["start"]) - float(left["end"]))
        left_text = str(left["text"]).strip()
        speaker_turn_boundary = any(
            float(left["end"]) < activity_start
            and activity_end < float(right["start"])
            for activity_start, activity_end in foreign_activity
        )
        strong_early_utterance = (
            left_text.endswith(_SENTENCE_PUNCTUATION)
            and left.get("source_segment_id") != right.get("source_segment_id")
            and pause >= 0.5
        )
        if (
            _has_incomplete_tail(left_context, source_language)
            or _cross_boundary_is_incomplete(left_text, right_context, source_language)
        ):
            local_decision = "HARD_CONTINUE"
            reason_code = "incomplete_tail"
            semantic_rank = 0
        elif left_text.endswith(_SENTENCE_PUNCTUATION):
            local_decision = "LOCAL_CUT_SENTENCE"
            reason_code = "sentence_final"
            semantic_rank = 3
        elif left_text.endswith(_SOFT_PUNCTUATION) and pause >= 0.2:
            local_decision = "LOCAL_CUT_CLAUSE"
            reason_code = "complete_clause"
            semantic_rank = 2
        else:
            local_decision = "AMBIGUOUS"
            reason_code = "ambiguous"
            semantic_rank = 1
        source_index = int(left.get("source_word_index", left.get("source_index", index)))
        candidate_id = _stable_hash(
            [SEMANTIC_PLANNER_VERSION, speaker, source_index, round(float(left["end"]) * 1000)]
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "id": candidate_id,
                "speaker": speaker,
                "source_index": source_index,
                "source_indexes": [
                    source_index,
                    int(right.get("source_word_index", right.get("source_index", index + 1))),
                ],
                "left_context": left_context,
                "right_context": right_context,
                "candidate_time": float(left["end"]),
                "source_pause": pause,
                "local_decision": local_decision,
                "llm_decision": None,
                "llm_confidence": None,
                "classifier_veto": False,
                "final_decision": (
                    "CUT" if local_decision.startswith("LOCAL_CUT") else "CONTINUE"
                ),
                "reason_code": reason_code,
                "classifier_mode": "deterministic-only",
                "fallback_reason": None,
                "chosen": False,
                "semantic_rank": semantic_rank,
                "decision_priority": 1,
                "punctuation_rank": semantic_rank,
                "boundary_type": "semantic",
                "continuation_id": None,
                "semantic_unit_id": None,
                "speaker_turn_boundary": speaker_turn_boundary,
                "strong_early_utterance": strong_early_utterance,
                "lineage": {
                    "source_segment_ids": _ordered_unique(
                        [left.get("source_segment_id"), right.get("source_segment_id")]
                    ),
                    "vad_region_ids": _ordered_unique(
                        [left.get("vad_region_id"), right.get("vad_region_id")]
                    ),
                },
            }
        )
    return candidates


def _apply_mandatory_boundaries(
    candidates: Sequence[MutableMapping[str, Any]], speaker: str
) -> None:
    for candidate in candidates:
        if not candidate["speaker_turn_boundary"]:
            continue
        candidate["final_decision"] = "CUT"
        candidate["reason_code"] = "speaker_turn"
        candidate["boundary_type"] = "speaker_turn"
        candidate["decision_priority"] = 3
        if candidate["local_decision"] == "HARD_CONTINUE":
            candidate["continuation_id"] = _stable_hash(
                [speaker, candidate["source_index"], candidate["candidate_time"]]
            )


def _is_mandatory_boundary(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate["speaker_turn_boundary"]) or (
        bool(candidate["strong_early_utterance"])
        and candidate["local_decision"] == "LOCAL_CUT_SENTENCE"
    )


def _classifier_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate["candidate_id"],
        "left": candidate["left_context"],
        "right": candidate["right_context"],
        "pause": candidate["source_pause"],
    }


def semantic_classification_cache_key(
    candidate_payload: Mapping[str, Any],
    *,
    source_language: str,
    classifier_context: Optional[Mapping[str, Any]] = None,
) -> str:
    return _stable_hash(
        {
            "prompt_parser": SEMANTIC_BOUNDARY_PROMPT_VERSION,
            "source_language": source_language,
            "classifier": dict(classifier_context or {}),
            "candidate": dict(candidate_payload),
        },
        32,
    )


def _apply_classifier_entry(candidate: MutableMapping[str, Any], entry: Mapping[str, Any]) -> bool:
    decision = str(entry.get("decision") or "").upper()
    try:
        confidence = float(entry.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        confidence = float("nan")
    if (
        decision not in {"CUT", "CONTINUE", "UNCERTAIN"}
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        return False
    candidate["llm_decision"] = decision
    candidate["llm_confidence"] = confidence
    if decision == "CUT" and confidence >= 0.75:
        candidate["final_decision"] = "CUT"
        candidate["decision_priority"] = 2
        candidate["reason_code"] = str(entry.get("reason_code") or "llm_cut")
    elif decision == "CONTINUE" and confidence >= 0.60:
        candidate["final_decision"] = "CONTINUE"
        candidate["classifier_veto"] = True
        candidate["reason_code"] = str(entry.get("reason_code") or "llm_continue")
    else:
        candidate["fallback_reason"] = "low_confidence_or_uncertain"
    return True


def _coerce_response(response: Any) -> Mapping[str, Any]:
    if hasattr(response, "text"):
        response = response.text
    if isinstance(response, str):
        response = json.loads(response)
    if not isinstance(response, Mapping):
        raise ValueError("Classifier response is not an object")
    return response


def _call_classifier_with_timeout(
    classifier: Callable[[dict[str, Any]], Any],
    request: dict[str, Any],
    timeout: float,
) -> Any:
    """Run a provider call with a hard wall-clock limit for the planner.

    Provider SDKs do not share a common timeout argument.  A daemon worker lets
    planning fall back deterministically even when a client blocks internally;
    the abandoned call cannot keep process shutdown alive.
    """
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put((True, classifier(request)))
        except Exception as exc:
            result_queue.put((False, exc))

    worker = threading.Thread(
        target=invoke,
        name="semantic-boundary-classifier",
        daemon=True,
    )
    worker.start()
    try:
        succeeded, value = result_queue.get(timeout=max(0.0, float(timeout)))
    except queue.Empty as exc:
        raise TimeoutError(
            f"Semantic boundary classifier exceeded {timeout:.3f}s"
        ) from exc
    if succeeded:
        return value
    raise value


def _classify_candidates(
    candidates: list[dict[str, Any]],
    classifier: Optional[Callable[[dict[str, Any]], Any]],
    classifier_status: str,
    source_language: str,
    config: SemanticPlannerConfig,
    classification_cache_get: Optional[Callable[[str], Any]] = None,
    classification_cache_set: Optional[Callable[[str, Any], None]] = None,
    classifier_cache_context: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, str]:
    ambiguous = [
        item
        for item in candidates
        if item["local_decision"] == "AMBIGUOUS"
        and not item["speaker_turn_boundary"]
    ]
    mode = classifier_status or "deterministic-only"
    for candidate in candidates:
        candidate["classifier_mode"] = mode
    if not ambiguous:
        return True, mode
    if mode == "deterministic-only":
        return True, mode

    persistable = True
    uncached: list[dict[str, Any]] = []
    for candidate in ambiguous:
        cache_key = semantic_classification_cache_key(
            _classifier_payload(candidate),
            source_language=source_language,
            classifier_context=classifier_cache_context,
        )
        candidate["_classification_cache_key"] = cache_key
        cached_entry = classification_cache_get(cache_key) if classification_cache_get else None
        if isinstance(cached_entry, Mapping) and _apply_classifier_entry(candidate, cached_entry):
            candidate["classification_cache_hit"] = True
        else:
            uncached.append(candidate)
    ambiguous = uncached
    if not ambiguous:
        return mode not in {"initialization_failed", "unavailable"}, mode
    if classifier is None or mode in {"initialization_failed", "unavailable"}:
        for candidate in ambiguous:
            candidate["fallback_reason"] = mode or "unavailable"
        return False, mode

    def build_request(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "source_language": source_language,
            "prompt_version": SEMANTIC_BOUNDARY_PROMPT_VERSION,
            "timeout": config.classifier_timeout,
            "candidates": [_classifier_payload(item) for item in batch],
        }

    def request_characters(batch: Sequence[Mapping[str, Any]]) -> int:
        return len(
            json.dumps(
                build_request(batch),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for candidate in ambiguous:
        proposed = [*current, candidate]
        if current and (
            len(proposed) > config.classifier_batch_size
            or request_characters(proposed) > config.classifier_batch_characters
        ):
            batches.append(current)
            current = []
        if request_characters([candidate]) > config.classifier_batch_characters:
            persistable = False
            candidate["fallback_reason"] = "classifier_request_too_large"
            continue
        current.append(candidate)
    if current:
        batches.append(current)

    for batch in batches:
        request = build_request(batch)
        try:
            response = _coerce_response(
                _call_classifier_with_timeout(
                    classifier,
                    request,
                    config.classifier_timeout,
                )
            )
            entries = response.get("boundaries")
            if not isinstance(entries, list):
                raise ValueError("Classifier response has no boundaries list")
        except Exception as exc:
            persistable = False
            for candidate in batch:
                candidate["fallback_reason"] = f"classifier_error:{type(exc).__name__}"
            continue

        by_id: dict[str, list[Mapping[str, Any]]] = {}
        known = {item["candidate_id"] for item in batch}
        invalid_batch = False
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
                invalid_batch = True
                continue
            by_id.setdefault(entry["id"], []).append(entry)
            if entry["id"] not in known:
                invalid_batch = True
        persistable = persistable and not invalid_batch

        for candidate in batch:
            matches = by_id.get(candidate["candidate_id"], [])
            if len(matches) != 1:
                persistable = False
                candidate["fallback_reason"] = "missing_id" if not matches else "duplicate_id"
                continue
            entry = matches[0]
            if not _apply_classifier_entry(candidate, entry):
                persistable = False
                candidate["fallback_reason"] = "invalid_result"
                continue
            if classification_cache_set:
                classification_cache_set(
                    candidate["_classification_cache_key"], dict(entry)
                )
    return persistable, mode


def _eligible_key(candidate: Mapping[str, Any], target: float) -> tuple[Any, ...]:
    return (
        -int(candidate["semantic_rank"]),
        -int(candidate["decision_priority"]),
        abs(float(candidate["candidate_time"]) - target),
        -float(candidate["source_pause"]),
        int(candidate["source_index"]),
    )


def _forced_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(candidate["punctuation_rank"]),
        -float(candidate["source_pause"]),
        -float(candidate["candidate_time"]),
        int(candidate["source_index"]),
    )


def _select_boundaries(
    items: Sequence[Mapping[str, Any]],
    candidates: list[dict[str, Any]],
    config: SemanticPlannerConfig,
    speaker: str,
) -> list[tuple[int, Optional[dict[str, Any]]]]:
    if not items:
        return []
    selected: list[tuple[int, Optional[dict[str, Any]]]] = []
    unit_start_index = 0
    chain_end = float(items[-1]["end"])
    effective_window = min(
        config.search_window,
        config.preferred_duration,
        config.hard_duration - config.preferred_duration,
    )
    effective_window = max(0.0, effective_window)

    while unit_start_index < len(items):
        if unit_start_index == len(items) - 1:
            selected.append((len(items), None))
            break
        unit_start = float(items[unit_start_index]["start"])
        target = unit_start + config.preferred_duration
        hard_end = min(chain_end, unit_start + config.hard_duration)
        search_start = max(unit_start, target - effective_window)
        search_end = min(hard_end, target + effective_window)
        mandatory_index = next(
            (
                index
                for index in range(unit_start_index, len(candidates))
                if _is_mandatory_boundary(candidates[index])
            ),
            None,
        )
        mandatory = candidates[mandatory_index] if mandatory_index is not None else None
        relevant_end = mandatory_index + 1 if mandatory_index is not None else len(candidates)
        relevant = candidates[unit_start_index:relevant_end]
        eligible = [
            item for item in relevant
            if item["final_decision"] == "CUT"
            and search_start <= item["candidate_time"] <= search_end
        ]
        chosen: Optional[dict[str, Any]] = min(eligible, key=lambda c: _eligible_key(c, target)) if eligible else None
        if chosen is None:
            later = [
                item for item in relevant
                if item["final_decision"] == "CUT"
                and search_end < item["candidate_time"] <= hard_end
            ]
            if later:
                chosen = min(later, key=lambda c: _eligible_key(c, target))
        if (
            chosen is None
            and mandatory is not None
            and float(mandatory["candidate_time"]) <= hard_end
        ):
            chosen = mandatory
        if chosen is None and mandatory is None and chain_end <= hard_end:
            selected.append((len(items), None))
            break
        if chosen is None:
            forced = [item for item in relevant if item["candidate_time"] <= hard_end]
            safe = [
                item
                for item in forced
                if item["local_decision"] != "HARD_CONTINUE"
                and not item.get("classifier_veto", False)
            ]
            pool = safe or forced
            if not pool:
                offending = items[unit_start_index]
                raise SemanticSegmentationError(
                    f"{speaker}: no complete word/segment ends before the {config.hard_duration:.3f}s "
                    f"hard limit; offending item {offending.get('text')!r} spans "
                    f"{offending['start']}..{offending['end']}"
                )
            chosen = min(pool, key=_forced_key)
            continuation_id = _stable_hash(
                [speaker, chosen["source_index"], chosen["candidate_time"]]
            )
            chosen["boundary_type"] = "technical_continuation"
            chosen["continuation_id"] = continuation_id
            chosen["reason_code"] = "forced_hard_limit"

        chosen["chosen"] = True
        cut_index = candidates.index(chosen) + 1
        if cut_index <= unit_start_index:
            raise SemanticSegmentationError(f"{speaker}: semantic planner made no forward progress")
        selected.append((cut_index, chosen))
        unit_start_index = cut_index
    return selected


def _ordered_unique(values: Sequence[Optional[str]]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _build_units(
    items: Sequence[Mapping[str, Any]],
    selections: Sequence[tuple[int, Optional[dict[str, Any]]]],
    speaker: str,
    words_mode: bool,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    assigned_source_segments: set[str] = set()
    assigned_vad_regions: set[str] = set()
    start_index = 0
    boundary_before: Optional[dict[str, Any]] = None
    previous_continuation: Optional[str] = None
    for end_index, boundary_after in selections:
        slice_items = list(items[start_index:end_index])
        if not slice_items:
            continue
        continuation_id = previous_continuation or (
            boundary_after.get("continuation_id") if boundary_after else None
        )
        source_word_range = (
            [slice_items[0]["source_word_index"], slice_items[-1]["source_word_index"] + 1]
            if words_mode else None
        )
        contributing_source_segment_ids = _ordered_unique(
            [str(item.get("source_segment_id")) if item.get("source_segment_id") else None for item in slice_items]
        )
        contributing_vad_region_ids = _ordered_unique(
            [str(item.get("vad_region_id")) if item.get("vad_region_id") else None for item in slice_items]
        )
        source_segment_ids = [
            value for value in contributing_source_segment_ids
            if value not in assigned_source_segments
        ]
        vad_region_ids = [
            value for value in contributing_vad_region_ids
            if value not in assigned_vad_regions
        ]
        assigned_source_segments.update(source_segment_ids)
        assigned_vad_regions.update(vad_region_ids)
        start = float(slice_items[0]["start"])
        end = float(slice_items[-1]["end"])
        first_source = source_word_range[0] if source_word_range else start_index
        last_source = source_word_range[1] - 1 if source_word_range else end_index - 1
        semantic_unit_id = _stable_hash(
            [
                SEMANTIC_PLANNER_VERSION, speaker, first_source, last_source,
                round(start * 1000), round(end * 1000),
            ]
        )
        boundary_metadata = None
        if boundary_before is not None:
            boundary_metadata = {
                "candidate_id": boundary_before["candidate_id"],
                "type": boundary_before["boundary_type"],
                "source_pause": boundary_before["source_pause"],
                "decision": boundary_before["final_decision"],
                "confidence": boundary_before["llm_confidence"],
                "reason_code": boundary_before["reason_code"],
                "classifier_mode": boundary_before["classifier_mode"],
            }
        unit: dict[str, Any] = {
            "semantic_unit_id": semantic_unit_id,
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": " ".join(str(item["text"]) for item in slice_items).strip(),
            "source_word_range": source_word_range,
            "source_segment_ids": source_segment_ids,
            "vad_region_ids": vad_region_ids,
            "lock_boundary_before": bool(units),
            "boundary_before": boundary_metadata,
            "continuation_id": continuation_id,
        }
        if words_mode:
            unit["words"] = [
                {key: value for key, value in item.items() if key not in {"text"}}
                for item in slice_items
            ]
        units.append(unit)
        if boundary_after is not None:
            boundary_after["semantic_unit_id"] = semantic_unit_id
        previous_continuation = (
            boundary_after.get("continuation_id") if boundary_after else None
        )
        boundary_before = boundary_after
        start_index = end_index
    return units


def _wordless_items(
    segments: Sequence[Mapping[str, Any]], regions: Sequence[Mapping[str, Any]], config: SemanticPlannerConfig,
    speaker: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        duration = float(segment["end"]) - float(segment["start"])
        if duration > config.hard_duration:
            raise SemanticSegmentationError(
                f"{speaker}: indivisible ASR segment {segment['source_segment_id']} spans "
                f"{segment['start']}..{segment['end']} ({duration:.3f}s), exceeding the "
                f"{config.hard_duration:.3f}s hard limit; use a word-timestamp backend "
                "or disable semantic splitting."
            )
        overlapping = [
            region for region in regions
            if region["end"] > segment["start"] and region["start"] < segment["end"]
        ]
        if regions and not overlapping:
            continue
        item = dict(segment)
        item["source_index"] = index
        item["vad_region_id"] = overlapping[0]["vad_region_id"] if overlapping else None
        items.append(item)
    return items


def plan_semantic_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    vad_regions: Sequence[Any] = (),
    foreign_activity: Sequence[Mapping[str, Any] | Sequence[float]] = (),
    speaker: str,
    source_language: str = "en",
    config: Optional[SemanticPlannerConfig] = None,
    classifier: Optional[Callable[[dict[str, Any]], Any]] = None,
    classifier_status: str = "deterministic-only",
    classification_cache_get: Optional[Callable[[str], Any]] = None,
    classification_cache_set: Optional[Callable[[str, Any], None]] = None,
    classifier_cache_context: Optional[Mapping[str, Any]] = None,
) -> SemanticPlanResult:
    """Normalize, classify and split one isolated-speaker continuity chain."""

    config = config or SemanticPlannerConfig()
    if (
        not math.isfinite(config.preferred_duration) or config.preferred_duration <= 0
        or not math.isfinite(config.hard_duration) or config.hard_duration < config.preferred_duration
        or not math.isfinite(config.search_window) or config.search_window < 0
    ):
        raise ValueError("Invalid semantic planner durations")
    normalized_segments = _normalize_segments(segments, speaker)
    normalized_regions = _normalize_regions(vad_regions, speaker)
    normalized_foreign_activity = _normalize_foreign_activity(foreign_activity, speaker)
    words = _normalize_words(normalized_segments, normalized_regions, speaker)
    words_mode = bool(words)
    if words_mode:
        items = [
            {
                **word,
                "text": word["word"],
            }
            for word in words
        ]
    else:
        items = _wordless_items(normalized_segments, normalized_regions, config, speaker)
    if not items:
        payload = {"version": SEMANTIC_PLANNER_VERSION, "units": [], "boundaries": []}
        return SemanticPlanResult([], [], _stable_hash(payload), True, classifier_status)

    candidates = _build_candidates(
        items, speaker, source_language, normalized_foreign_activity
    )
    persistable, classifier_mode = _classify_candidates(
        candidates,
        classifier,
        classifier_status,
        source_language,
        config,
        classification_cache_get,
        classification_cache_set,
        classifier_cache_context,
    )
    _apply_mandatory_boundaries(candidates, speaker)
    selections = _select_boundaries(items, candidates, config, speaker)
    units = _build_units(items, selections, speaker, words_mode)
    canonical = {
        "version": SEMANTIC_PLANNER_VERSION,
        "units": [
            {
                key: unit.get(key)
                for key in (
                    "semantic_unit_id", "speaker", "start", "end", "text",
                    "source_word_range", "source_segment_ids", "vad_region_ids",
                    "lock_boundary_before", "boundary_before", "continuation_id",
                )
            }
            for unit in units
        ],
        "boundaries": [
            {
                "candidate_id": item["candidate_id"],
                "final_decision": item["final_decision"],
                "chosen": item["chosen"],
                "boundary_type": item["boundary_type"],
                "continuation_id": item["continuation_id"],
                "speaker_turn_boundary": item["speaker_turn_boundary"],
                "strong_early_utterance": item["strong_early_utterance"],
            }
            for item in candidates
        ],
    }
    fingerprint = _stable_hash(canonical)
    return SemanticPlanResult(units, candidates, fingerprint, persistable, classifier_mode)


class SemanticSegmentPlanner:
    """Stateful facade useful to callers that keep classifier configuration."""

    def __init__(
        self,
        config: Optional[SemanticPlannerConfig] = None,
        *,
        source_language: str = "en",
        classifier: Optional[Callable[[dict[str, Any]], Any]] = None,
        classifier_status: str = "deterministic-only",
    ) -> None:
        self.config = config or SemanticPlannerConfig()
        self.source_language = source_language
        self.classifier = classifier
        self.classifier_status = classifier_status

    def plan(
        self, segments: Sequence[Mapping[str, Any]], *, vad_regions: Sequence[Any], speaker: str
    ) -> SemanticPlanResult:
        return plan_semantic_segments(
            segments,
            vad_regions=vad_regions,
            speaker=speaker,
            source_language=self.source_language,
            config=self.config,
            classifier=self.classifier,
            classifier_status=self.classifier_status,
        )
