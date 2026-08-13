"""Per-run state shared by SmartDubbing pipeline services."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PipelineRunContext:
    """The call-order-dependent state shared within one pipeline run."""

    semantic_plan_fingerprint: Optional[str] = None
    semantic_plan_cache_persistable: bool = True
    plan_dependent_cache_allowed: bool = True
    timing_source_audio_file: Optional[str] = None
    timing_source_duration: Optional[float] = None


_FACADE_MIRRORS = {
    "semantic_plan_fingerprint": "_semantic_plan_fingerprint",
    "semantic_plan_cache_persistable": "_semantic_plan_cache_persistable",
    "plan_dependent_cache_allowed": "_plan_dependent_cache_allowed",
    "timing_source_audio_file": "_timing_source_audio_file",
    "timing_source_duration": "_timing_source_duration",
}


def snapshot_context(facade: Any) -> PipelineRunContext:
    """Build a context from the five compatibility attributes on a facade."""
    defaults = PipelineRunContext()
    return PipelineRunContext(
        **{
            field_name: getattr(
                facade, facade_name, getattr(defaults, field_name)
            )
            for field_name, facade_name in _FACADE_MIRRORS.items()
        }
    )


def commit_context(
    facade: Any,
    context: PipelineRunContext,
    *,
    changed_fields: Iterable[str],
) -> None:
    """Copy only stage-changed context values to compatibility attributes."""
    changed_fields = set(changed_fields)
    unknown_fields = changed_fields.difference(_FACADE_MIRRORS)
    if unknown_fields:
        raise ValueError(
            f"Unknown PipelineRunContext fields: {sorted(unknown_fields)}"
        )
    for field_name in changed_fields:
        facade_name = _FACADE_MIRRORS[field_name]
        setattr(facade, facade_name, getattr(context, field_name))


def active_context(facade: Any) -> PipelineRunContext:
    """Return a run's shared context, or a short snapshot for a direct call."""
    context = getattr(facade, "_pipeline_run_context", None)
    if context is not None:
        return context
    return snapshot_context(facade)


def validate_plan_dependent_segments(
    context: PipelineRunContext, segments: List[Dict[str, Any]]
) -> None:
    """Reject segment payloads that do not match the active semantic plan."""
    expected = context.semantic_plan_fingerprint
    if expected is None:
        return
    if any(
        segment.get("semantic_plan_fingerprint") != expected
        for segment in segments
    ):
        raise ValueError(
            "Cached artifact semantic_plan_fingerprint is absent or does not "
            "match the active semantic plan"
        )
