# SmartDubbing Modular Refactor Design

## Мэта

Раздзяліць `src/dubbing/core/smart_dubbing.py` на невялікія модулі з выразнымі
адказнасцямі, пакінуўшы `SmartDubbing` тонкім facade і lifecycle-owner.

Рэфактарынг захоўвае runtime-паводзіны, праграмны API, Gradio UI, парадак
этапаў, cache identities і фарматы, artifact paths і фарматы, segment mutation
contract і тыпы памылак. Два ўзгодненыя breaking changes:

1. чысты CLI выдаляецца;
2. legacy per-speaker TTS configuration выдаляецца на карысць `voices:`.

## Зафіксаваны baseline

- `smart_dubbing.py`: 4,124 радкі, каля 63 метадаў.
- Поўны baseline: `346 passed, 41 warnings`.
- Прамы runtime entry point: `dubbing.core.smart_dubbing.SmartDubbing`.
- `src/dubbing/core/runner.py` застаецца агульным праграмным/Gradio entry point.
- Baseline code review не знайшоў Critical issues, але выявіў Important
  extraction risks, апісаныя ніжэй.

## Межы scope

### У scope

- Модульны падзел `SmartDubbing` па адказнасцях.
- Яўны per-run context для схаванага mutable state.
- Тонкія compatibility-метады на `SmartDubbing` з тымі ж descriptors,
  signatures і observable behavior.
- Выдаленне чыста CLI-файлаў, entry point, argparse-кода, CLI-дакументацыі і
  CLI-only тэстаў.
- Выдаленне legacy per-speaker TTS configuration, яго normalization, mirrors,
  UI/runner/config paths, дакументацыі і тэстаў.
- Fail-fast migration error пры наяўнасці legacy TTS keys.
- Выдаленне толькі доказна мёртвага або недасягальнага кода.
- Characterization, module, integration і regression tests.
- Baseline і final code review.

### Не ў scope

- Выпраўленне pre-existing behavioral, security, lifecycle, cache-integrity або
  performance праблем.
- Змена cache schema, key strings, prefixes, serialization або cache layout.
- Змена artifact names, paths, formats або source-selection order.
- Змена segment dictionaries на dataclasses/immutable models.
- Змена translation prompts, semantic planning, TTS candidate policy,
  reference rules, timing або audio assembly algorithms.
- Змена backend factories, provider contracts або UI workflow.
- Новы lint/format setup або шырокае перафарматаванне суседніх файлаў.

## Абраны падыход

Выкарыстоўваецца `facade + focused services`. `SmartDubbing` захоўвае
аркестрацыю, lifecycle і compatibility surface. Вылучаныя модулі атрымліваюць
яўныя ўваходы і залежнасці, не чытаюць адвольны `SmartDubbing.__dict__` і не
валодаюць provider lifecycle.

Адхіленыя варыянты:

- Mixins: меншая пачатковая рызыка, але захоўваюць схаванае coupling праз
  `self` і даюць пераважна касметычны падзел.
- Поўнае перапісванне stage pipeline: чысцейшая новая архітэктура, але
  непрымальная рызыка змяніць кэшы, resume/editor behavior і таймінг.

## Мэтавая структура файлаў

```text
src/dubbing/core/
├── smart_dubbing.py          # facade, orchestration, lifecycle, delegates
└── pipeline/
    ├── __init__.py
    ├── context.py            # PipelineRunContext
    ├── cache_keys.py         # exact cache identities and content fingerprints
    ├── references.py         # segment/speaker/configured reference preparation
    ├── transcription.py      # transcription, isolated tracks, semantic planning
    ├── translation.py        # translation, snapshots, plan validation
    ├── emotions.py           # emotion analysis and emotion cache flow
    ├── synthesis.py          # TTS pools, candidates, resynthesis, raw cache
    ├── audio_assembly.py     # trimming, timing, tempo and final audio assembly
    └── artifacts.py          # subtitles, debug files and final video handoff
```

Новыя файлы ствараюцца толькі калі ў іх пераносіцца адна з названых
адказнасцей. Дадатковыя abstraction layers не дадаюцца.

## Module interfaces і facade mapping

Кожны module мае адзін implementation owner. `SmartDubbing` wrappers застаюцца
адзіным compatibility surface; module callables лічацца internal.

| Owner | Facade methods | Explicit dependencies | Return/mutation contract |
|---|---|---|---|
| `smart_dubbing.py` | `__init__`, усе `run_*`, `generate_diarization_report`, `_handle_debug_diarize_only`, `_cleanup`, `_get_torch_device`, `_apply_speaker_filter`, `_format_component_init_error` | config і ўсе initialized collaborators | Захоўвае stage order, facade attributes і cleanup behavior |
| `context.py` | `_validate_plan_dependent_segments` | `PipelineRunContext`, segment list | Validation не мутуе segments; context snapshot/commit абнаўляе толькі пяць frozen fields і facade mirrors |
| `cache_keys.py` | `_cache_fingerprint`, `_effective_tts_cache_fingerprint`, `_file_content_identity`, `_shared_audio_transcription_identity`, `_effective_translation_cache_dimensions`, `_build_dubbing_text_snapshot_key`, `_build_translation_cache_key`, `_build_emotions_cache_key`, `_isolated_tracks_cache_key`, `_isolated_tracks_raw_cache_key`, `_semantic_classifier_identity`, `_tts_selection_cache_fingerprint`, `_segment_cache_metadata_path`, `_raw_tts_segment_cache_key` | explicit config values, cache manager identity helpers, context fingerprint, profiles/segment payload | Вяртае дакладна тыя ж strings/paths/dicts; не мутуе facade ці payload |
| `references.py` | `_attach_segment_reference`, `_canonical_segment_index`, `_segment_reference_artifact_paths`, `_segment_reference_error`, `_prepare_segment_reference`, `_resolve_segment_reference` | config, profile, speaker/audio paths, decoded-audio cache | Мутуе толькі перададзены `tts_segment_data_args` так, як цяпер; вяртае той жа tuple; тыя ж `ValueError` boundaries |
| `transcription.py` | `_initialize_transcriber`, `_require_transcriber`, `_load_cached_diarize_and_transcribe`, `diarize_and_transcribe`, `_semantic_classifier`, `_write_semantic_boundary_diagnostics`, `_diarize_and_transcribe_isolated`, `_isolated_inner_kwargs`, `_restore_semantic_plan_fingerprint` | config, cache manager, transcriber/factory, translator classifier, processors, tracker, context, debug/artifact callbacks | Вяртае той жа `(speakers_rolls, segments)`; segment dictionaries і context fingerprint/persistability мутуюцца ў тым жа парадку |
| `translation.py` | `_initialize_translator`, `_require_translator`, `translate_segments`, `_build_translation_prompt_prefix`, `_persist_dubbing_text_snapshot`, `_persist_synthesis_results` | config, translator/factory, cache manager, context, debug paths | Захоўвае exact segment list/pickle mutation і persistence order; тыя ж initialization/runtime exceptions |
| `emotions.py` | `analyze_emotions`, `_analyze_emotions_gemini`, `_analyze_emotions_speechbrain` | config, cache manager, context, audio path | Мутуе перададзеныя segment dictionaries in place; вяртае той жа list; provider failures апрацоўваюцца як цяпер |
| `synthesis.py` | `_default_tts_system`, `_resolve_voice_profile`, `_profile_pool_key`, `_global_omnivoice_kwargs`, `_build_tts_client`, `_initialize_tts_systems`, `synthesize_speech`, `_synthesize_measured_candidates`, `_load_or_synthesize_candidate`, `resynthesize_one_segment`, `_get_tts_system_for_speaker`, `_preflight_tts_pools`, `_cache_raw_tts_segment`, `_cached_segment_metadata`, `_cached_segment_contract` | config, cache manager, profiles, TTS clients/factory, reference service, assembler callbacks, paths, context | Захоўвае pool/candidate order, in-place metadata, chunk names, retry/fallback і exceptions |
| `audio_assembly.py` | `_trim_trailing_silence`, `_measure_raw_tts_for_timing`, `_adjust_and_combine_audio_grouped_legacy`, `_adjust_and_combine_audio_grouped`, `rebuild_translated_audio_from_chunks` | config, chunk paths, cache manager, timing source з context | Захоўвае in-place chronological reorder, metadata injection, final duration і diagnostic payload |
| `artifacts.py` | `_prepare_audio_inputs`, `_load_required_cached_step`, `_build_speaker_rolls_from_segments`, `_save_requested_subtitles`, `_combine_final_video`, `_reset_input_cache`, `_save_transcription_file`, `_get_subtitle_path`, `adjust_subtitle_timestamps` | config, processors, cache manager, subtitle/debug/video helpers | Захоўвае exact paths, formats, return values, deletion scope і error messages |

Module objects ствараюцца facade wrapper-ам з актуальных attributes у момант
выкліку. Яны не кэшуюць copies dependencies паміж выклікамі. Гэта захоўвае
`__new__` partial construction і monkeypatch facade attributes.

Усе facade signatures ніжэй frozen. Рэфактарынг не мяняе parameters, defaults,
keyword-only markers, return annotations або descriptor type. Golden
`inspect.signature`/descriptor test фіксуе гэты inventory перад extraction:

```text
__init__(self, config: DubbingConfig)
_get_torch_device(self, device_str: Optional[str] = None) -> torch.device
_apply_speaker_filter(self, segments: List[Dict]) -> List[Dict]
_initialize_translator(self) -> None
_default_tts_system(self) -> str
_resolve_voice_profile(self, speaker: str) -> VoiceProfile
_profile_pool_key(self, profile: VoiceProfile) -> tuple
_global_omnivoice_kwargs(self) -> Dict[str, Any]
_build_tts_client(self, profile: VoiceProfile) -> Any
_initialize_tts_systems(self) -> None
_initialize_transcriber(self) -> None
_format_component_init_error(self, component_name: str, configured_backend: str, init_error: Optional[Exception]) -> str
_require_transcriber(self)
_require_translator(self)
_attach_segment_reference(self, *, tts_segment_data_args: Dict[str, Any], segment_dict: Dict[str, Any], speaker: str, segment_index: int, original_audio_segment: Optional[AudioSegment], segment_reference_min_duration: float, segment_reference_min_duration_ms: int) -> tuple[Dict[str, Any], Optional[AudioSegment]]
_canonical_segment_index(segment_dict: Dict[str, Any], chronological_index: int) -> int [staticmethod]
_segment_reference_artifact_paths(self, processed_source_path: Optional[str] = None) -> tuple[Path, Path]
_segment_reference_error(speaker: str, segment_index: int, source_path: Path, reason: str) -> ValueError [staticmethod]
_prepare_segment_reference(self, *, segment_dict: Dict[str, Any], speaker: str, chronological_index: int, reuse_existing: bool, processed_source_path: Optional[str] = None, decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None) -> tuple[str, Optional[str]]
_prepare_audio_inputs(self) -> tuple[str, Optional[str], str]
_cache_fingerprint(dimensions: Dict[str, Any]) -> str [staticmethod]
_effective_tts_cache_fingerprint(self, speakers: Iterable[str]) -> str
_file_content_identity(path: Optional[str]) -> str [staticmethod]
_shared_audio_transcription_identity(self, audio_file: str) -> str
_effective_translation_cache_dimensions(self) -> Dict[str, Any]
_build_dubbing_text_snapshot_key(self, audio_file: str) -> str
_build_translation_cache_key(self, audio_file: str) -> str
_build_emotions_cache_key(self, audio_file: str, segments: List[Dict[str, Any]], provider: Optional[str] = None, model: Optional[str] = None) -> str
_restore_semantic_plan_fingerprint(self, audio_file: str) -> None
_validate_plan_dependent_segments(self, segments: List[Dict[str, Any]]) -> None
_load_required_cached_step(self, *, step_name: str, cache_key: str, hint: str) -> Any
_build_speaker_rolls_from_segments(self, segments: List[Dict]) -> Dict[Tuple[float, float], str]
_save_requested_subtitles(self, segments_for_output: List[Dict], *, save_original_subtitles: bool, save_translated_subtitles: bool, pause_adjustments: Optional[List[Dict[str, float]]] = None) -> None
_combine_final_video(self, *, translated_audio_path: str, background_audio_path: Optional[str], speakers_rolls: Dict[Tuple[float, float], str]) -> tuple[str, List[Dict[str, float]]]
_persist_dubbing_text_snapshot(self, segments: List[Dict], audio_file: str) -> None
_persist_synthesis_results(self, segments: List[Dict], audio_file: str) -> None
_reset_input_cache(self, reason: str) -> None
run_transcribe_only(self, save_original_subtitles: bool = False) -> str
run_translate_only(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
run_analyze_emotions_only(self, save_translated_subtitles: bool = False) -> str
run_from_scratch(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
run_from_tts(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
run_pipeline(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
_handle_debug_diarize_only(self, audio_file: str, speakers_rolls: Dict) -> str
generate_diarization_report(self) -> Tuple[str, str]
_load_cached_diarize_and_transcribe(self, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]
diarize_and_transcribe(self, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]
_isolated_tracks_cache_key(self, audio_file: str, isolated_tracks: Dict[str, str]) -> str
_isolated_tracks_raw_cache_key(self, isolated_tracks: Dict[str, str]) -> str
_semantic_classifier(self) -> Tuple[Optional[Any], str]
_semantic_classifier_identity(self) -> Dict[str, Any]
_write_semantic_boundary_diagnostics(path: str, records: List[Dict[str, Any]]) -> None [staticmethod]
_diarize_and_transcribe_isolated(self, audio_file: str, isolated_tracks: Dict[str, str]) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]
_isolated_inner_kwargs(self, inner_system: str) -> Dict[str, Any]
translate_segments(self, transcription: List[Dict], audio_file: str) -> List[Dict]
_build_translation_prompt_prefix(self, base_prompt_prefix: Optional[str]) -> str
analyze_emotions(self, segments: List[Dict], audio_file: str) -> List[Dict]
_analyze_emotions_gemini(self, segments: List[Dict], audio_file: str, model: str) -> None
_analyze_emotions_speechbrain(self, segments: List[Dict], audio_file: str) -> None
synthesize_speech(self, segments: List[Dict], speakers_rolls: Dict, audio_file: str) -> str
_tts_selection_cache_fingerprint(self, segments: List[Dict[str, Any]]) -> str
_synthesize_measured_candidates(self, metadata: Dict[str, Any], policy) -> None
_load_or_synthesize_candidate(self, metadata: Dict[str, Any], *, variant: str, text: str, attempts: int) -> Optional[Dict[str, Any]]
resynthesize_one_segment(self, segments: List[Dict], segment_index: int, override_text: Optional[str] = None) -> Dict[str, Any]
rebuild_translated_audio_from_chunks(self) -> Optional[str]
_save_transcription_file(self, transcription: List[Dict]) -> None
_cleanup(self) -> None
_get_tts_system_for_speaker(self, speaker_id: str) -> str
_resolve_segment_reference(self, *, tts_segment_data_args: Dict[str, Any], segment_dict: Dict[str, Any], profile: VoiceProfile, provider_capability: str, speaker: str, segment_index: int, original_audio_segment: Optional[AudioSegment], segment_reference_min_duration: float, for_resynthesis: bool = False, processed_source_path: Optional[str] = None, decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None) -> tuple[Dict[str, Any], Optional[AudioSegment]]
_preflight_tts_pools(segments_by_pool: Dict[tuple, List[Any]], clients: Dict[tuple, Any], initial_issues: Optional[List[tuple[int, str]]] = None) -> None [staticmethod]
_trim_trailing_silence(audio_path: str, silence_threshold_db: float = -40.0, keep_tail_ms: int = 100, window_ms: int = 10) -> None [staticmethod]
_measure_raw_tts_for_timing(self, audio_path: str, segment_index: int) -> float
_segment_cache_metadata_path(cache_path: Path) -> Path [staticmethod]
_raw_tts_segment_cache_key(*, base_cache_prefix: str, tts_system: str, segment: Dict[str, Any], speaker: str, translation: str, style_prompt: str, reference_audio_path: Optional[str], reference_mode: Optional[str] = None, reference_text: Optional[str] = None, client_pool_settings: Any = None, legacy_index: Any = 0, emotion: Optional[str] = 'Neutral', tts_prompt_prefix: Optional[str] = None, voice_prompt: Any = None) -> str [staticmethod]
_cache_raw_tts_segment(self, source_path: str, cache_path: Path, *, synthesized_text: str = '') -> None
_cached_segment_metadata(self, cache_path: Path) -> Dict[str, Any]
_cached_segment_contract(self, cache_path: Path) -> str
_adjust_and_combine_audio_grouped_legacy(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]
_adjust_and_combine_audio_grouped(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]
_get_subtitle_path(self, subtitle_type: str, input_path: str, language: str) -> str
adjust_subtitle_timestamps(self, segments: List[Dict], pause_adjustments: List[Dict[str, float]]) -> List[Dict]
```

`voice_prompt` застаецца ў frozen signature `_raw_tts_segment_cache_key` толькі
як compatibility parameter; legacy config больш не падае ў яго значэнне.

## Data flow

```text
Gradio / programmatic caller
            |
            v
core.runner.run_dubbing_job
            |
            v
SmartDubbing facade
  | config, providers, processors, lifecycle
  | PipelineRunContext
  |
  +--> transcription service --> speakers_rolls, mutable segment list
  +--> translation service   --> same mutable segment records
  +--> emotion service       --> same mutable segment records
  +--> reference service     --> resolved/prepared reference artifacts
  +--> synthesis service     --> chosen chunks + segment metadata updates
  +--> audio assembly        --> translated audio + timing diagnostics
  +--> artifact service      --> subtitles/debug/final video
```

`full_pipeline`, `from_scratch`, `transcribe_only`, `translate_only`,
`analyze_emotions_only`, `tts_to_end`, `combine_video` і speaker-report paths
выкарыстоўваюць тыя ж сэрвісы і тыя ж cache/artifact contracts.

## Кіраванне станам

`PipelineRunContext` мае роўна пяць палёў, якія цяпер узнікаюць дынамічна і
залежаць ад call order:

| Field | Default/absence semantics | Writer | Readers |
|---|---|---|---|
| `semantic_plan_fingerprint: Optional[str]` | `None`, як існы `getattr(..., None)` | transcription plan/load/restore | cache keys, translation/emotion/TTS validation |
| `semantic_plan_cache_persistable: bool` | `True`, як існыя persistence defaults | transcription planner/load | snapshot/translation/TTS persistence |
| `plan_dependent_cache_allowed: bool` | `True`; synthesis перазапісвае з persistable перад першым use | synthesis entry | raw/final TTS cache reads/writes |
| `timing_source_audio_file: Optional[str]` | `None`, як існы assembler fallback | synthesis/rebuild entry | audio assembly |
| `timing_source_duration: Optional[float]` | `None`, assembler тады вымярае file | synthesis/rebuild entry | timing plan/audio assembly |

`debug_data`, `real_segment_positions`, `pause_adjustments`, client pools і
initialization errors застаюцца facade-owned: яны ўжо initialized у `__init__`,
маюць асобныя public/test seams і не з'яўляюцца схаваным dynamic protocol.

Context ствараецца адзін раз на ўваходзе кожнага top-level `run_*` метаду з
актуальных facade attributes і перадаецца ўсім stages гэтага run. Ён не
reset-іць існыя значэнні пры паўторным выкарыстанні facade, бо гэта змяніла б
бягучую reuse semantics. Direct helper wrapper стварае кароткі context snapshot
з тых жа attributes.

Пасля кожнага stage толькі палі, якія stage змяніў, запісваюцца назад у
аднайменныя facade attributes. Direct assignment/monkeypatch facade attribute
перад наступным wrapper call заўсёды мае прыярытэт пры стварэнні snapshot.
Partial `__new__` object выкарыстоўвае таблічныя defaults толькі для гэтых
пяці палёў; адсутнасць іншых facade dependencies працягвае даваць той жа
`AttributeError`/runtime failure, а не аўтаматычнае стварэнне collaborators.

Provider clients, processors, cache manager, trackers і cleanup lifecycle
застаюцца ў facade. Сэрвісы не закрываюць і не пераствараюць іх самастойна.

## Semantic-plan і cache ownership

- `transcription.py` — адзіны writer semantic plan fingerprint і
  persistability. Ён стварае fingerprint пры planning, аднаўляе яго з
  `isolated_tracks_semantic_plan` cache і запісвае ў context і segments.
- `context.py` валідуюць, што plan-dependent segment payload мае fingerprint,
  роўны context. Ён не стварае і не персістуе fingerprint.
- `cache_keys.py` — read-only consumer context/segment fingerprint. Ён толькі
  будуе frozen keys і не загружае/захоўвае payload.
- `translation.py` захоўвае fingerprint у нязменным segment payload, правярае
  one-to-one propagation і персістуе translation/snapshot пасля context
  validation.
- `emotions.py` і `synthesis.py` спачатку валідуюць payload праз context helper,
  затым выкарыстоўваюць fingerprint у keys. Яны не могуць замяніць яго.
- Resume paths спачатку выклікаюць transcription-owned restore, потым
  downstream cache lookup. Artifact service fingerprint не чытае.

## Segment mutation contract

Segment payload застаецца `list[dict]`. Рэфактарынг захоўвае:

- in-place замену парадку праз `segments[:]`;
- stable original/canonical indexes;
- дакладныя `_timing_*`, cache-contract, selected-variant, synthesized-text,
  duration і file metadata keys;
- парадак mutation і persistence;
- структуру pickle payload, якую выкарыстоўвае Dubbing Texts editor;
- chunk numbering і сувязь з `audio_chunks/<index>.wav`.

Сэрвісы не вяртаюць новыя copies там, дзе цяпер змяняецца caller-owned list.

## Compatibility facade

`SmartDubbing` захоўвае runtime і de-facto API, уключаючы direct Gradio/test
helpers. Forwarding methods маюць тыя ж імёны, signatures і descriptor type
(instance/static method). Яны збіраюць актуальныя залежнасці пры выкліку, таму
падмена facade attributes пасля `__new__` працягвае працаваць.

Implementation functions не прывязваюцца так, каб monkeypatch на
`SmartDubbing` перастаў уплываць на nested helper calls.

## CLI removal

Выдаляюцца:

- `dubblm_cli.py`;
- `src/dubbing/cli/`;
- `dubblm = "dubbing.cli.main:main"` з `pyproject.toml`;
- argparse-only construction з `core/config.py`;
- CLI-only references у README, launch docs і repository guidance;
- CLI-only tests.

Не выдаляюцца:

- `core/runner.py`;
- `build_config_from_overrides`, `run_dubbing_job` і streaming runner;
- Gradio `dubblm-gradio` entry point;
- run-step functionality, якую выкарыстоўваюць runner і Gradio.

## Legacy TTS configuration removal

Адзіная per-speaker TTS schema пасля рэфактарынгу — `voices:` з
`VoiceProfile`.

Выдаляецца падтрымка:

- `tts_system_mapping`;
- `voice_prompt`;
- `reference_audio_mapping`;
- `reference_text_mapping`;
- dict-shaped legacy `voice_name`;
- адпаведных `SmartDubbing` mirrors;
- legacy normalization і deprecation flow у `voice_profiles.py`;
- legacy UI/runner/config parsing, docs і tests.

`voice_profiles.py` вызначае адзін exception:

```python
class LegacyVoiceConfigError(ValueError):
    pass
```

і адзін validation entry point:

```python
reject_legacy_voice_config(config: Mapping[str, Any], *, source: str) -> None
```

Ён правярае top-level presence `tts_system_mapping`, `voice_prompt`,
`reference_audio_mapping`, `reference_text_mapping`, а таксама dict-shaped або
старую comma/colon mapping-form `voice_name`. Scalar top-level `voice_name`
застаецца дапушчальным global default; `voice_name` унутры `voices:` таксама
дапушчальны. Nested modern profile fields не лічацца legacy.

Памылка мае стабільны template:

```text
Legacy per-speaker TTS setting '<key>' is no longer supported in <source>;
migrate speaker configuration to 'voices:'.
```

Validation запускаецца на ўсіх ingress boundaries:

1. у `DubbingConfig.load_from_yaml` на raw YAML mapping да merge;
2. у `build_config_from_overrides` на raw programmatic/Gradio overrides;
3. у `DubbingConfig.process_special_parameters` пасля merge як safety net;
4. у `normalize_voices` для direct library use;
5. у `SmartDubbing.__init__` перад initialization для plain dict-like config і
   direct construction.

`LegacyVoiceConfigError` не ператвараецца ў `TypeError` і не ігнаруецца.
`run_dubbing_job` захоўвае свой існы boundary: ён вяртае UI/programmatic failed
result з гэтым паведамленнем; direct config/normalizer/constructor calls
атрымліваюць exception.

Provider-level TTS interfaces, wrapper methods і factory contracts яўна не
выдаляюцца ў гэтым рэфактарынгу, нават калі repository search не знаходзіць
call site. Dead-code removal абмежаваны private/internal кодам target modules,
узгодненым CLI і дакладна названай legacy config schema.

## Dead-code policy

Кандыдат выдаляецца толькі пасля:

1. repository-wide search усіх imports, calls і attribute access;
2. праверкі dynamic/Gradio/programmatic paths;
3. characterization test, калі observable behavior магчыма;
4. focused і full regression run.

Вядомыя кандыдаты:

- unreachable historical body пасля delegate у
  `_adjust_and_combine_audio_grouped_legacy`;
- дубляваны `warnings.filterwarnings("ignore")` call;
- CLI і ўзгоднены legacy config compatibility;
- compatibility mirrors, якія існавалі толькі для выдаленай legacy schema.

Сам compatibility method можа застацца thin delegate, калі яго яшчэ выклікае
Gradio, праграмны код або тэст. Import-time environment/warning behavior не
змяняецца, акрамя выдалення дакладнага дубля, бо гэта асобная behavioral
cleanup задача.

## Cache і artifact contracts

Наступныя значэнні лічацца frozen:

- cache step names, prefixes, schema/algorithm versions;
- canonical JSON options, digest algorithm і truncation;
- усе dimensions і effective default resolution;
- semantic-plan fingerprint propagation і validation;
- `.pkl`, `.json`, `.wav`, TSV і subtitle payload formats;
- project/artifact directory layout;
- reference і chunk file naming;
- source/vocals/isolated-track selection order.

`pipeline/cache_keys.py` і `pipeline/references.py` выносяцца першымі, каб
наступныя сэрвісы залежалі ад ужо frozen contracts.

## Error handling і cleanup

- Існыя exception types, actionable messages і fail-fast boundaries
  захоўваюцца.
- Deferred translator/transcriber/TTS initialization failures застаюцца
  deferred у тых жа runtime paths.
- Сэрвісы не дадаюць broad catches і не ператвараюць памылкі ў `None`.
- Facade захоўвае існы `try/finally` cleanup order.
- Памылка legacy config validation — адзіная новая ўзгодненая памылка.

Pre-existing bugs з baseline review не выпраўляюцца ў гэтым scope:

- fallback на default TTS client пасля failure requested pool;
- магчымы свежы transcriber call у некаторых `translate_only` compatibility
  paths;
- небяспечныя speaker labels у file paths;
- non-atomic pickle/metadata writes;
- паўторнае поўнае hashing reference files;
- няпоўная/неаднастайная cleanup semantics;
- не exception-safe temporary debug config mutation;
- destructive legacy cache reset scope;
- глабальныя import-time environment/warning mutations.

Яны трапляюць у final review report як deferred findings, але не ў diff.

## TDD і verification

Кожны extraction task праходзіць Red-Green-Refactor:

1. Дадаць failing test для новага module API або compatibility delegate.
2. Запусціць яго і пацвердзіць чаканую failure прычыну.
3. Перанесці мінімальны блок без змены алгарытму.
4. Запусціць focused tests.
5. Запусціць звязаныя regression modules.
6. Пасля буйнога блока запусціць поўны suite.

Characterization coverage абавязкова фіксуе:

- exact cache keys і artifact paths;
- context propagation праз full, isolated, resume і editor paths;
- in-place segment mutations і cache payload shape;
- partial `__new__` facade construction;
- private/static compatibility descriptors і signatures;
- reference selection/preflight;
- TTS candidate order, raw cache і resynthesis;
- timing/assembly diagnostics і final duration;
- runner/Gradio run-step behavior;
- fail-fast legacy migration error;
- адсутнасць CLI entry points.

CLI-only tests выдаляюцца. Legacy-success tests замяняюцца modern `voices:`
coverage і fail-fast migration tests. Таму фінальная колькасць тэстаў не
абавязана быць роўнай 346; усе захаваныя і новыя тэсты павінны праходзіць.

## Code review

1. Baseline review вызначае hidden coupling, state і frozen contracts.
2. Прамежкавы review праводзіцца пасля кожнага буйнога extraction блока.
3. Final independent review параўноўвае поўны diff са spec і правярае scope,
   architecture, tests, compatibility і dead-code evidence.
4. Усе Critical і Important заўвагі, звязаныя з узгодненым scope,
   выпраўляюцца да завяршэння.
5. Pre-existing unrelated findings дакументуюцца і не выпраўляюцца.

## Парадак extraction

1. Замарозіць compatibility, state, cache і artifact contracts тэстамі.
2. Дадаць `PipelineRunContext` і facade mirrors.
3. Вынесці cache-key utilities.
4. Вынесці reference resolution/preparation.
5. Вынесці transcription і semantic planning orchestration.
6. Вынесці translation і emotion flows.
7. Вынесці TTS pools, measured candidates і resynthesis.
8. Вынесці timing/audio assembly.
9. Вынесці artifact/subtitle/final-video helpers.
10. Выдаліць CLI, legacy config compatibility і доказна мёртвы код.
11. Правесці final review і поўную verification.

## Acceptance criteria

- `SmartDubbing` застаецца thin facade па ранейшым import path.
- Runtime/Gradio/programmatic функцыянал працуе без змяненняў.
- Усе run-step і editor/resynthesis paths захаваныя.
- Cache identities, payloads і artifact layout не змяніліся.
- Segment mutation contract захаваны.
- CLI entry points і argparse-only code адсутнічаюць.
- Legacy per-speaker keys fail fast з migration error.
- Сучасны `voices:`/`VoiceProfile` flow праходзіць усе tests.
- Доказна мёртвы код выдалены; патрэбныя delegates захаваныя.
- Няма unrelated code changes або pre-existing bug fixes.
- Focused, regression і full test suites праходзяць.
- Final code review не мае незакрытых Critical/Important заўваг у scope.
