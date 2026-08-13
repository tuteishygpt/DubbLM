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

`PipelineRunContext` робіць яўнымі палі, якія цяпер узнікаюць дынамічна і
залежаць ад call order:

- semantic plan fingerprint;
- semantic-plan cache persistence flag;
- plan-dependent cache permission;
- timing source audio path і duration;
- іншыя выключна per-run flags, якія ўжо існуюць у `SmartDubbing`.

`SmartDubbing` валодае context. Існыя facade attributes/properties застаюцца
compatibility mirrors, каб не зламаць direct helpers, monkeypatching і
часткова створаныя праз `SmartDubbing.__new__` аб'екты.

Provider clients, processors, cache manager, trackers і cleanup lifecycle
застаюцца ў facade. Сэрвісы не закрываюць і не пераствараюць іх самастойна.

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

Калі любы legacy key прысутнічае ў загружанай канфігурацыі, validation
спыняецца з actionable error, які называе ключ і патрабуе міграцыі ў `voices:`.
Палі не ігнаруюцца ціха.

Provider-level APIs выдаляюцца толькі калі reference search па ўсім
рэпазіторыі і тэсты даказваюць, што яны не патрэбныя сучаснаму
`TTSSegmentData` flow.

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
