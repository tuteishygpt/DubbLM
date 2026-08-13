# DubbLM: фактычная логіка працы праекта па коду

## Мэта і межы

Гэты дакумент апісвае, як DubbLM фактычна працуе ў runtime паводле бягучага кода. Гэта не прадуктовы агляд і не спецыфікацыя жаданага паводзінаў. Тут зафіксавана:

- як ідзе end-to-end пайплайн дубляжу;
- якія backend-ы падтрымліваюцца на кожным этапе;
- як прымаюцца ключавыя рашэнні ў runtime;
- якія artifacts і cache ствараюцца;
- што азначаюць палі інтэрфейсу і асноўныя налады.

Дакумент арыентаваны на распрацоўшчыка. Код не цытуецца, але апісанне прывязана да рэальных модуляў і патоку выканання.

## End-to-end pipeline

У нармальным рэжыме праект праходзіць наступныя этапы:

1. Загрузка `.env`, YAML-канфігу і CLI/UI overrides.
2. Валідацыя ўваходных параметраў і стварэнне project-specific каталогаў.
3. Вылучэнне аўдыя з відэа.
4. Апцыянальнае падзяленне аўдыя на background і vocals.
5. Дыярызацыя спікераў і транскрыпцыя.
6. Вылучэнне эталонных аўдыякліпаў па спікерах.
7. Пераклад з кантэкстным аналізам, chunking і refinement.
8. Апцыянальны аналіз эмоцый.
9. Падбор рэферэнсаў і/або галасоў для TTS.
10. Ацэнка працягласці розных тэкставых версій і выбар лепшага варыянту.
11. Сінтэз асобных сегментаў.
12. Групавая зборка адной агульнай дубляванай аўдыядарожкі.
13. Захаванне translated samples і debug-артыфактаў.
14. Зборка фінальнага відэа з новай аўдыядарожкай.
15. Апцыянальнае скарочванне доўгіх паўз у відэа і карэкцыя subtitle timestamps.

Акрамя поўнага пайплайна ёсць два спецыяльныя рэжымы:

- `generate_speaker_report`: спыняецца пасля дыярызацыі і падрыхтоўкі сэмплаў спікераў.
- `run_step=combine_video`: прапускае ўсе папярэднія крокі і толькі збірае відэа з ужо існуючых artifacts.

## Этап 1. Config і праектныя дырэкторыі

### Крыніцы налад

Налады збіраюцца з трох крыніц:

- defaults у runtime config;
- YAML-файл, па змаўчанні `dubbing_config.yml`;
- CLI або UI overrides.

Прыярытэт стандартны: overrides перакрываюць YAML, YAML перакрывае defaults.

### Што робіцца пры валідацыі

Падчас валідацыі праект:

- правярае, што зададзены `input`, `source_language`, `target_language`;
- правярае існаванне ўваходнага відэафайла;
- стварае асобную працоўную папку побач з уваходным відэа з імем `input_stem`;
- унутры яе стварае `artifacts/` і набор стандартных падкаталогаў.

### Асноўныя runtime-шляхі

Для кожнага ўваходнага відэа ствараюцца:

- `project_dir`: каранёвая папка канкрэтнага запуску;
- `artifacts_dir`: усе тэхнічныя прамежкавыя файлы;
- `audio_artifacts_dir`: агульныя аўдыяфайлы пайплайна;
- `speakers_audio_dir`: эталонныя WAV па спікерах;
- `audio_chunks_dir`: вынікі сінтэзу па сегментах;
- `su_audio_chunks_dir`: тэмпавыя або speed-adjusted файлы;
- `debug_dir`: TSV, debug video, translation logs;
- `translated_samples_dir`: параўнальныя original/translated samples;
- `transcription_path`, `timecodes_report_path`, `translated_audio_path`, `background_audio_path`.

### Асаблівасці нармалізацыі параметраў

У config-слоі таксама:

- парсяцца JSON-падобныя мапінгі;
- пераўтвараюцца `keep_original_audio_ranges` у спіс часавых інтэрвалаў;
- абразаюцца да дапушчальных межаў `group_overflow_tolerance`;
- валідуецца `segment_reference_min_duration`;
- аўтаматычна генеруецца `output`, калі ён не зададзены.

## Этап 2. Audio extraction і background separation

### Вылучэнне аўдыя

Праект заўсёды пачынае з падрыхтоўкі `source.wav`.

Паводзіны:

- калі `start_time` і/або `duration` зададзены, з відэа выразаецца толькі патрэбны фрагмент;
- калі не, выцягваецца аўдыя з усяго файла;
- разам з гэтым вызначаецца `total_duration`, якое выкарыстоўваецца пазней у зборцы і метрыках.

### Захаванне background

Калі ўключаны `keep_background`, праект спрабуе падзяліць `source.wav` на:

- `background.wav`;
- `vocals.wav`.

Калі падзел удалы:

- фонавая дарожка можа быць дамяшаная назад пры фінальнай зборцы;
- `vocals.wav` выкарыстоўваецца як больш чыстая крыніца для segment-specific reference clips.

Калі optional dependency для separation не ўсталявана, праект лагуе папярэджанне і працягвае без background separation.

## Этап 3. Transcription і diarization

### Factory і падтрыманыя backend-ы

Transcription layer выбіраецца праз factory. Код падтрымлівае:

- `whisper`
- `openai`
- `pyannote_openai`
- `whisperx`
- `assemblyai`
- `gemini`

У UI-dropdown `pyannote_openai` не паказаны як асобны choice, але backend у кодзе існуе.

### Агульны выхад этапу

Кожны transcriber павінен вярнуць:

- `speakers_rolls`: адпаведнасць інтэрвалу часу і спікера;
- `transcription`: спіс сегментаў з `start`, `end`, `speaker`, `text`.

### Варыянт `gemini`

Gemini backend:

- загружае аўдыя ў Gemini Files;
- просіць адразу зрабіць і транскрыпцыю, і speaker diarization;
- чакае JSON з храналагічнымі сегментамі;
- нармалізуе speaker labels у фармат `SPEAKER_00`, `SPEAKER_01` і г.д.

Гэта самы просты па архітэктуры варыянт: уся лагіка распазнавання вынесена ў знешні API.

### Варыянт `pyannote_openai` / `openai` / `whisper`

Гэты backend значна больш складаны:

1. Аўдыя рэжацца на доўгія чанкі прыблізна па 15 хвілін, пажадана па silent points.
2. Для кожнага чанка запускаецца speaker diarization.
3. Кожны speaker segment асобна транскрыбуецца праз Whisper або OpenAI.
4. Таймкоды з сегментаў зрушваюцца назад у глабальную шкалу часу.
5. Паміж чанкамі робіцца speaker matching праз эмбедынгі.
6. Пасля matching спікеры пераймяноўваюцца па долі агульнага speaking time.

Гэта дае большы кантроль над сегментацыяй, але і большую складанасць.

### Іншыя backend-ы

- `whisperx`: асобны transcriber, прызначаны для WhisperX-патоку.
- `assemblyai`: асобны wrapper вакол AssemblyAI.

Аркестратар чакае ад усіх аднолькавую структуру выхаду, таму астатнія крокі пайплайна не залежаць ад канкрэтнага backend-а транскрыпцыі.

### Спецыяльны рэжым report/debug

На гэтым этапе могуць актывавацца:

- `generate_speaker_report`: спыняе pipeline пасля дыярызацыі і speaker samples;
- `debug_diarize_only`: генеруе debug artifacts і выходзіць без перакладу і TTS.

## Этап 4. Translation і refinement

### Агульная архітэктура

Пераклад у праекце двухступенчаты:

1. Initial translation.
2. Global refinement.

Перакладчык у кодзе толькі адзін тыпу `llm`, але ён падтрымлівае розныя LLM provider-ы:

- `gemini`
- `openrouter`

### Што робіцца перад перакладам

Да пачатку перакладу праект:

- аналізуе ўвесь транскрыпт;
- вызначае domain, tone, terminology, themes;
- генеруе overall summary і chapters;
- складае text summary, якое будзе выкарыстоўвацца як кантэкст для далейшых prompt-аў;
- піша human-readable `timecodes.txt`.

### Аптымізацыя і chunking

Перад LLM translation сегменты:

- першапачаткова злепліваюцца, калі гэта суседнія рэплікі аднаго спікера і яны блізкія па часе;
- затым разбіваюцца на translation chunks.

Chunking арыентуецца на:

- максімальны памер чанка;
- колькасць пераключэнняў паміж спікерамі.

Мэта chunking-а: даць LLM дастатковы кантэкст, але не рабіць prompt занадта вялікім і хаатычным.

### Initial translation

Для кожнага чанка LLM атрымлівае:

- агульны summary;
- context before;
- сам chunk;
- context after;
- domain/tone/terms;
- glossary, калі ён зададзены;
- custom translation prompt prefix, калі ён зададзены.

Ад LLM чакаецца JSON з такім жа наборам радкоў і тымі ж speaker IDs.

Калі source і target language супадаюць, translation stage можа быць прапушчаны, а тэкст проста пераносіцца далей у refinement.

### Retry, validation, split-on-failure

Initial translation валідуюцца па наступных крытэрыях:

- колькасць радкоў павінна супадаць;
- парадак спікераў павінен супадаць;
- адказ павінен быць JSON, які можна разабраць.

Калі пераклад не праходзіць валідацыю:

- робяцца retry;
- пры працяглым фэйлe chunk дзеліцца напалам і перакладаецца рэкурсіўна.

### Refinement

Пасля initial translation праект робіць refinement ужо не для аднаго радка, а для групы chunk-аў.

Refinement:

- выкарыстоўвае summary усяго дыялогу;
- бачыць original conversation і ўжо перакладзены conversation;
- можа бачыць суседні кантэкст;
- выкарыстоўвае persona-specific prompt.

### Refinement personas

UI паказвае:

- `normal`
- `casual_manager`
- `child`
- `housewife`
- `science_popularizer`
- `it_buddy`
- `ai_buddy`

Код prompt-аў падтрымлівае набор persona templates, і менавіта яны вызначаюць стыль, кампактнасць, тэрміналогію і чаканыя alternative versions.

### Вынікі refinement

На выхадзе refinement кожны радок можа мець:

- `text`: асноўны refined translation;
- `very_short`: вельмі сціслы варыянт;
- `short`: скарочаны варыянт;
- `long`: пашыраны варыянт.

Пасля гэтага праект раскідвае refined data назад на segment-level.

### Канчатковая segment-level структура

На ўзроўні асобнага сегмента далей выкарыстоўваюцца:

- `translation`
- `very_short_translation`
- `short_translation`
- `long_translation`

Менавіта гэтыя палі потым выкарыстоўваюцца пры падборы лепшага тэксту пад TTS timing.

## Этап 5. Voice/reference selection

### Два прынцыпова розныя механізмы

У праекце ёсць два незалежныя сцэнары:

1. Выбар рэферэнснага аўдыя для systems з voice cloning або reference-based synthesis.
2. Падбор гатовага голасу з каталога для systems з voice matching.

### Прыярытэт рэферэнсных крыніц

Для кожнага сегмента reference audio выбіраецца ў такім парадку:

1. `reference_audio_mapping` і `reference_text_mapping`, калі зададзены ўручную.
2. Агульны WAV спікера з `speakers_audio_dir`.
3. Segment-specific reference clip, выразаны з арыгінальнага аўдыя, калі сегмент дастаткова доўгі.

`segment_reference_min_duration` кіруе мінімальнай працягласцю сегмента, каб з яго дазволена было зрабіць асобны reference clip.

### Speaker samples

Пасля дыярызацыі праект збірае для кожнага спікера поўны эталонны WAV:

- усе кавалкі гэтага спікера склейваюцца;
- вынік абразаецца прыкладна да 1 хвіліны.

Гэтыя файлы выкарыстоўваюцца як stable speaker references і як матэрыял для report/debug.

### Voice matching для OpenAI/Gemini

Для backend-аў з каталогам фіксаваных галасоў:

- з reference audio бяруцца некалькі фрагментаў;
- з іх лічацца эмбедынгі;
- эмбедынгі параўноўваюцца з загадзя падрыхтаванымі sample embeddings даступных галасоў;
- па voting выбіраецца найбольш падобны voice.

### Reference-based synthesis

Для backend-аў кшталту OmniVoice, XTTS, F5, BexTTS больш важны не choice з каталога, а наяўнасць валіднага reference audio, а часам яшчэ і reference text.

## Этап 6. TTS duration estimation і выбар версіі тэксту

### Што спрабуе зрабіць праект

Перад сінтэзам праект хоча выбраць такі варыянт тэксту, які максімальна добра ўпішацца ў арыгінальны таймінг сегмента.

Для кожнага сегмента ён:

1. Бярэ `translation`.
2. Ацэньвае, колькі прыкладна будзе гучаць гэты тэкст у абраным TTS backend-е.
3. Параўноўвае гэта з арыгінальнай працягласцю сегмента.
4. Калі асноўны варыянт відавочна занадта кароткі або занадта доўгі, спрабуе `very_short`, `short` або `long`.
5. Выбірае тэкст з найменшай deviation ад камфортнага інтэрвалу.

### Comfort zone

Параўнанне робіцца праз суадносіны:

- `original_duration / estimated_tts_duration`

Для выбару лічыцца, што прымальная зона:

- не менш за `0.75`;
- не больш за `1.15`.

Інтэрпрэтацыя:

- калі ацэнены TTS доўжыцца занадта доўга, праект спрабуе больш кароткія версіі;
- калі ён занадта кароткі, спрабуе больш доўгі варыянт.

### Як ацэньваецца працягласць у залежнасці ад backend-а

#### OmniVoice

Ацэнка простая і эврыстычная:

- колькасць слоў;
- фіксаваны множнік секунд на слова;
- карэкцыя на `speed`.

#### OpenAI TTS

Ацэнка больш дарагая, але дакладная:

- wrapper генеруе пробны аўдыёфайл;
- вымярае фактычную працягласць;
- кэшуе вынік.

Пры памылцы генерацыі ёсць fallback-эврыстыка па словах і base WPM.

#### Gemini TTS

Ацэнка заснавана на статыстыцы па канкрэтным voice:

- words per minute;
- characters per second;
- complexity factor па пунктуацыі, даўжыні слоў і структуры сказаў.

Гэтая статыстыка генеруецца загадзя на sample texts і захоўваецца ў duration database.

#### Іншыя backend-ы

Іншыя TTS wrappers таксама павінны рэалізоўваць уласную ацэнку, але ўзровень дакладнасці розны: ад простай эврыстыкі да больш складзеных rule-based схем.

### Праверка пасля фактычнага synthesis

Duration estimation выкарыстоўваецца толькі для першаснага выбару тэксту. Пасля фактычнага synthesis праект яшчэ раз глядзіць на рэальную даўжыню файла.

Калі яна ўсё яшчэ дрэнна трапляе ў патрэбны інтэрвал:

- праект можа перасінтэзаваць сегмент з іншым тэкставым варыянтам;
- гэта ўжо post-estimation correction, а не толькі папярэдні выбар.

## Этап 7. Audio assembly

### Сінтэз па сегментах

Пасля выбару тэксту праект сінтэзуе асобны WAV для кожнага сегмента.

Падтрымліваецца group-by-TTS-system логіка:

- калі розным спікерам прызначаны розныя TTS systems, сегменты групуюцца адпаведна;
- кожная група сінтэзуецца сваім backend-ам;
- сегментны cache выкарыстоўваецца, каб не перагенераваць аднолькавыя кавалкі.

### Калі сегмент не атрымалася сінтэзаваць

Паводзіны залежыць ад backend-а і месца фэйлу, але агульная ідэя такая:

- лагуецца памылка;
- для segment output можа быць створаны пусты або silent placeholder;
- pipeline па магчымасці не валяцца цалкам на адным сегменце.

### Групаванне і таймінг

Пасля synthesis праект не проста расстаўляе сегменты на timeline па адным.

Ён:

- групуе суседнія сегменты па спікерах;
- захоўвае паўзы паміж імі;
- збірае group audio;
- вылічвае, ці трэба паскорыць або запаволіць усю групу;
- карэктуе тэмп праз FFmpeg;
- абмяжоўвае overflow па `group_overflow_tolerance`;
- потым змешвае speaker tracks у адну фінальную дарожку.

Такі падыход лепш захоўвае натуральнасць унутры бесперапыннай рэплікі аднаго спікера.

### Real segment positions

Пасля group-level adjustments праект захоўвае рэальныя пазіцыі сегментаў у фінальнай аўдыядарожцы. Гэта патрэбна для:

- debug;
- subtitle correction;
- аналізу наступных этапаў.

## Этап 8. Video muxing, pause removal, subtitles

### Базавая зборка

Фінальная зборка відэа ўключае:

- арыгінальнае відэа;
- новы dubbed audio;
- апцыянальны background audio;
- апцыянальны original audio як другі track;
- watermark image і/або watermark text;
- апцыянальны upscale.

### Калі відэа можна не рээнкодзіць

Калі няма video filters, overlays, pause cuts і іншых змяненняў відэапатоку, праект спрабуе выкарыстаць stream copy для захавання якасці.

Калі ёсць:

- overlay;
- drawtext;
- upscale;
- pause removal;
- trim, які не сумяшчальны з keyframes,

тады ідзе re-encoding.

### Audio mixing

Праект можа:

- проста падставіць dubbed audio;
- дамяшаць background;
- пакінуць selected ranges арыгінальнага аўдыя;
- дадаць original audio як асобны track.

### Pause removal

Калі `remove_pauses=true`, праект:

1. Спачатку робіць фінальны audio mix для аналізу.
2. Шукае доўгія участкі цішыні.
3. Атрымае keyframes арыгінальнага відэа.
4. Не чапае паўзы каля keyframes і на краях відэа.
5. Скарачае толькі дазволеныя паўзы, пакідаючы частку `preserve_pause_duration`.
6. Разлічвае cuts.
7. Рэжа і відэа, і ўсе задзейнічаныя аўдыяфайлы.
8. Запамінае `pause_adjustments` для subtitles.

### Subtitles

Субцітры захоўваюцца ў двух рэжымах:

- без pause removal: з арыгінальнымі timestamps;
- з pause removal: пасля карэкцыі timestamps па `pause_adjustments`.

Таксама пішацца debug TSV з original і translated text па сегментах.

## Палі інтэрфейсу і значэнне налад

Ніжэй апісаны перш за ўсё палі Gradio UI. У канцы секцыі ёсць дадатковыя налады, якія прысутнічаюць у YAML/CLI, але не выведзены ў інтэрфейс.

### Workflow

| Поле | Што значыць |
|---|---|
| `Input video` | Уваходны файл відэа. Абавязковы для поўнага запуску. |
| `Source language` | Код мовы арыгінальнага аўдыя. Ідзе ў transcription і translation. |
| `Target language` | Код мовы дубляжу. Ідзе ў translation, subtitles, metadata і TTS. |
| `Output path` | Яўны шлях для фінальнага файла. Калі пусты, генеруецца аўтаматычна ў project directory. |
| `Config path` | YAML-файл з persisted settings. |
| `Run step` | Спецыяльны рэжым partial pipeline. Зараз фактычна прызначаны для `combine_video`. |
| `Generate speaker report only` | Замест поўнага дубляжу робіць дыярызацыю, сэмплы спікераў і report. |
| `Save original subtitles` | Захаваць SRT на зыходнай мове. |
| `Save translated subtitles` | Захаваць SRT на мове дубляжу. |
| `Keep background audio` | Паспрабаваць аддзяліць фон і дамяшаць яго назад у фінальны mix. |
| `Include original audio in final video` | Дадаць арыгінальную аўдыядарожку ў фінальны файл; пры некаторых сцэнарыях таксама выкарыстоўваецца selective keep ranges. |
| `Remove pauses` | Уключыць этап скарачэння доўгіх паўз у канчатковым відэа. |

### Settings: Transcription

| Поле | Што значыць |
|---|---|
| `Whisper model` | Памер/назва лакальнай Whisper-мадэлі для whisper-based backend-аў. |
| `Gemini transcription model` | Назва Gemini-мадэлі для backend-а `gemini`. |
| `Transcription system` | Які transcriber выкарыстоўваць. |
| `Start time (seconds)` | Пачатак апрацоўкі ўнутры ўваходнага відэа. |
| `Duration (seconds)` | Даўжыня фрагмента для апрацоўкі. Нуль і адмоўныя значэнні фактычна ігнаруюцца. |
| `Disable cache` | Адключае reuse кэшу для асноўных этапаў. |

### Settings: Translation

| Поле | Што значыць |
|---|---|
| `Translator type` | Тып translator-а. Фактычна падтрымліваецца `llm`. |
| `LLM provider` | Backend для initial translation. |
| `LLM model` | Канкрэтная мадэль provider-а для initial translation. |
| `LLM temperature` | Temperature для initial translation. |
| `Translation prompt prefix` | Дадатковы тэкст, які ўстаўляецца ў translation prompts як extra context. |
| `Glossary JSON` | Мапінг тэрмінаў у базавыя пераклады. Праект прымушае LLM выкарыстоўваць гэтыя тэрміны з граматычнай адаптацыяй. |

### Settings: Refinement

| Поле | Што значыць |
|---|---|
| `Refinement LLM provider` | Provider для другога этапу, refinement. |
| `Refinement model` | Мадэль для refinement. |
| `Refinement temperature` | Temperature для refinement-stage. |
| `Refinement max tokens` | Ліміт генерацыі для refinement provider-а, калі ён яго падтрымлівае. |
| `Refinement persona` | Prompt template для перапрацоўкі і стварэння alternative versions. |

### Settings: TTS

| Поле | Што значыць |
|---|---|
| `TTS system` | Галоўны TTS backend. |
| `TTS model` | Назва мадэлі канкрэтнага TTS backend-а, калі ён падтрымлівае гэта як параметр. |
| `Fallback TTS model` | Рэзервовая мадэль, галоўным чынам для Gemini TTS. |
| `Voice name or speaker mapping` | Альбо адзін global voice, альбо speaker-to-voice mapping. Найбольш актуальна для voice-catalog backends. |
| `Automatic voice selection` | Уключае voice matching для backend-аў, якія ўмеюць падбіраць голас па эмбедынгах. |
| `Reference audio path` | Global reference audio для reference-based synthesis. |
| `Reference text` | Тэкст, які адпавядае global reference audio, калі backend яго патрабуе. |
| `Speaker reference mappings` | Табліца ручных per-speaker reference audio/reference text. Мае прыярытэт над auto-generated references. |
| `Library speaker ID` / `Library reference audio file` / `Library reference text` | Інструменты для захавання speaker references у лакальную бібліятэку. |
| `Speaker reference library` | Read-only табліца захаваных speaker references, якія можна падцягнуць назад у mapping. |
| `Enable emotion analysis` | Уключае этап аналізу эмоцый перад synthesis. |
| `Min segment reference duration` | Мінімальная даўжыня сегмента, каб праект мог выразаць з яго асобны reference clip. |
| `TTS system mapping JSON` | Per-speaker назначэнне розных TTS systems. |
| `TTS prompt prefix` | Global prompt prefix для TTS backend-аў, якія падтрымліваюць prompt-driven synthesis. |
| `Voice prompt JSON` | Per-speaker style prompts. |

### Settings: Video / Audio

| Поле | Што значыць |
|---|---|
| `Watermark image path` | Шлях да watermark image. |
| `Watermark text` | Тэкст, які малюецца на відэа. |
| `Keep original audio ranges` | Спіс часавых інтэрвалаў, дзе трэба пакінуць арыгінальны audio ў асноўным mix. |
| `Min pause duration` | Парог, пасля якога цішыня лічыцца кандыдатам на скарочванне. |
| `Keyframe buffer` | Буфер каля keyframes, дзе паўзы не выразаюцца. |
| `Use two-pass encoding` | Уключае двухпраходнае кадзіраванне ў сцэнарыях, дзе гэта дарэчы і не канфліктуе з фільтрамі. |
| `Dubbed volume` | Gain multiplier для translated track. |
| `Background volume` | Gain multiplier для background track пасля separation. |
| `Group overflow tolerance` | Наколькі можна дапусціць выход group audio за межы арыгінальнага акна пры group-level timing adjustment. |

### Settings: Debug / Advanced

| Поле | Што значыць |
|---|---|
| `Debug info` | Уключае захаванне дадатковых debug artifacts і можа генераваць debug video. |
| `Debug TTS` | Уключае дадатковы TTS-debug flow, залежны ад backend-а. |
| `Debug diarize only` | Спыняе pipeline пасля дыярызацыі/transcription і генеруе debug artifacts замест поўнага дубляжу. |

### Дадатковыя YAML/CLI-налады, якія не ўсе бачныя ў UI

| Налада | Што значыць |
|---|---|
| `omnivoice_space_id` | Які Hugging Face Space выкарыстоўваць для OmniVoice. |
| `omnivoice_api_name` | Які Gradio endpoint выклікаць у OmniVoice Space. |
| `omnivoice_lang` | Моўны параметр для OmniVoice synthesis. |
| `omnivoice_num_steps` | Колькасць generation steps у OmniVoice. |
| `omnivoice_guidance_scale` | Guidance scale у OmniVoice. |
| `omnivoice_denoise` | Ці ўключаць denoise у OmniVoice. |
| `omnivoice_speed` | Базавы speed для OmniVoice. |
| `omnivoice_duration` | Базавы duration control для OmniVoice. |
| `omnivoice_preprocess_prompt` | Preprocess prompt у OmniVoice. |
| `omnivoice_postprocess_output` | Postprocess audio output у OmniVoice. |
| `reference_audio_mapping` | Per-speaker mapping рэферэнсных аўдыяфайлаў. У UI прадстаўлена праз dataframe. |
| `reference_text_mapping` | Per-speaker mapping reference texts. |
| `translator_type` | Формальна падтрымлівае выбар translator-а, фактычна ў кодзе рэалізаваны толькі `llm`. |
| `run_step` | Частковы запуск pipeline. |
| `group_overflow_tolerance` | Дапушчальны overflow для group timing. |
| `segment_reference_min_duration` | Мінімальная даўжыня сегмента для segment reference export. |

## Caching і artifacts

### Cache

Праект выкарыстоўвае step-based cache manager з прывязкай да ўваходнага файла. Cache ахоплівае:

- transcription;
- translation;
- emotion analysis;
- background separation;
- speaker audio;
- synthesized full audio;
- segment-level synthesis;
- audio normalization;
- дапаможныя voice stats / embeddings у асобных backend-ах.

### Асноўныя artifacts

Тыповыя artifacts:

- `transcription.txt`
- `timecodes.txt`
- `audio/source.wav`
- `audio/output.wav`
- `audio/background.wav`
- `speakers_audio/*.wav`
- `translated_samples/*`
- `debug/translations.tsv`
- translation/refinement debug logs
- temp файлы для pause removal і re-encoding

## Што фактычна актыўна ў бягучым config

На момант падрыхтоўкі дакумента ў `dubbing_config.yml` уключаны наступны маршрут:

- `transcription_system = gemini`
- `translator_type = llm`
- `llm_provider = gemini`
- `refinement_llm_provider = gemini`
- `tts_system = omnivoice`
- `keep_background = true`
- `remove_pauses = true`
- `include_original_audio = true`
- `save_original_subtitles = true`
- `save_translated_subtitles = true`
- `voice_auto_selection = true`
- `enable_emotion_analysis = false`

Гэта значыць, што фактычны рабочы end-to-end path зараз такі:

1. Gemini diarization + transcription.
2. Gemini initial translation.
3. Gemini refinement.
4. OmniVoice reference-based synthesis.
5. Background separation і дамешванне.
6. Pause removal пры фінальнай зборцы.
7. Захаванне абодвух відаў subtitles.

## Асноўныя нелінейныя месцы і fallback-і

### 1. Translation можа рэкурсіўна дзяліць chunks

Калі LLM не вяртае валідны JSON або ламае структуру радкоў, chunk дзеліцца і перакладаецца паўторна.

### 2. Refinement можа рэкурсіўна дзяліць batches

Калі refinement для batch не праходзіць, batch дзеліцца на меншыя часткі.

### 3. Duration estimation не з’яўляецца канчатковай ісцінай

Пасля ацэнкі і выбару `short/long` варыянту праект усё роўна пераправярае ўжо рэальны synthesized audio.

### 4. Voice selection і reference selection не адно і тое ж

Для адных backend-аў галоўны механізм гэта selection з каталога галасоў, для іншых ключавое значэнне мае reference audio.

### 5. UI і код падтрымліваюць не зусім ідэнтычныя наборы choices

Некаторыя backend-ы або рэжымы існуюць у кодзе, але не ўсе экспанаваны ў UI як explicit choice. Пры падтрымцы праекта трэба глядзець не толькі на UI, але і на config/factory layers.

### 6. Частка налад з’яўляецца агульнай, але фактычна працуе толькі для асобных backend-аў

Напрыклад:

- `tts_prompt_prefix` карысны толькі для часткі TTS wrappers;
- `reference_text` важны толькі для reference-based systems;
- `voice_auto_selection` мае сэнс толькі там, дзе ёсць voice matching logic.

### 7. Pause removal працуе не па арыгінальным відэаасобна, а па фінальным audio mix

Гэта важна: праект спачатку стварае фінальны audio mix для аналізу паўз, і толькі потым вырашае, што выразаць у відэа.

## Кароткая выснова

DubbLM фактычна пабудаваны як шматэтапны orchestration pipeline вакол трох зменных плоскасцей:

- які transcription backend выкарыстаны;
- які translation/refinement backend выкарыстаны;
- які TTS backend і які механізм voice/reference selection выкарыстаны.

Асноўная інжынерная ідэя праекта не ў простым “speech-to-text -> translate -> TTS”, а ў тым, што паміж гэтымі крокамі ёсць:

- кантэкстны аналіз усяго дыялогу;
- refinement з multiple text variants;
- timing-aware выбар версіі для TTS;
- group-level timing correction;
- pause-aware фінальная зборка відэа.

Менавіта гэта вызначае фактычныя паводзіны праекта значна мацней, чым сам выбар канкрэтнага backend-а.
