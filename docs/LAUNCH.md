# DubbLM — Дакумент па запуску

> **Важна**: Праект выкарыстоўвае `.venv` у каранёвай дырэкторыі.
> Заўсёды запускайце праект праз `.venv\Scripts\python.exe` або актывуйце асяродзе перад запускам.

---

## Патрабаванні

- Python 3.10+
- FFmpeg (даступны ў `PATH`)
- Файл `.env` з ключамі API (гл. `.env.example`)

---

## 1. Актывацыя віртуальнага асяродзя

### PowerShell (рэкамендуецца)

```powershell
# З каранёвай папкі праекта D:\CodexPRJ\DubbLM
.\.venv\Scripts\Activate.ps1
```

Пасля актывацыі ў прыглашэнні зявіцца `(.venv)`:
```
(.venv) PS D:\CodexPRJ\DubbLM>
```

### CMD

```cmd
.venv\Scripts\activate.bat
```

> **Калі атрымліваеце памылку** `cannot be loaded because running scripts is disabled`:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 2. Запуск праз CLI (`dubblm_cli.py`)

### Базавы сінтаксіс

```powershell
# З актываваным .venv:
python dubblm_cli.py --input "шлях\да\відэа.mp4" --config dubbing_config.yml

# БЕЗ актывацыі (прамы шлях да .venv python):
.\.venv\Scripts\python.exe dubblm_cli.py --input "шлях\да\відэа.mp4" --config dubbing_config.yml
```

### Асноўныя параметры CLI

| Параметр | Апісанне | Прыклад |
|----------|----------|---------|
| `--input` | Уваходны відэафайл (абавязкова) | `--input ori.mp4` |
| `--config` | YAML-файл канфігурацыі | `--config dubbing_config.yml` |
| `--source_language` | Мова арыгінала | `--source_language en` |
| `--target_language` | Мова перакладу | `--target_language ru` |
| `--output` | Шлях да выходнага файла | `--output result.mp4` |
| `--transcription_system` | Сістэма транскрыпцыі | `--transcription_system deepgram` |
| `--tts_system` | Сістэма TTS | `--tts_system omnivoice` |
| `--start_time` | Пачатак апрацоўкі (сек / ЧЧ:ММ:СС) | `--start_time 00:01:30` |
| `--duration` | Доўжыня фрагмента ў секундах | `--duration 60` |
| `--no_cache` | Адключыць кэш | `--no_cache` |
| `--debug_info` | Дадатковы лог | `--debug_info` |

### Прыклады запуску

```powershell
# Поўны дублікат з канфігам (рэкамендуецца)
python dubblm_cli.py --input ori.mp4 --config dubbing_config.yml

# Толькі першыя 5 хвілін відэа
python dubblm_cli.py --input ori.mp4 --config dubbing_config.yml --duration 300

# З вызначанага часу
python dubblm_cli.py --input ori.mp4 --config dubbing_config.yml --start_time 00:02:00 --duration 120

# Без кэша (прымусовая перагенерацыя)
python dubblm_cli.py --input ori.mp4 --config dubbing_config.yml --no_cache

# Толькі аб'яднаць відэа з ужо гатовым аудыа
python dubblm_cli.py --input ori.mp4 --config dubbing_config.yml --run_step combine_video
```

---

## 3. Запуск праз Gradio UI (`gradio_app.py`)

```powershell
# З актываваным .venv:
python gradio_app.py

# БЕЗ актывацыі:
.\.venv\Scripts\python.exe gradio_app.py
```

Пасля запуску адкрыйце браўзер па адрасе: **http://localhost:7860**

---

## 4. Файл канфігурацыі `dubbing_config.yml`

Файл змяшчае налады па змаўчанні для ўсіх запускаў. Параметры з YAML перазапісваюцца аргументамі CLI.

> **Увага**: Параметр `input` у YAML **ігнаруецца** — відэа заўсёды паказваецца праз `--input` у CLI.

### Ключавыя налады

```yaml
source_language: en          # Мова арыгінала
target_language: ru          # Мова перакладу
transcription_system: deepgram  # deepgram | whisper | gemini | assemblyai
tts_system: omnivoice        # omnivoice | coqui | google | elevenlabs
llm_provider: gemini         # Пастаўшчык LLM для перакладу
llm_model_name: gemini-3.1-pro-preview
keep_background: true        # Захаваць фонавае аудыа
save_original_subtitles: true
save_translated_subtitles: true
duration: 300                # Максімальная доўжыня ў секундах (0 = увесь файл)
watermark_text: "TuteishyGPT"
```

---

## 5. Файл `.env` (ключы API)

Стварыце `.env` на аснове `.env.example`:

```env
# Google Cloud / Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global

# Deepgram (для transcription_system: deepgram)
DEEPGRAM_API_KEY=your-deepgram-key

# AssemblyAI (для transcription_system: assemblyai)
ASSEMBLYAI_API_KEY=your-assemblyai-key

# OpenAI (калі выкарыстоўваецца)
OPENAI_API_KEY=your-openai-key

# OpenRouter (калі выкарыстоўваецца)
OPENROUTER_API_KEY=your-openrouter-key
```

---

## 6. Структура выходных файлаў

Вынікі захоўваюцца ў папцы `prj/<назва_відэа>/`:

```
prj/
└── ori/
    ├── ori_dubbed.mp4          ← Гатовы дублікат
    ├── ori_original.srt        ← Арыгінальныя субтытры (калі ўключана)
    ├── ori_translated.srt      ← Перакладзеныя субтытры (калі ўключана)
    ├── audio/                  ← Прамежкавыя аудыафайлы
    └── cache/                  ← Кэш транскрыпцыі і перакладу
```

---

## 7. Частыя памылкі

### ❌ `deepgram-sdk package not found`

**Прычына**: Запуск праз сістэмны Python замест `.venv`.

**Рашэнне**:
```powershell
# Заўсёды актывуйце .venv або выкарыстоўвайце:
.\.venv\Scripts\python.exe dubblm_cli.py ...
```

### ❌ `No module named 'dubbing'`

**Прычына**: Пакет не ўсталяваны ў рэжыме editable.

**Рашэнне**: Праект аўтаматычна дадае `src/` у `sys.path` пры запуску праз `dubblm_cli.py` або `gradio_app.py` — заўсёды запускайце **праз гэтыя файлы**, не напрамую праз `python src/dubbing/...`.

### ❌ `Set-ExecutionPolicy` (блакіроўка скрыпту)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 8. Хуткі старт (капіруйце і ўстаўце)

```powershell
# 1. Перайдзіце ў папку праекта
cd D:\CodexPRJ\DubbLM

# 2. Актывуйце .venv
.\.venv\Scripts\Activate.ps1

# 3. Запусціце дублікацыю
python dubblm_cli.py --input ori.mp4 --config dubbing_config.yml
```

---

*Дакумент створаны: 2026-08-05*
