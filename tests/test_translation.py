from pathlib import Path
import tomllib

import translation.llm_translator as llm_translator
from translation.llm_translator import LLMTranslator


def test_llm_translator_initializes_without_json_repair(monkeypatch):
    created = []

    def fake_create_llm(self, provider, model_name, temperature, max_tokens=None, purpose="default"):
        handle = object()
        created.append((purpose, handle))
        return handle

    monkeypatch.setattr(llm_translator, "JSON_REPAIR_AVAILABLE", False)
    monkeypatch.setattr(llm_translator, "json_repair", None, raising=False)
    monkeypatch.setattr(LLMTranslator, "_create_llm", fake_create_llm)

    translator = LLMTranslator(enable_cache=False)
    translator.initialize()

    assert [purpose for purpose, _ in created] == ["translation", "refinement"]
    assert translator.llm is created[0][1]
    assert translator.refinement_llm is created[1][1]


def test_pyproject_declares_json_repair_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("json-repair") for dep in dependencies)


def test_pyproject_declares_google_genai_llamaindex_dependency_for_default_translator():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("llama-index-llms-google-genai") for dep in dependencies)


def test_gemini_translator_uses_vertex_ai_config(monkeypatch):
    created = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_translator, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(llm_translator, "Gemini", FakeGemini, raising=False)
    monkeypatch.setattr(llm_translator, "GoogleGenAI", FakeGemini, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "vertex-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    translator = LLMTranslator(enable_cache=False)

    llm = translator._create_llm(
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.5,
        purpose="translation",
    )

    assert isinstance(llm, FakeGemini)
    assert created["model"] == "gemini-2.5-flash"
    assert created["vertexai_config"] == {
        "project": "vertex-project",
        "location": "global",
    }
    assert "api_key" not in created


def test_gemini_translator_normalizes_legacy_models_prefix(monkeypatch):
    created = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(llm_translator, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(llm_translator, "Gemini", FakeGemini, raising=False)
    monkeypatch.setattr(llm_translator, "GoogleGenAI", FakeGemini, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "vertex-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    translator = LLMTranslator(enable_cache=False)

    translator._create_llm(
        provider="gemini",
        model_name="models/gemini-2.5-flash",
        temperature=0.5,
        purpose="translation",
    )

    assert created["model"] == "gemini-2.5-flash"


def test_translate_single_chunk_prompt_requires_json_only_and_stress_marks():
    captured_prompt = {}

    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeLLM:
        def complete(self, prompt):
            captured_prompt["value"] = prompt
            return FakeResponse(
                '{"translations": [{"speaker": "SPEAKER_00", "text": "каса\\u0301"}]}'
            )

    translator = LLMTranslator(
        enable_cache=False,
        glossary={"LLM": "ИИ"},
    )
    translator.llm = FakeLLM()

    chunk = {
        "text": "SPEAKER_00: Hello there",
        "original_speaker_texts": [{"speaker": "SPEAKER_00", "text": "Hello there"}],
        "segments": [{"speaker": "SPEAKER_00", "text": "Hello there"}],
    }
    chunks = [
        {"text": "SPEAKER_99: Previous context", "translation": "SPEAKER_99: Папярэдні кантэкст"},
        chunk,
        {"text": "SPEAKER_01: Next context"},
    ]
    context_info = {
        "domain": "podcast",
        "tone": "playful",
        "themes": ["cats", "comedy"],
        "terminology": ["LLM", "API"],
    }

    translator._translate_single_chunk(
        chunk=chunk,
        i=1,
        chunks=chunks,
        context_info=context_info,
        source_language="en",
        target_language="be",
        source_summary="Кароткі змест відэа",
        enable_cache=False,
    )

    prompt = captured_prompt["value"]

    assert "Кароткі змест відэа" in prompt
    assert "SPEAKER_99: Папярэдні кантэкст" in prompt
    assert "SPEAKER_01: Next context" in prompt
    assert "podcast" in prompt
    assert "playful" in prompt
    assert "cats, comedy" in prompt
    assert "LLM, API" in prompt
    assert 'CRITICAL: Respond with valid JSON only.' in prompt
    assert 'Each translation object must contain exactly two keys: "speaker" and "text".' in prompt
    assert "Use the combining acute accent symbol U+0301" in prompt
    assert "каса́" in prompt
