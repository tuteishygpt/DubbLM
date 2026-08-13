import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clear_tts_modules():
    for module_name in list(sys.modules):
        if module_name == "tts" or module_name.startswith("tts."):
            sys.modules.pop(module_name, None)


def test_tts_factory_import_does_not_eagerly_load_all_wrappers():
    _clear_tts_modules()

    importlib.import_module("tts.tts_factory")

    assert "tts.omnivoice_wrapper" not in sys.modules
    assert "tts.gemini_tts_wrapper" not in sys.modules
    assert "tts.openai_tts_wrapper" not in sys.modules
    assert "tts.xtts_local_wrapper" not in sys.modules


def test_tts_factory_lists_omnivoice_provider():
    _clear_tts_modules()

    factory = importlib.import_module("tts.tts_factory")

    assert "omnivoice" in factory.TTSFactory.get_available_providers()


def test_tts_factory_lists_higgs_without_importing_wrapper():
    _clear_tts_modules()

    factory = importlib.import_module("tts.tts_factory")

    assert "higgs" in factory.TTSFactory.get_available_providers()
    assert "tts.higgs_audio_wrapper" not in sys.modules


def test_gemini_tts_client_uses_vertex_ai_client_configuration(monkeypatch):
    module = importlib.import_module("tts.gemini_tts_wrapper")

    created = {}

    class FakeClient:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(module, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(module.genai, "Client", FakeClient)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "vertex-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    client = module.GeminiAPIClient(module.GeminiTTSConfig())
    client.initialize()

    assert created == {
        "vertexai": True,
        "project": "vertex-project",
        "location": "global",
    }


def test_gemini_retries_keep_the_selected_model(monkeypatch):
    module = importlib.import_module("tts.gemini_tts_wrapper")
    attempted_models = []

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            attempted_models.append(model)
            if len(attempted_models) < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(
                                        data=b"audio",
                                        mime_type="audio/L16;rate=24000",
                                    )
                                )
                            ]
                        )
                    )
                ]
            )

    monkeypatch.setattr(
        module.genai_types,
        "GenerateContentConfig",
        lambda **kwargs: kwargs,
    )
    config = module.GeminiTTSConfig(
        model="selected-model", max_retries=3, retry_delay_base=0
    )
    client = module.GeminiAPIClient(config)
    client.client = SimpleNamespace(models=FakeModels())

    assert client.synthesize_chunk("hello", object()) == b"audio"
    assert attempted_models == ["selected-model", "selected-model", "selected-model"]


def test_factory_discards_removed_fallback_model_inputs(monkeypatch):
    factory = importlib.import_module("tts.tts_factory")
    received = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def initialize(self):
            pass

    monkeypatch.setattr(factory, "_load_provider_class", lambda _name: FakeProvider)

    factory.TTSFactory.create_tts(
        "gemini",
        model="selected-model",
        fallback_model="old-profile-model",
        tts_fallback_model="old-top-level-model",
    )

    assert received == {"model": "selected-model"}
