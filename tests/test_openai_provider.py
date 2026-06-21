from app.providers.openai_provider import OpenAIProvider


class FakeUsage:
    prompt_tokens = 9
    completion_tokens = 4


class FakeMessage:
    content = "ok"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()

    def model_dump(self) -> dict:
        return {"id": "fake-response"}


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.chat = FakeChat()


def test_openai_provider_uses_max_completion_tokens(monkeypatch) -> None:
    fake_client_holder = {}

    def fake_openai(api_key: str) -> FakeOpenAIClient:
        client = FakeOpenAIClient(api_key=api_key)
        fake_client_holder["client"] = client
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.providers.openai_provider.OpenAI", fake_openai)

    provider = OpenAIProvider()
    response = provider.chat_completion(
        messages=[{"role": "user", "content": "Say ok."}],
        model_config={
            "model_id": "gpt-test",
            "input_cost_per_1m": 1.0,
            "output_cost_per_1m": 2.0,
        },
        temperature=0.2,
        max_tokens=5,
    )

    kwargs = fake_client_holder["client"].chat.completions.kwargs
    assert kwargs["max_completion_tokens"] == 5
    assert "max_tokens" not in kwargs
    assert response.text == "ok"

