"""Non-interactive smoke test: config + Gemini streaming, no audio/GUI."""
from assistant import config
from assistant.llm.gemini import GeminiClient


def main():
    print("GOOGLE_API_KEY set:", bool(config.GOOGLE_API_KEY))
    print("Model:", config.GEMINI_MODEL)
    client = GeminiClient()
    print("\nUser: Fuel entha undi?")
    print("Jarvis:", end=" ", flush=True)
    for sentence in client.stream("Fuel entha undi?"):
        print(sentence, end=" ", flush=True)
    print("\n\nSmoke test OK.")


if __name__ == "__main__":
    main()
