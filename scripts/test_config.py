from src.core.config import get_settings


def main():
    settings = get_settings()

    print("=" * 60)
    print("Configuration Check")
    print("=" * 60)

    print(f"Application : {settings.app_name}")
    print(f"Environment : {settings.app_env}")
    print(f"Log Level   : {settings.log_level}")
    print(f"LLM Model   : {settings.llm_model}")
    print(f"Embedding   : {settings.embedding_model}")

    print(
        "OpenAI Key  : "
        + ("SET" if settings.openai_api_key else "NOT SET")
    )

    print(
        "LangSmith   : "
        + ("SET" if settings.langsmith_api_key else "NOT SET")
    )

    print("=" * 60)


if __name__ == "__main__":
    main()