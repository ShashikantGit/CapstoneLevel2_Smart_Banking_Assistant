import sys


def main():
    print("=" * 60)
    print("Smart Banking Assistant - Environment Check")
    print("=" * 60)

    print(f"Python: {sys.version}")

    try:
        import fastapi
        print(f"FastAPI: {fastapi.__version__}")
    except ImportError:
        print("FastAPI: FAILED")

    try:
        import langchain
        print(f"LangChain: {langchain.__version__}")
    except ImportError:
        print("LangChain: FAILED")

    try:
        import langgraph
        print("LangGraph: OK")
    except ImportError:
        print("LangGraph: FAILED")

    try:
        from langchain_openai import ChatOpenAI
        print("ChatOpenAI: OK")
    except ImportError:
        print("ChatOpenAI: FAILED")

    try:
        import psycopg
        print("Psycopg: OK")
    except ImportError:
        print("Psycopg: FAILED")

    try:
        import langsmith
        print("LangSmith: OK")
    except ImportError:
        print("LangSmith: FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()