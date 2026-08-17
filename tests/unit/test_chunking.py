from src.ingestion.chunker import TextChunker


def test_chunker_creates_chunks():
    text = "A" * 2500

    chunker = TextChunker(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = chunker.split(text)

    assert len(chunks) > 1


def test_chunker_returns_empty_for_empty_text():
    chunker = TextChunker()

    assert chunker.split("") == []


def test_chunker_preserves_short_text():
    text = "Hello banking"

    chunker = TextChunker()

    chunks = chunker.split(text)

    assert chunks == [text]


def test_invalid_chunk_size():
    try:
        TextChunker(chunk_size=0)
        assert False
    except ValueError:
        assert True


def test_invalid_overlap():
    try:
        TextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )
        assert False
    except ValueError:
        assert True