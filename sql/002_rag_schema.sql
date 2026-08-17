-- ==========================================
-- Documents table
-- ==========================================

CREATE TABLE IF NOT EXISTS documents (

    id UUID PRIMARY KEY
        DEFAULT gen_random_uuid(),

    document_name TEXT UNIQUE NOT NULL,

    file_path TEXT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);



-- ==========================================
-- Multimodal chunks table
-- ==========================================

CREATE TABLE IF NOT EXISTS multimodal_chunks (

    id UUID PRIMARY KEY
        DEFAULT gen_random_uuid(),


    doc_id UUID NOT NULL,

    chunk_type VARCHAR(50),

    element_type VARCHAR(100),


    content TEXT NOT NULL,


    image_path TEXT,

    mime_type VARCHAR(100),


    page_number INTEGER,


    section TEXT,


    source_file TEXT,


    position JSONB,


    embedding VECTOR(1536),


    metadata JSONB DEFAULT '{}'::jsonb,


    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT fk_document
        FOREIGN KEY(doc_id)
        REFERENCES documents(id)
        ON DELETE CASCADE

);



-- ==========================================
-- Vector search index
-- ==========================================

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_embedding

ON multimodal_chunks

USING hnsw
(
    embedding vector_cosine_ops
);



-- ==========================================
-- Full text search index
-- ==========================================

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_fts

ON multimodal_chunks

USING gin
(
    to_tsvector(
        'english',
        content
    )
);



-- ==========================================
-- Filtering indexes
-- ==========================================

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_doc_id

ON multimodal_chunks(doc_id);



CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_page

ON multimodal_chunks(page_number);



CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_type

ON multimodal_chunks(chunk_type);