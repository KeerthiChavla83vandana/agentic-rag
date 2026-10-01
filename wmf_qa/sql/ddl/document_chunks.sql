-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table (mirrors primary backend model for standalone wmf_ip_qa database)
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(255) PRIMARY KEY,
    source_file VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    folder_name VARCHAR(255),
    subfolder_name VARCHAR(255),
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tagging_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    embedded_at TIMESTAMP WITH TIME ZONE
);

-- Create document_chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL, -- 1024 dimensions for BAAI/bge-m3
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsvector_content_english tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Index for dense vector cosine similarity (using HNSW)
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Index for sparse Full-Text Search
CREATE INDEX IF NOT EXISTS document_chunks_fts_idx 
    ON document_chunks USING gin (tsvector_content_english);

-- Index on document_id foreign key
CREATE INDEX IF NOT EXISTS document_chunks_doc_id_idx 
    ON document_chunks (document_id);
