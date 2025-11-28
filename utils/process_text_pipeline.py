import uuid
from typing import Dict,Any
import utils.rag_initialization as rag_state
from loguru import logger


def process_text_pipeline(text: str, metadata: Dict[str, Any]):
    """
    Processing Pipeline: Chunk (Sliding Window) -> Embed (SentenceTransformers) -> Store (Chroma)
    """
    if not text.strip():
        return
        
    # --- 1. Intelligent Chunking (Sliding Window) ---
    # Size: 1000 chars (approx 200-300 words), Overlap: 200 chars
    # Overlap ensures context isn't lost if a sentence is split at the chunk boundary.
    chunk_size = 1000
    overlap = 200
    chunks = []

    #ChromaDB cloud has a one time hard limit of 300 records
    CHROMA_BATCH_LIMIT = 300
    
    # Iterate through text with a step size of (chunk_size - overlap)
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        # Ignore very small trailing chunks (e.g. whitespace or just a few chars)
        if len(chunk) > 50: 
            chunks.append(chunk)

    if not chunks:
        return

    # --- 2. Generate Embeddings ---
    # encode() returns a list of vectors (numpy arrays). We convert to list for JSON serialization compatibility if needed, 
    # though Chroma handles numpy arrays usually. .tolist() is safer.
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = rag_state.embedding_model.encode(chunks).tolist()

    # --- 3. Store in ChromaDB ---
    # Prepare IDs and Metadata for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Chroma requires metadata to be flat key-value pairs. 
    # Ensure our passed metadata is clean. We replicate the 'doc' metadata for every chunk.
    # Note: Chroma does not support nested dicts in metadata.
    safe_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe_metadata[k] = v
        else:
            safe_metadata[k] = str(v) # Convert complex types to string

    metadatas = [safe_metadata for _ in chunks]

   # --- 4. Store in ChromaDB (Batched) ---
    total_records = len(chunks)
    
    # Loop through the data in steps of CHROMA_BATCH_LIMIT
    for i in range(0, total_records, CHROMA_BATCH_LIMIT):
        batch_end = i + CHROMA_BATCH_LIMIT
        
        # Slice the lists to create a batch
        batch_documents = chunks[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]
        
        try:
            rag_state.collection.add(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"-> Batch {i//CHROMA_BATCH_LIMIT + 1}: Stored records {i} to {min(batch_end, total_records)}")
        except Exception as e:
            logger.error(f"Error adding batch {i} to {batch_end}: {e}")
            # Optional: You might want to raise the error or continue depending on your requirements
            raise e
        
    logger.info(f"-> Successfully completed storage of {total_records} chunks for Test ID: {metadata.get('test_id')}")