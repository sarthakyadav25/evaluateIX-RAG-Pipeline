import uuid
import json
import asyncio
import utils.rag_initialization as rag_state
from models.RetrieveRequest import RetrieveRequest
from models.SearchResult import SearchResult
from models.RetrieveResponse import RetrieveResponse
from fastapi import FastAPI, HTTPException
from utils.parse_markdown_json import parse_markdown_json
from utils.queryexpansion import query_expansion
from utils.test_ai_content import zero_gpt_test
from loguru import logger


async def retrieval(payload: RetrieveRequest):
    """
    Retrieve relevant context for a user query using Vector Similarity, 
    then generate an answer using Gemini 2.5 Flash.
    """

    # ---0. Generate generalized response ---
    # --- also generate joint_query with generalized response and user query
    generalized_response = await query_expansion(payload.query)
    joint_query = payload.query + " " + generalized_response

    #check zero gpt AI plagarism score
    ai_score = zero_gpt_test(payload.query)



    # --- 1. Generate Embedding for Query ---
    # We must use the SAME model for query embedding as we did for document embedding
    query_vector = rag_state.embedding_model.encode(joint_query).tolist()
    
    target_test_id = payload.filters.get("test_id")
    if not target_test_id:
        raise HTTPException(status_code=400, detail="Missing test_id in filters")

    # --- 2. Query ChromaDB ---
    # We use the 'where' clause to strictly filter chunks by test_id (Tenancy Isolation)
    logger.info(f"Querying Chroma for Test ID: {target_test_id}")
    search_results = rag_state.collection.query(
        query_embeddings=[query_vector],
        n_results=payload.top_k,
        where={"test_id": target_test_id} 
    )

     # --- 3. Format Retrieval Results (with IDs for prompt) ---
    # Chroma returns lists of lists (because it supports batch queries). We take index 0.
    documents = search_results.get('documents', [])[0]
    distances = search_results.get('distances', [])[0]  # smaller is better
    metadatas = search_results.get('metadatas', [])[0]
    ids = search_results.get('ids', [])[0] if 'ids' in search_results else [str(uuid.uuid4()) for _ in documents]

    formatted_results = []
    context_text = ""
    retrieved_docs_payload = []

    if documents:
        for i in range(len(documents)):
            # Convert distance to a similarity-like score (approx 0..1)
            # Protect against division by zero if distance==0
            try:
                score = 1 / (1 + float(distances[i]))
            except Exception:
                score = 0.0

            doc_id = ids[i] if i < len(ids) else str(uuid.uuid4())

            formatted_results.append(SearchResult(
                content=documents[i],
                score=score,
                metadata=metadatas[i]
            ))

            # Build a compact context text for human / logging if needed
            context_text += f"Source {i+1} (id={doc_id}): {documents[i]}\n\n"

            # Build the retrieved_docs payload that will be injected into the LLM prompt
            # Keep the snippet length reasonable so we don't blow token budget; trim if needed.
            snippet = documents[i]
            # Optionally trim long snippets to e.g. 3000 chars
            max_snippet_len = 3000
            if len(snippet) > max_snippet_len:
                snippet = snippet[:max_snippet_len] + " ...[truncated]"

            retrieved_docs_payload.append({
                "id": str(doc_id),
                "text": snippet,
                "relevance_score": round(float(score), 4)
            })
    else:
        context_text = "No relevant context found."

    # --- 4. Call LLM (Gemini 1.5 Flash) ---
    answer = "LLM generation failed or key not configured."
    if rag_state.GEMINI_API_KEY:
        try:
            # Initialize Model
            model = rag_state.genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=(
                    "You are an objective, impartial technical interviewer and answer evaluator. "
                    "Use ONLY the candidate_answer and the retrieved_docs provided (treat retrieved_docs as ground-truth context). "
                    "Prioritize evidence in retrieved_docs: reward supported claims, penalize contradicted or unsupported claims. "
                    "Return ONLY the exact JSON matching the schema described below, nothing else."
                )
            )

            # Prepare retrieved_docs as a compact JSON string to inject into the prompt
            # We use json.dumps to ensure valid JSON formatting inside the prompt.
            retrieved_docs_json = json.dumps(retrieved_docs_payload, ensure_ascii=False)

            # Construct Prompt (compact, deterministic, and aligned with your schema)
            full_prompt = f"""
                                Inputs:
                                question: {json.dumps(payload.question, ensure_ascii=False)}
                                candidate_answer: {json.dumps(payload.query, ensure_ascii=False)}
                                retrieved_docs: {retrieved_docs_json}

                                Task:
                                1) Read the 'question', 'candidate_answer', and each item in 'retrieved_docs' (each item has id, text, relevance_score).
                                2) Score the candidate_answer OUT OF 100 using this rubric (weights sum to 100):
                                - Accuracy / Correctness (40): factual correctness relative to retrieved_docs.
                                - Completeness (25): covers required parts and key points in retrieved_docs.
                                - Relevance / Use of Evidence (15): directly uses or aligns with retrieved_docs; cites doc ids.
                                - Reasoning / Explanation (10): logic, steps, justifications when applicable.
                                - Clarity & Conciseness (5): clear, readable, not overly verbose.
                                - Citations & Traceability (5): references or matches retrieved_doc ids.

                                Scoring rules:
                                - Scores must be integers and sum to 100.
                                - Compute per-criterion integer scores (0..max) and sum to overall_score (0..100).
                                - If a claim directly contradicts any retrieved_doc, deduct proportionally under Accuracy.
                                - If a claim is unsupported (not contradicted), include it in 'unsupported_claims' with suggested_penalty_points.
                                - Award Relevance & Citations when the candidate paraphrases or cites retrieved_docs correctly.
                                - Provide 2-5 short actionable improvement bullets referencing doc ids when helpful.
                                - Provide up to 3 supporting_doc_ids and up to 3 contradicting_doc_ids.
                                - Include 'confidence' as a float 0.0-1.0 based on how well retrieved_docs cover the question.

                                Required JSON output (return this EXACT structure, JSON only, no extra text):
                                {{
                                "overall_score": integer,
                                "breakdown": [
                                    {{"criterion":"accuracy","score": integer,"max":40}},
                                    {{"criterion":"completeness","score": integer,"max":25}},
                                    {{"criterion":"relevance","score": integer,"max":15}},
                                    {{"criterion":"reasoning","score": integer,"max":10}},
                                    {{"criterion":"clarity","score": integer,"max":5}},
                                    {{"criterion":"citations","score": integer,"max":5}}
                                ],
                                "confidence": float,
                                "pass": boolean,
                                "rationale": "short explanation (1-3 sentences)",
                                "improvements": ["short bullet 1","short bullet 2"],
                                "evidence": {{
                                    "supporting_doc_ids": ["id1","id2"],
                                    "contradicting_doc_ids": ["id3"],
                                    "unsupported_claims": [
                                    {{"claim":"short text","suggested_penalty_points": integer}}
                                    ]
                                }}
                                }}

                                Notes:
                                - Keep 'rationale' to 1-3 sentences.
                                - Keep 'improvements' to 2-5 concise bullets.
                                - When listing unsupported_claims, paraphrase the claim and include integer penalty points deducted from Accuracy.
                                - Choose 'confidence' >0.8 when retrieved_docs clearly cover the question; lower otherwise.

                                Do the work and return ONLY the JSON described above.
                                """
            # Generate
            response = await asyncio.to_thread(model.generate_content,full_prompt)

            if response.parts:
                print(response)
                answer = parse_markdown_json(response.text)

                if answer is None:
                    answer = {}
            else:
                logger.warning("Gemini response for analyzing answers was blocked or empty")
                answer = {}

        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            answer = f"Error generating answer: {str(e)}"
    return RetrieveResponse(results=formatted_results, answer=answer, ai_score=ai_score)