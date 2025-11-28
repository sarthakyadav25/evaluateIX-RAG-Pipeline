import json
import asyncio
from models.QuestionGenerationRequest import QuestionGenerationRequest
from models.QuestionGenerationResponse import QuestionItem,QuestionGenerationResponse
from fastapi import HTTPException
import utils.rag_initialization as rag_state
from loguru import logger

async def question_generation(payload: QuestionGenerationRequest):
    """
    Generates interview questions based on ALL content ingested for a specific test_id.
    """

    payload.test_id = str(payload.test_id)
    if not rag_state.GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")

    # 1. Fetch ALL content for this test_id
    # collection.get() allows filtering by metadata without a query vector
    logger.info(f"Fetching all context for Test ID: {payload.test_id}")
    db_response = rag_state.collection.get(
        where={"test_id": payload.test_id}
    )
    
    documents = db_response.get("documents", [])
    
    if not documents:
        raise HTTPException(status_code=404, detail=f"No content found for test_id: {payload.test_id}")

    # 2. Prepare Context (Concatenate all chunks)
    # Gemini 1.5 Flash has a large context window, so we can dump a lot of text here.
    full_context = "\n\n".join(documents)
    
    # 3. Prompt Engineering
    prompt = f"""
    You are an expert technical interviewer. 
    Based ONLY on the provided context below, generate {payload.num_questions} {payload.difficulty} interview questions.
    These are the list of already existing question so don't include similiary type of questions: {payload.already_has}
    
    Output Format:
    Return ONLY a raw JSON list of objects. Do not use Markdown code blocks.
    Example: [{{"question_no": 1, "content": "What is the difference between TCP and UDP?"}}, {{"question_no": 2, "content": "Explain the CAP theorem."}}]
    
    Context:
    {full_context}
    """

    try:
        model = rag_state.genai.GenerativeModel("gemini-2.5-flash")
        response = await asyncio.to_thread(model.generate_content,prompt)

        
        if response.parts:
            # 4. Parse Response
            # Clean up any potential markdown formatting the LLM might still add
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "").replace("```", "")
            
            questions_data = json.loads(clean_text)
            
            if not isinstance(questions_data, list):
                raise ValueError("LLM did not return a list")

            # Validate and convert to model list
            formatted_questions = [QuestionItem(**q) for q in questions_data]
            return QuestionGenerationResponse(questions=formatted_questions)
        else:
            logger.warning("Gemini output was empty in question generation")
            raise ValueError("LLM did not return a proper ouput")

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Fallback if LLM returns bad JSON or wrong structure
        logger.error(f"Parsing failed ({e}), falling back to line splitting.")
        lines = [line.strip() for line in response.text.split('\n') if line.strip() and '?' in line]
        
        # Manually construct the objects
        fallback_questions = [
            QuestionItem(question_no=i+1, content=line) 
            for i, line in enumerate(lines[:payload.num_questions])
        ]
        return QuestionGenerationResponse(questions=fallback_questions)
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")