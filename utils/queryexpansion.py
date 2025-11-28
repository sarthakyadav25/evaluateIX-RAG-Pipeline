import utils.rag_initialization as rag_state
import asyncio
from .parse_markdown_json import parse_markdown_json

async def query_expansion(query):
    """This function takes the user query and gives it to LLM to get a generalized
    answer to be passed along with the original user query"""

    print("Using query to generate generalized answer...")

    if rag_state.GEMINI_API_KEY:
        try:
            model = rag_state.genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
               "You are an AI assistant that helps users to provide with a generalized answer to their query."
               "You have to provide answer that is relevant to the query"
               "You can use different sources to get the answer"
                )
            )
            
            response = await asyncio.to_thread(model.generate_content,query)

            if response.parts:
                return response.text
            else:
                print("Gemini Response for generalized answer was blocked or empty")
                return ""

        except Exception as e:
            print(f"Error in query expansion: {e}")


