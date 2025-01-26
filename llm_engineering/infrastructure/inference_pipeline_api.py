import opik
from fastapi import FastAPI, HTTPException
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

configure_opik()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str

@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    model_id = "jessemeng/TwinLlama-3.2-1B-DPO"

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )
  
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
  
        logger.info("Preparing input...")
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"
        
        logger.info("Tokenizing input...")
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_INPUT_LENGTH
        ).to(model.device)
        
        # 移除 token_type_ids
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        logger.info("Generating response...")
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs['input_ids'],  # 只传递 input_ids
                attention_mask=inputs.get('attention_mask'),  # 可选的 attention_mask
                max_new_tokens=settings.MAX_NEW_TOKENS_INFERENCE,
                temperature=settings.TEMPERATURE_INFERENCE,
                top_p=settings.TOP_P_INFERENCE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        logger.info("Decoding response...")
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        logger.info("Response generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error in call_llm_service: {str(e)}")
        logger.exception("Detailed traceback:")
        raise RuntimeError(f"LLM service failed: {str(e)}")


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

