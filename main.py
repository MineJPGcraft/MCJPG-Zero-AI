# --- START OF FILE main.py ---

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Literal, Tuple, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionChunk
# Import types for other endpoints if needed for stricter validation, or handle as Dict/Any
from openai.types.create_embedding_response import CreateEmbeddingResponse, Embedding
from openai.types.audio import Transcription
from dotenv import load_dotenv
import httpx # Keep for potential direct streaming later if needed, though openai lib handles most cases

# --- 配置 ---
load_dotenv() # 加载 .env 文件 (如果存在)

PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "http://127.0.0.1:3000")
if not PROXY_BASE_URL.endswith('/'):
    PROXY_BASE_URL += '/'
V1_ROUTE_PREFIX = "v1"

SELF_MODEL_ID = "MCJPG-Zero-v1"
SELF_MODEL_CONTEXT_LENGTH = 65536

ROUTING_MODEL = "gemini-2.0-flash"
DIRECT_TOOL_CALL_MODEL = "gpt-4o-mini" # 指定处理工具调用的模型

# --- NEW: Upstream model definitions ---
UPSTREAM_EMBEDDING_MODEL = "text-embedding-3-large"
UPSTREAM_TTS_MODEL = "tts-1"
UPSTREAM_STT_MODEL = "whisper-1"

MCJPG_SYSTEM_PROMPT = "你是由MCJPG组织开发的AI大语言模型,MCJPG组织是一个致力于Minecraft技术交流和服务器宣传的组织,你的输入输出内容应符合中华人民共和国法律。"

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI 应用 ---
app = FastAPI(
    title="MCJPG AI Model Router",
    description="Routes requests to different upstream models (Chat, Embeddings, TTS, STT) based on content and handles tool calls.",
    version="1.2.0", # Version bump for new features
)

# --- Pydantic 模型定义 (匹配 OpenAI API 格式) ---

# Model Listing Models
class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610600
    owned_by: str = "mcjpg"
    # Note: Standard OpenAI model object doesn't explicitly list capabilities like 'tts', 'embedding'.
    # Listing the model ID implies it's available for endpoints that accept it.
    permission: List = []
    root: str = Field(default_factory=str)
    parent: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelData]

# Chat Completion Models (Existing)
class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], None] = None # Content can be None for tool calls
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # We will ignore most non-core params unless forwarding directly (like tool calls or other endpoints)

# --- NEW: Embedding Models ---
class EmbeddingRequest(BaseModel):
    model: str # Will be validated to be SELF_MODEL_ID
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[Literal["float", "base64"]] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None

# --- NEW: TTS Models ---
class TTSRequest(BaseModel):
    model: str # Will be validated to be SELF_MODEL_ID
    input: str
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "mp3" # Default added
    speed: Optional[float] = None # Range 0.25 to 4.0

# --- NEW: STT - Note: Handled via Form data, not a single Pydantic model for the body ---

# --- Routing Function Definition (Existing) ---
class SelectModelParams(BaseModel):
    reasoning: str = Field(description="解释为什么选择这个模型 Explain why this model was chosen.")
    selected_task_type: Literal[
        "translation", "roleplay", "general_chat", "simple_vision", "web_search",
        "search_and_reason", "coding", "writing", "math_data_analysis",
        "image_generation", "image_editing", "video_generation", "music_generation",
        "unknown"
    ] = Field(description="根据用户输入分析得出的最合适的任务类型 The most appropriate task type analyzed from the user input.")

select_model_tool = {
    "type": "function",
    "function": {
        "name": "select_upstream_model",
        "description": "根据用户的请求内容和提供的模型特征，分析并选择最合适的上游模型任务类型 Analyze the user's request and select the most appropriate upstream model task type based on the provided model features.",
        "parameters": SelectModelParams.model_json_schema()
    }
}

# --- 辅助函数 ---

def get_user_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """从 Authorization Header 中提取 API Key"""
    if authorization and authorization.startswith("Bearer "):
        return authorization.split("Bearer ")[1]
    return None

def contains_image(messages: List[ChatCompletionMessage]) -> bool:
    """检查消息列表中是否包含图片内容"""
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False

# Map Task (Existing - unchanged)
def map_task_to_model(task_type: str, has_image: bool) -> str:
    """根据任务类型和是否有图片，映射到具体的上游聊天模型名称"""
    logger.info(f"Mapping task type: {task_type}, Has image: {has_image}")
    if task_type in ["translation", "roleplay", "general_chat", "simple_vision"]:
        return "gemini-2.0-flash"
    elif task_type == "web_search":
        return "gemini-2.0-flash-search"
    elif task_type == "search_and_reason":
        return "jina-deepsearch-v1"
    elif task_type == "coding":
        return "claude-3-7-sonnet-20250219"
    elif task_type == "writing":
        return "claude-3-7-sonnet-20250219" if has_image else "DeepSeek-V3"
    elif task_type == "math_data_analysis":
        return "gemini-2.5-pro-exp-03-25" if has_image else "DeepSeek-R1"
    elif task_type in ["image_generation", "image_editing"]:
        return "gpt-4o-image"
    elif task_type == "video_generation":
        return "sora-16:9-720p-5s"
    elif task_type == "music_generation":
        return "udio32-v1.5"
    else:
        logger.warning(f"Unknown or unhandled task type '{task_type}'. Defaulting to gemini-2.0-flash.")
        return "gemini-2.0-flash"


def prepare_upstream_messages(user_messages: List[ChatCompletionMessageParam]) -> List[ChatCompletionMessageParam]:
    """准备发送给上游模型的最终消息列表，加入固定系统提示词 (For Chat)"""
    processed_messages: List[ChatCompletionMessageParam] = []
    user_system_prompt_content = None
    other_messages = []

    for msg in user_messages:
        if msg.get("role") == "system":
            user_system_prompt_content = msg.get("content")
        else:
            other_messages.append(msg)

    processed_messages.append({"role": "system", "content": MCJPG_SYSTEM_PROMPT})

    if user_system_prompt_content:
        if isinstance(user_system_prompt_content, str):
             processed_messages.append({"role": "system", "content": user_system_prompt_content})
        elif isinstance(user_system_prompt_content, list): # Support vision system prompt
             processed_messages.append({"role": "system", "content": user_system_prompt_content})
        else:
            logger.warning("User system prompt has an unexpected type, skipping adding it.")

    processed_messages.extend(other_messages)

    if not any(msg.get("role") != "system" for msg in processed_messages) and len(processed_messages) > 1:
         logger.warning("Request might contain only system messages after processing.")

    return processed_messages

async def get_openai_client(api_key: str) -> AsyncOpenAI:
    """获取配置好的 AsyncOpenAI 客户端"""
    # Ensure httpx client is implicitly created/managed by AsyncOpenAI
    return AsyncOpenAI(api_key=api_key, base_url=f"{PROXY_BASE_URL}{V1_ROUTE_PREFIX}")

# --- Helper for handling API errors ---
async def handle_openai_api_error(e: Exception, upstream_model_name: str):
    logger.exception(f"Error during request to upstream model {upstream_model_name}: {e}")
    status_code = 500
    error_detail = f"Error communicating with upstream model '{upstream_model_name}': {str(e)}"

    # Try to extract status code and details from OpenAI/HTTPX errors
    if isinstance(e, httpx.HTTPStatusError):
        status_code = e.response.status_code
        try:
            # Attempt to parse the error response body
            error_body = e.response.json()
            error_detail = error_body.get("error", {}).get("message", e.response.text)
        except json.JSONDecodeError:
            error_detail = e.response.text # Fallback to raw text
        error_detail = f"Upstream API error ({status_code}) for model '{upstream_model_name}': {error_detail}"
    elif hasattr(e, 'status_code'): # Handle potential errors from the openai library itself
         status_code = e.status_code
         error_detail = f"Upstream API error ({status_code}) for model '{upstream_model_name}': {str(e)}"

    raise HTTPException(status_code=status_code, detail=error_detail)


# --- API Endpoints ---

@app.get(f"/{V1_ROUTE_PREFIX}/models", response_model=ModelList)
async def list_models():
    """
    提供模型列表，只包含我们自定义的模型。
    Note: Listing MCJPG-Zero-v1 implies it can be used with /chat/completions,
    /embeddings, /audio/speech, /audio/transcriptions endpoints where appropriate.
    """
    model_data = ModelData(
        id=SELF_MODEL_ID,
        root=SELF_MODEL_ID,
        # owned_by="mcjpg" # Already set by default
    )
    return ModelList(data=[model_data])

# --- NEW: Embeddings Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/embeddings", response_model=CreateEmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Handles embedding requests.
    If model is SELF_MODEL_ID, forwards to UPSTREAM_EMBEDDING_MODEL.
    """
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if request.model != SELF_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model requested. This endpoint only supports '{SELF_MODEL_ID}' for routing."
        )

    logger.info(f"Received embedding request for model: {request.model}. Forwarding to {UPSTREAM_EMBEDDING_MODEL}.")

    client = None
    try:
        client = await get_openai_client(user_api_key)
        embedding_params = {
            "model": UPSTREAM_EMBEDDING_MODEL,
            "input": request.input,
        }
        if request.encoding_format:
            embedding_params["encoding_format"] = request.encoding_format
        if request.dimensions:
            embedding_params["dimensions"] = request.dimensions
        if request.user:
            embedding_params["user"] = request.user # Pass user if provided

        response = await client.embeddings.create(**embedding_params)
        logger.info(f"Successfully received embedding response from {UPSTREAM_EMBEDDING_MODEL}.")
        # Pydantic response_model will handle serialization
        return response
    except Exception as e:
        await handle_openai_api_error(e, UPSTREAM_EMBEDDING_MODEL)
    finally:
        if client:
            await client.close()


# --- NEW: TTS Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/audio/speech")
async def create_speech(
    request: TTSRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Handles Text-to-Speech (TTS) requests.
    If model is SELF_MODEL_ID, forwards to UPSTREAM_TTS_MODEL.
    Returns audio stream or file.
    """
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if request.model != SELF_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model requested. This endpoint only supports '{SELF_MODEL_ID}' for routing."
        )

    logger.info(f"Received TTS request for model: {request.model}. Forwarding to {UPSTREAM_TTS_MODEL}.")

    client = None
    upstream_response: Optional[httpx.Response] = None
    try:
        client = await get_openai_client(user_api_key)
        tts_params = {
            "model": UPSTREAM_TTS_MODEL,
            "input": request.input,
            "voice": request.voice,
        }
        # Add optional parameters if provided
        if request.response_format:
            tts_params["response_format"] = request.response_format
        if request.speed:
            tts_params["speed"] = request.speed

        # The OpenAI library's method returns an httpx.Response containing the audio
        # We need to stream this response back to the client
        upstream_response = await client.audio.speech.create(**tts_params)
        upstream_response.raise_for_status() # Raise exception for 4xx/5xx errors

        logger.info(f"Successfully received TTS audio stream from {UPSTREAM_TTS_MODEL}.")

        # Determine content type from upstream or request format
        content_type = upstream_response.headers.get("content-type", f"audio/{request.response_format}")

        # Stream the audio content back
        async def audio_stream_generator() -> AsyncGenerator[bytes, None]:
            async for chunk in upstream_response.aiter_bytes():
                yield chunk
            # Ensure the client is closed *after* streaming is done
            if client:
                 await client.close()
                 client = None # Prevent double close in finally
            logger.info("TTS audio streaming finished.")


        return StreamingResponse(
             audio_stream_generator(),
             media_type=content_type,
             headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"} # Optional: suggest filename
        )

    except httpx.HTTPStatusError as e:
         # Close client before raising HTTP exception if stream setup failed early
        if client:
            await client.close()
        await handle_openai_api_error(e, UPSTREAM_TTS_MODEL)
    except Exception as e:
        if client:
            await client.close()
        # Handle generic errors during request setup
        await handle_openai_api_error(e, UPSTREAM_TTS_MODEL)
    finally:
        # Close response stream if it exists and wasn't handled by the generator
        if upstream_response:
            await upstream_response.aclose()
        # Close client only if it wasn't closed by the generator's success path
        if client:
            await client.close()


# --- NEW: STT Endpoint ---
# Note: Uses Form data, not JSON body
@app.post(f"/{V1_ROUTE_PREFIX}/audio/transcriptions", response_model=Transcription)
async def create_transcription(
    authorization: Optional[str] = Header(None),
    # --- Parameters from multipart/form-data ---
    file: UploadFile = File(...),
    model: str = Form(...), # Model ID is part of the form
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Form(None) # New parameter in OpenAI API
):
    """
    Handles Speech-to-Text (STT) requests using multipart/form-data.
    If model is SELF_MODEL_ID, forwards to UPSTREAM_STT_MODEL.
    """
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    # Validate the model ID received in the form data
    if model != SELF_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model requested ('{model}'). This endpoint only supports '{SELF_MODEL_ID}' for routing."
        )

    logger.info(f"Received STT request for model: {model}. Forwarding to {UPSTREAM_STT_MODEL}.")

    client = None
    try:
        client = await get_openai_client(user_api_key)

        # Prepare parameters, carefully handling optional ones
        stt_params = {
            "model": UPSTREAM_STT_MODEL,
            "file": (file.filename, await file.read(), file.content_type), # Pass file tuple
            "response_format": response_format,
            "temperature": temperature
        }
        if language:
            stt_params["language"] = language
        if prompt:
            stt_params["prompt"] = prompt
        if timestamp_granularities:
             # The SDK expects a list for this parameter if provided
             stt_params["timestamp_granularities"] = timestamp_granularities

        response = await client.audio.transcriptions.create(**stt_params)
        logger.info(f"Successfully received STT response from {UPSTREAM_STT_MODEL}.")
        # Pydantic response_model handles serialization
        return response
    except Exception as e:
        await handle_openai_api_error(e, UPSTREAM_STT_MODEL)
    finally:
        # Ensure the uploaded file resource is closed
        await file.close()
        if client:
            await client.close()


# --- Chat Completions Endpoint (Existing Logic) ---
@app.post(f"/{V1_ROUTE_PREFIX}/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Handles chat completion requests.
    Routes based on content analysis or forwards directly for tool calls.
    """
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if request.model != SELF_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model requested. This endpoint only supports '{SELF_MODEL_ID}' for chat."
        )

    logger.info(f"Received chat request for model: {request.model}, Stream: {request.stream}, Tools specified: {bool(request.tools or request.tool_choice)}")

    upstream_model_name = None
    routing_reasoning = "N/A" # For logging

    is_direct_tool_call = bool(request.tools or request.tool_choice)

    if is_direct_tool_call:
        upstream_model_name = DIRECT_TOOL_CALL_MODEL
        logger.info(f"Chat request contains tools/tool_choice. Directly forwarding to {upstream_model_name}.")
    else:
        # --- Execute Routing Logic ---
        logger.info("Chat request does not contain tools/tool_choice. Proceeding with model routing.")
        routing_client = None
        try:
            routing_client = await get_openai_client(user_api_key)
            # Prepare messages for routing model (text only)
            routing_messages: List[ChatCompletionMessageParam] = []
            user_system_prompt_content = None
            user_messages_for_routing = []

            for msg in request.messages:
                msg_dict = msg.model_dump(exclude_none=True)
                if msg_dict["role"] == "system":
                    user_system_prompt_content = msg_dict.get("content")
                # Process content for routing model
                content_for_routing = None
                if isinstance(msg_dict.get("content"), list):
                     text_content = " ".join([item["text"] for item in msg_dict["content"] if item.get("type") == "text"])
                     if any(item.get("type") == "image_url" for item in msg_dict["content"]):
                          text_content += " [Image provided by user]"
                     content_for_routing = text_content
                elif isinstance(msg_dict.get("content"), str):
                    content_for_routing = msg_dict.get("content")

                # Add message if it has processable content for routing
                if content_for_routing:
                    user_messages_for_routing.append({"role": msg_dict["role"], "content": content_for_routing})
                # else: Ignore messages without text content for routing (e.g., tool responses)

            # Build routing system prompt
            routing_system_prompt = (
                 "You are an AI assistant responsible for routing user requests to the appropriate specialized AI model. "
                 "Analyze the user's message content (text only, image presence is indicated by '[Image provided by user]') and determine the primary task. "
                 "Use the 'select_upstream_model' function to indicate your choice based on the following criteria:\n"
                "- **translation, roleplay, general_chat, simple_vision**: Use 'gemini-2.0-flash'.\n"
                "- **web_search**: Use 'gemini-2.0-flash-search' for questions needing live web data.\n"
                "- **search_and_reason**: Use 'jina-deepsearch-v1' for questions requiring web search combined with reasoning.\n"
                "- **coding**: Use 'claude-3-7-sonnet-20250219' for programming or code-related questions.\n"
                "- **writing**: Use 'DeepSeek-V3' for creative writing, summaries, etc. (If image present, use 'claude-3-7-sonnet-20250219').\n"
                "- **math_data_analysis**: Use 'DeepSeek-R1' for math problems, data analysis. (If image present, use 'gemini-2.5-pro-exp-03-25').\n"
                "- **image_generation, image_editing**: Use 'gpt-4o-image' if the user asks to generate or edit an image.\n"
                "- **video_generation**: Use 'sora-16:9-720p-5s' if the user asks to generate a video.\n"
                "- **music_generation**: Use 'udio32-v1.5' if the user asks to generate music.\n"
                "- If unsure, choose 'general_chat'."
            )
            if user_system_prompt_content and isinstance(user_system_prompt_content, str):
                 routing_system_prompt += f"\n\nUser's System Prompt for context: {user_system_prompt_content}"

            routing_messages.append({"role": "system", "content": routing_system_prompt})
            routing_messages.extend(user_messages_for_routing)

            # Check if there are any messages left for routing after filtering
            if not user_messages_for_routing:
                 logger.warning("No user/assistant messages with text content found for routing. Defaulting to general_chat.")
                 upstream_model_name = map_task_to_model("general_chat", False) # Default if only system/tool messages exist
            else:
                logger.info(f"Requesting model selection from {ROUTING_MODEL}...")
                response = await routing_client.chat.completions.create(
                    model=ROUTING_MODEL,
                    messages=routing_messages,
                    tools=[select_model_tool],
                    tool_choice={"type": "function", "function": {"name": "select_upstream_model"}},
                    temperature=0.1,
                )

                if not response.choices or not response.choices[0].message.tool_calls:
                    logger.error("Routing model did not return a function call.")
                    # Defaulting instead of failing hard
                    logger.warning("Defaulting to general_chat due to routing model failure.")
                    upstream_model_name = map_task_to_model("general_chat", False) # Default
                    routing_reasoning = "Routing failed, defaulted to general_chat."
                else:
                    tool_call = response.choices[0].message.tool_calls[0]
                    if tool_call.function.name != "select_upstream_model":
                        logger.error(f"Routing model returned unexpected function call: {tool_call.function.name}. Defaulting.")
                        upstream_model_name = map_task_to_model("general_chat", False) # Default
                        routing_reasoning = f"Routing used wrong tool ({tool_call.function.name}), defaulted."
                    else:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            selected_task_type = function_args.get("selected_task_type")
                            routing_reasoning = function_args.get("reasoning", "No reasoning provided.")

                            if not selected_task_type:
                                logger.error(f"Routing model did not provide 'selected_task_type': {function_args}. Defaulting.")
                                upstream_model_name = map_task_to_model("general_chat", False) # Default
                                routing_reasoning = "Routing missing task type, defaulted."
                            else:
                                logger.info(f"Routing decision: Task Type='{selected_task_type}', Reasoning='{routing_reasoning}'")
                                has_image_input = contains_image(request.messages)
                                upstream_model_name = map_task_to_model(selected_task_type, has_image_input)
                                logger.info(f"Selected upstream model after routing: {upstream_model_name}")
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse routing function arguments: {tool_call.function.arguments}. Defaulting.")
                            upstream_model_name = map_task_to_model("general_chat", False) # Default
                            routing_reasoning = "Routing argument parsing failed, defaulted."

        except Exception as e:
            # Log the error but default to a basic model instead of failing the request
            logger.exception(f"Error during model selection using {ROUTING_MODEL}: {e}. Defaulting to general_chat.")
            upstream_model_name = map_task_to_model("general_chat", False) # Default on error
            routing_reasoning = f"Routing exception occurred ({type(e).__name__}), defaulted."
        finally:
            if routing_client:
                await routing_client.close()
        # --- Routing Logic End ---

    if not upstream_model_name:
         # This case should ideally be covered by defaults in the routing logic now
         logger.error("Upstream model name could not be determined after routing attempts. Failing request.")
         raise HTTPException(status_code=500, detail="Failed to determine upstream chat model.")

    # --- Forward request to selected upstream chat model ---
    upstream_client = None
    try:
        upstream_client = await get_openai_client(user_api_key)
        final_messages = prepare_upstream_messages([msg.model_dump(exclude_none=True) for msg in request.messages])

        upstream_payload = {
            "model": upstream_model_name,
            "messages": final_messages,
            "stream": request.stream,
            # Conditionally include tools/tool_choice only for direct tool call forwarding
            "tools": request.tools if is_direct_tool_call else None,
            "tool_choice": request.tool_choice if is_direct_tool_call else None,
            # Pass other relevant parameters
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stop": request.stop,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "logit_bias": request.logit_bias,
            "user": request.user,
        }
        upstream_payload = {k: v for k, v in upstream_payload.items() if v is not None} # Clean None values

        if request.stream:
            # --- Handle Streaming Response ---
            async def stream_generator():
                response_stream = None
                try:
                    logger.info(f"Streaming chat request to upstream model: {upstream_model_name} with payload keys: {list(upstream_payload.keys())}")
                    response_stream = await upstream_client.chat.completions.create(**upstream_payload)
                    async for chunk in response_stream:
                        # Add routing reasoning to the first chunk's metadata if available (optional)
                        # This is non-standard OpenAI, use with caution or custom clients
                        # chunk_dict = chunk.model_dump(exclude_unset=True)
                        # if routing_reasoning != "N/A":
                        #    if 'choices' in chunk_dict and chunk_dict['choices']:
                        #        chunk_dict['choices'][0]['routing_info'] = {"reasoning": routing_reasoning}
                        #    routing_reasoning = "N/A" # Only send once
                        # chunk_json = json.dumps(chunk_dict)

                        chunk_json = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {chunk_json}\n\n"
                    yield "data: [DONE]\n\n"
                    logger.info("Chat streaming finished.")
                except Exception as e:
                    logger.exception(f"Error during streaming from upstream chat: {e}")
                    error_payload = { "error": { "message": f"Upstream streaming error: {str(e)}", "type": "upstream_error", "code": 500 } }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    # Close stream resources if applicable (openai lib might handle this internally)
                    # Close the client *after* the stream is fully consumed
                    if upstream_client:
                       await upstream_client.close()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # --- Handle Non-Streaming Response ---
            logger.info(f"Sending non-streaming chat request to upstream model: {upstream_model_name} with payload keys: {list(upstream_payload.keys())}")
            completion = await upstream_client.chat.completions.create(**upstream_payload)
            logger.info("Received non-streaming chat response from upstream.")
            completion_dict = completion.model_dump(exclude_unset=True)
            # Add routing reasoning (optional, non-standard)
            # completion_dict['routing_info'] = {"model_selected": upstream_model_name, "reasoning": routing_reasoning}
            if upstream_client:
                 await upstream_client.close()
            return completion_dict # FastAPI handles JSON serialization

    except Exception as e:
        # Ensure client is closed on error before raising
        if upstream_client:
            await upstream_client.close()
        await handle_openai_api_error(e, upstream_model_name)


# --- 运行服务器 (用于本地测试) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting MCJPG AI Router v{app.version}")
    logger.info(f"Proxying to base URL: {PROXY_BASE_URL}{V1_ROUTE_PREFIX}")
    logger.info(f"Chat routing model: {ROUTING_MODEL}")
    logger.info(f"Direct tool call model: {DIRECT_TOOL_CALL_MODEL}")
    logger.info(f"Upstream Embeddings: {UPSTREAM_EMBEDDING_MODEL}")
    logger.info(f"Upstream TTS: {UPSTREAM_TTS_MODEL}")
    logger.info(f"Upstream STT: {UPSTREAM_STT_MODEL}")
    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)