import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Literal, Tuple, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile, Form
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionChunk
from openai.types.create_embedding_response import CreateEmbeddingResponse, Embedding
from openai.types.audio import Transcription
# Import Image generation types if available, or define basic ones
try:
    from openai.types import ImagesResponse, Image # Use official types if installed
except ImportError:
    # Define basic fallback types if openai library version is older
    class Image(BaseModel):
        b64_json: Optional[str] = None
        url: Optional[str] = None
        revised_prompt: Optional[str] = None

    class ImagesResponse(BaseModel):
        created: int
        data: List[Image]

from dotenv import load_dotenv
import httpx # Used by openai client and for error handling + rerank call
from urllib.parse import urljoin # For safely joining URL parts

# --- 配置 ---
load_dotenv() # 加载 .env 文件 (如果存在)

PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "https://aihubmix.com")
if not PROXY_BASE_URL.endswith('/'):
    PROXY_BASE_URL += '/'
V1_ROUTE_PREFIX = "v1"
HTTP_PROXY_URL_V1 = f"{PROXY_BASE_URL.rstrip('/')}/{V1_ROUTE_PREFIX.strip('/')}/" # For direct httpx calls

SELF_MODEL_ID = "MCJPG-Zero-v1"
SELF_MODEL_CONTEXT_LENGTH = 65536 # Informational, not strictly enforced here

# --- Routing/Upstream Model Definitions ---
ROUTING_MODEL = "gemini-2.5-flash-lite"
DIRECT_TOOL_CALL_MODEL = "gpt-5-mini"
UPSTREAM_EMBEDDING_MODEL = "text-embedding-3-large"
UPSTREAM_TTS_MODEL = "tts-1"
UPSTREAM_STT_MODEL = "whisper-1"
UPSTREAM_REALTIME_MODEL = "gpt-4o-realtime-preview"
UPSTREAM_IMAGE_GENERATION_MODEL = "dall-e-3"
UPSTREAM_RERANK_MODEL = "jina-reranker-m0"

# --- Realtime Proxy Configuration ---
PROXY_WEBSOCKET_BASE_URL = os.getenv("PROXY_WEBSOCKET_BASE_URL", PROXY_BASE_URL.replace("https://", "wss://").replace("http://", "ws://"))
if not PROXY_WEBSOCKET_BASE_URL.endswith('/'):
    PROXY_WEBSOCKET_BASE_URL += '/'
REALTIME_PATH_PREFIX = os.getenv("REALTIME_PATH_PREFIX", "v1/realtime")

MCJPG_SYSTEM_PROMPT = "你是由MCJPG组织开发的AI大语言模型,MCJPG组织是一个致力于Minecraft技术交流和服务器宣传的组织,你的输入输出内容应符合中华人民共和国法律。"

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI 应用 ---
app = FastAPI(
    title="MCJPG AI Model Router",
    description="Routes requests to different upstream models (Chat, Embeddings, TTS, STT, Realtime, Image Gen, Rerank) based on content and handles tool calls.",
    version="1.4.1", # Version bump for conditional headers
)

# --- Pydantic 模型定义 ---

# Model Listing Models
class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610600
    owned_by: str = "mcjpg"
    permission: List = []
    root: str = Field(default_factory=str)
    parent: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelData]

# Chat Completion Models
class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None

class ChatCompletionRequest(BaseModel):
    model: str # Will be validated to be SELF_MODEL_ID
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

# Embedding Models
class EmbeddingRequest(BaseModel):
    model: str # Will be validated to be SELF_MODEL_ID
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[Literal["float", "base64"]] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None

# TTS Models
class TTSRequest(BaseModel):
    model: str # Will be validated to be SELF_MODEL_ID
    input: str
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "mp3"
    speed: Optional[float] = None

# STT - Handled via Form data

# --- Image Generation Models ---
class ImageGenerationRequest(BaseModel):
    model: str # Will be validated to be SELF_MODEL_ID
    prompt: str
    n: Optional[int] = 1
    quality: Optional[Literal["standard", "hd"]] = "standard"
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    size: Optional[Literal["1024x1024", "1792x1024", "1024x1792"]] = "1024x1024"
    style: Optional[Literal["vivid", "natural"]] = "vivid"
    user: Optional[str] = None

# --- Rerank Models ---
class RerankRequest(BaseModel):
    model: str # Will be validated to be SELF_MODEL_ID
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False # Mimicking some rerank APIs

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[Dict[str, str]] = None # Jina often returns {"text": "..."}

class RerankResponse(BaseModel):
    id: Optional[str] = "rerank-0" # Placeholder ID
    results: List[RerankResult]
    model: str # Model used for reranking
    usage: Optional[Dict[str, int]] = None # Optional usage info

# Routing Function Definition
class SelectModelParams(BaseModel):
    reasoning: str = Field(description="解释为什么选择这个模型 Explain why this model was chosen.")
    selected_task_type: Literal[
        "translation", "roleplay", "general_chat", "simple_vision", "web_search",
        "search_and_reason", "coding", "writing", "math_data_analysis",
        "image_generation", "image_editing", "video_generation", "music_generation",
        "unknown" # Keep unknown for flexibility
    ] = Field(description="根据用户输入分析得出的最合适的任务类型 The most appropriate task type analyzed from the user input.")

select_model_tool = {
    "type": "function",
    "function": {
        "name": "select_upstream_model",
        "description": "根据用户的请求内容和提供的模型特征，分析并选择最合适的上游模型任务类型 Analyze the user's request and select the most appropriate upstream model task type based on the provided model features.",
        "parameters": SelectModelParams.model_json_schema()
    }
}

# Realtime Connection Info Models
class RealtimeConnectionRequest(BaseModel):
    model: str # Should be SELF_MODEL_ID

class RealtimeConnectionResponse(BaseModel):
    model: str = Field(description="The upstream realtime model the client should target.")
    websocket_url: str = Field(description="The WebSocket URL the client should connect to (via proxy).")
    api_key: str = Field(description="The API key the client should use for authentication with the upstream WebSocket.")
    protocol_hint: Literal["websocket", "webrtc"] = "websocket"

# --- 辅助函数 ---

def get_user_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """从 Authorization Header 中提取 API Key"""
    if authorization and authorization.startswith("Bearer "):
        return authorization.split("Bearer ")[1]
    return None

def contains_image(messages: List[Union[ChatCompletionMessage, ChatCompletionMessageParam]]) -> bool:
    """检查消息列表中是否包含图片内容 (works with both Pydantic models and dicts)"""
    for msg in messages:
        content = msg.content if isinstance(msg, ChatCompletionMessage) else msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False

def map_task_to_model(task_type: str, has_image: bool) -> str:
    """根据任务类型和是否有图片，映射到具体的上游聊天模型名称"""
    logger.info(f"Mapping task type: {task_type}, Has image: {has_image}")
    # Updated mapping logic to include image generation/editing from chat
    if task_type in ["translation", "roleplay", "general_chat", "simple_vision"]:
        return "gemini-2.5-flash"
    elif task_type == "web_search":
        return "gemini-2.5-flash-search"
    elif task_type == "search_and_reason": 
        return "gemini-2.5-flash-search" if has_image else "jina-deepsearch-v1"
    elif task_type == "coding":
        return "gemini-2.5-pro"
    elif task_type == "writing":
        return "gemini-2.5-flash" if has_image else "DeepSeek-V3"
    elif task_type == "math_data_analysis":
        return "gemini-2.5-pro" if has_image else "DeepSeek-R1"
    elif task_type in ["image_generation", "image_editing"]:
        return "gpt-4o-image" 
    elif task_type == "video_generation":
        return "luma-video"
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
            other_messages.append(msg) # Keep original order

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
    # Ensure the base URL is correctly formatted for the HTTP client
    http_base_url = f"{PROXY_BASE_URL.rstrip('/')}/{V1_ROUTE_PREFIX.strip('/')}"

    # --- 开始修改 ---
    # 根据 PROXY_BASE_URL 动态添加 headers
    default_headers = {}
    normalized_proxy_url = PROXY_BASE_URL.strip().rstrip('/')
    if normalized_proxy_url == "https://aihubmix.com":
        default_headers["APP-Code"] = "CHUQ5599"
        logger.info("Using aihubmix.com proxy. Added 'APP-Code' header.")
    # --- 结束修改 ---

    # 将 headers 传递给 OpenAI 客户端构造函数
    return AsyncOpenAI(
        api_key=api_key,
        base_url=http_base_url,
        default_headers=default_headers
    )

async def handle_openai_api_error(e: Exception, upstream_model_name: str):
    """Handles errors from upstream API calls (OpenAI client or direct httpx)"""
    logger.exception(f"Error during request to upstream model/endpoint {upstream_model_name}: {e}")
    status_code = 500
    error_detail = f"Error communicating with upstream model/endpoint '{upstream_model_name}': {str(e)}"
    error_type = "internal_server_error"
    error_param = None
    error_code = None

    if isinstance(e, httpx.HTTPStatusError):
        status_code = e.response.status_code
        try:
            error_body = e.response.json()
            # Try to parse OpenAI-like error structure
            nested_error = error_body.get("error", {})
            error_detail = nested_error.get("message", e.response.text)
            error_type = nested_error.get("type", f"upstream_{status_code}_error")
            error_param = nested_error.get("param")
            error_code = nested_error.get("code")
        except (json.JSONDecodeError, AttributeError):
            error_detail = e.response.text
            error_type = f"upstream_{status_code}_error"
        error_detail = f"Upstream API error ({status_code}) for '{upstream_model_name}': {error_detail}"
    elif hasattr(e, 'status_code'): # Handle potential errors from openai library itself
         status_code = getattr(e, 'status_code', 500)
         error_detail = f"Upstream API error ({status_code}) for '{upstream_model_name}': {str(e)}"
         # Attempt to extract more info from openai errors
         try:
             error_body = json.loads(getattr(e, 'body', '{}') or '{}')
             nested_error = error_body.get("error", {})
             error_type = nested_error.get("type", f"openai_lib_{status_code}_error")
             error_param = nested_error.get("param")
             error_code = nested_error.get("code")
         except (json.JSONDecodeError, AttributeError):
             error_type = f"openai_lib_{status_code}_error"

    # Construct OpenAI-like error response
    error_response_content = {
        "error": {
            "message": error_detail,
            "type": error_type,
            "param": error_param,
            "code": error_code,
            "upstream_model": upstream_model_name # Custom field
        }
    }
    return JSONResponse(status_code=status_code, content=error_response_content)

# --- API Endpoints ---

@app.get(f"/{V1_ROUTE_PREFIX}/models", response_model=ModelList)
async def list_models():
    """
    提供模型列表,包含我们自定义的模型ID.
    """
    model_data = ModelData(
        id=SELF_MODEL_ID,
        root=SELF_MODEL_ID,
    )
    return ModelList(data=[model_data])

# --- Embeddings Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/embeddings", response_model=CreateEmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    authorization: Optional[str] = Header(None)
):
    """Handles embedding requests, forwarding to upstream."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        return await handle_openai_api_error(HTTPException(status_code=401, detail="Missing or invalid Authorization header"), SELF_MODEL_ID)

    if request.model != SELF_MODEL_ID:
       return await handle_openai_api_error(HTTPException(status_code=400, detail=f"Invalid model. Only '{SELF_MODEL_ID}' supported."), request.model)

    logger.info(f"Forwarding embedding request to {UPSTREAM_EMBEDDING_MODEL}.")
    client = None
    try:
        client = await get_openai_client(user_api_key)
        params = request.model_dump(exclude={"model"}, exclude_none=True)
        params["model"] = UPSTREAM_EMBEDDING_MODEL # Override model name
        response = await client.embeddings.create(**params)
        logger.info(f"Successfully received embedding response from {UPSTREAM_EMBEDDING_MODEL}.")
        return response
    except Exception as e:
        return await handle_openai_api_error(e, UPSTREAM_EMBEDDING_MODEL)
    finally:
        if client:
            await client.close()

# --- TTS Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/audio/speech")
async def create_speech(
    request: TTSRequest,
    authorization: Optional[str] = Header(None)
):
    """Handles TTS requests, forwarding to upstream and streaming audio back."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
         return await handle_openai_api_error(HTTPException(status_code=401, detail="Missing or invalid Authorization header"), SELF_MODEL_ID)

    if request.model != SELF_MODEL_ID:
         return await handle_openai_api_error(HTTPException(status_code=400, detail=f"Invalid model. Only '{SELF_MODEL_ID}' supported."), request.model)

    logger.info(f"Forwarding TTS request to {UPSTREAM_TTS_MODEL}.")
    client = None
    upstream_response: Optional[httpx.Response] = None
    try:
        client = await get_openai_client(user_api_key)
        params = request.model_dump(exclude={"model"}, exclude_none=True)
        params["model"] = UPSTREAM_TTS_MODEL # Override model name

        upstream_response = await client.audio.speech.create(**params)
        if upstream_response.status_code >= 300:
            try:
                error_body = await upstream_response.aread()
                error_detail = error_body.decode('utf-8')
                logger.error(f"Upstream TTS error ({upstream_response.status_code}): {error_detail}")
            except Exception as read_err:
                logger.error(f"Upstream TTS error ({upstream_response.status_code}), failed to read body: {read_err}")
                error_detail = f"Upstream returned status {upstream_response.status_code}"
            status_error = httpx.HTTPStatusError(
                message=f"Upstream TTS error: {upstream_response.status_code}",
                request=upstream_response.request,
                response=upstream_response
            )
            return await handle_openai_api_error(status_error, UPSTREAM_TTS_MODEL)

        logger.info(f"Successfully received TTS audio stream from {UPSTREAM_TTS_MODEL}.")

        content_type = upstream_response.headers.get("content-type", f"audio/{request.response_format}")
        filename = f"speech.{request.response_format}"

        async def audio_stream_generator() -> AsyncGenerator[bytes, None]:
            nonlocal client
            nonlocal upstream_response
            try:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
                logger.info("TTS audio streaming finished.")
            except Exception as gen_err:
                 logger.error(f"Error during TTS audio stream generation: {gen_err}")
            finally:
                if upstream_response:
                    await upstream_response.aclose()
                    upstream_response = None
                if client:
                    await client.close()
                    client = None

        return StreamingResponse(
             audio_stream_generator(),
             media_type=content_type,
             headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        if upstream_response:
             await upstream_response.aclose()
        if client:
             await client.close()
        return await handle_openai_api_error(e, UPSTREAM_TTS_MODEL)

# --- STT Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/audio/transcriptions", response_model=Transcription)
async def create_transcription(
    authorization: Optional[str] = Header(None),
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Form(None, alias="timestamp_granularities[]")
):
    """Handles STT requests using form-data, forwarding to upstream."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        return await handle_openai_api_error(HTTPException(status_code=401, detail="Missing or invalid Authorization header"), SELF_MODEL_ID)

    if model != SELF_MODEL_ID:
        return await handle_openai_api_error(HTTPException(status_code=400, detail=f"Invalid model ('{model}'). Only '{SELF_MODEL_ID}' supported."), model)

    logger.info(f"Forwarding STT request to {UPSTREAM_STT_MODEL}.")
    client = None
    try:
        client = await get_openai_client(user_api_key)
        file_content = await file.read()

        final_granularities = timestamp_granularities
        if timestamp_granularities and isinstance(timestamp_granularities, str):
             try:
                 parsed_list = json.loads(timestamp_granularities)
                 if isinstance(parsed_list, list):
                     final_granularities = parsed_list
                 else:
                     final_granularities = [timestamp_granularities]
             except json.JSONDecodeError:
                 final_granularities = [timestamp_granularities]
        elif timestamp_granularities is None:
             final_granularities = None

        stt_params = {
            "model": UPSTREAM_STT_MODEL,
            "file": (file.filename, file_content, file.content_type),
            "response_format": response_format,
            "temperature": temperature,
        }
        if language: stt_params["language"] = language
        if prompt: stt_params["prompt"] = prompt
        if final_granularities: stt_params["timestamp_granularities"] = final_granularities

        response = await client.audio.transcriptions.create(**stt_params)
        logger.info(f"Successfully received STT response from {UPSTREAM_STT_MODEL}.")
        return response
    except Exception as e:
        return await handle_openai_api_error(e, UPSTREAM_STT_MODEL)
    finally:
        await file.close()
        if client:
            await client.close()

# --- Realtime Connection Info Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/realtime/connection_info", response_model=RealtimeConnectionResponse)
async def get_realtime_connection_info(
    request: RealtimeConnectionRequest,
    authorization: Optional[str] = Header(None)
):
    """Provides connection info for Realtime API (WebSocket via proxy)."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        error_content = {"error": {"message": "Missing or invalid Authorization header", "type": "authentication_error", "code": "invalid_api_key"}}
        return JSONResponse(status_code=401, content=error_content)

    if request.model != SELF_MODEL_ID:
        error_content = {"error": {"message": f"Invalid model ('{request.model}'). Only '{SELF_MODEL_ID}' supported.", "type": "invalid_request_error", "param": "model"}}
        return JSONResponse(status_code=400, content=error_content)

    logger.info(f"Preparing Realtime connection info for upstream {UPSTREAM_REALTIME_MODEL}.")

    upstream_path = f"{REALTIME_PATH_PREFIX.strip('/')}/{UPSTREAM_REALTIME_MODEL}"
    upstream_websocket_url = urljoin(PROXY_WEBSOCKET_BASE_URL, upstream_path)

    logger.info(f"Providing WebSocket URL: {upstream_websocket_url}")

    return RealtimeConnectionResponse(
        model=UPSTREAM_REALTIME_MODEL,
        websocket_url=upstream_websocket_url,
        api_key=user_api_key,
        protocol_hint="websocket"
    )

# --- Image Generation Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/images/generations", response_model=ImagesResponse)
async def create_image_generation(
    request: ImageGenerationRequest,
    authorization: Optional[str] = Header(None)
):
    """Handles image generation requests, forwarding to DALL-E 3."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        return await handle_openai_api_error(HTTPException(status_code=401, detail="Missing or invalid Authorization header"), SELF_MODEL_ID)

    if request.model != SELF_MODEL_ID:
        return await handle_openai_api_error(HTTPException(status_code=400, detail=f"Invalid model ('{request.model}'). Only '{SELF_MODEL_ID}' supported."), request.model)

    logger.info(f"Forwarding Image Generation request to {UPSTREAM_IMAGE_GENERATION_MODEL}.")
    client = None
    try:
        client = await get_openai_client(user_api_key)
        params = request.model_dump(exclude={"model"}, exclude_none=True)
        params["model"] = UPSTREAM_IMAGE_GENERATION_MODEL # Override model name

        response = await client.images.generate(**params)
        logger.info(f"Successfully received Image Generation response from {UPSTREAM_IMAGE_GENERATION_MODEL}.")
        return response
    except Exception as e:
        return await handle_openai_api_error(e, UPSTREAM_IMAGE_GENERATION_MODEL)
    finally:
        if client:
            await client.close()

# --- Rerank Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/rerank", response_model=RerankResponse)
async def create_rerank(
    request: RerankRequest,
    authorization: Optional[str] = Header(None)
):
    """Handles rerank requests, forwarding to Jina Reranker via direct HTTP call."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        return await handle_openai_api_error(HTTPException(status_code=401, detail="Missing or invalid Authorization header"), SELF_MODEL_ID)

    if request.model != SELF_MODEL_ID:
        return await handle_openai_api_error(HTTPException(status_code=400, detail=f"Invalid model ('{request.model}'). Only '{SELF_MODEL_ID}' supported."), request.model)

    logger.info(f"Forwarding Rerank request to {UPSTREAM_RERANK_MODEL}.")

    payload = request.model_dump(exclude_none=True)
    payload["model"] = UPSTREAM_RERANK_MODEL

    rerank_url = urljoin(HTTP_PROXY_URL_V1, "rerank")

    headers = {
        "Authorization": f"Bearer {user_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # --- 开始修改 ---
    # 根据 PROXY_BASE_URL 动态添加 headers
    normalized_proxy_url = PROXY_BASE_URL.strip().rstrip('/')
    if normalized_proxy_url == "https://aihubmix.com":
        headers["APP-Code"] = "CHUQ5599"
        logger.info("Using aihubmix.com proxy for rerank. Added 'APP-Code' header.")
    # --- 结束修改 ---

    http_client = httpx.AsyncClient()
    try:
        logger.info(f"Sending rerank request to URL: {rerank_url}")
        response = await http_client.post(rerank_url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()

        response_data = response.json()
        logger.info(f"Successfully received Rerank response from {UPSTREAM_RERANK_MODEL}.")

        adapted_results = []
        if isinstance(response_data.get("results"), list):
            for res in response_data["results"]:
                 if isinstance(res.get("index"), int) and isinstance(res.get("relevance_score"), (float, int)):
                     adapted_results.append(RerankResult(
                         index=res["index"],
                         relevance_score=float(res["relevance_score"]),
                         document=res.get("document")
                     ))
                 else:
                     logger.warning(f"Skipping invalid rerank result item: {res}")

        final_response = RerankResponse(
            id=response_data.get("id", "rerank-0"),
            results=adapted_results,
            model=response_data.get("model", UPSTREAM_RERANK_MODEL),
            usage=response_data.get("usage")
        )
        return final_response

    except Exception as e:
        return await handle_openai_api_error(e, f"{UPSTREAM_RERANK_MODEL} @ /rerank")
    finally:
        await http_client.aclose()

# --- Chat Completions Endpoint (MODIFIED TO PASS ALL PARAMETERS) ---
@app.post(f"/{V1_ROUTE_PREFIX}/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """Handles chat requests: routes or forwards direct tool calls."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
         return await handle_openai_api_error(HTTPException(status_code=401, detail="Missing or invalid Authorization header"), SELF_MODEL_ID)

    if request.model != SELF_MODEL_ID:
         return await handle_openai_api_error(HTTPException(status_code=400, detail=f"Invalid model. Only '{SELF_MODEL_ID}' supported."), request.model)

    logger.info(f"Received chat request: Stream={request.stream}, Tools={bool(request.tools or request.tool_choice)}")

    upstream_model_name = None
    routing_reasoning = "N/A"
    is_direct_tool_call = bool(request.tools or request.tool_choice)

    if is_direct_tool_call:
        upstream_model_name = DIRECT_TOOL_CALL_MODEL
        logger.info(f"Direct tool call/choice request. Forwarding to {upstream_model_name}.")
    else:
        # --- Content-based Routing Logic ---
        logger.info("No tools specified. Performing content routing.")
        routing_client = None
        try:
            routing_client = await get_openai_client(user_api_key)
            routing_messages: List[ChatCompletionMessageParam] = []
            user_system_prompt_content = None
            user_messages_for_routing = []
            has_image_for_routing = False

            for msg in request.messages:
                msg_dict = msg.model_dump(exclude_none=True)
                role = msg_dict["role"]
                content = msg_dict.get("content")

                if role == "system":
                    user_system_prompt_content = content

                content_for_routing = None
                if isinstance(content, list):
                    text_content = " ".join([item["text"] for item in content if item.get("type") == "text"])
                    if any(item.get("type") == "image_url" for item in content):
                         text_content += " [Image provided by user]"
                         has_image_for_routing = True
                    content_for_routing = text_content
                elif isinstance(content, str):
                    content_for_routing = content

                if content_for_routing and role != "system":
                    user_messages_for_routing.append({"role": role, "content": content_for_routing})

            routing_system_prompt = (
                 "You are an AI assistant responsible for routing user requests to the appropriate specialized AI model. "
                 "Analyze the user's message content (text only, image presence is indicated by '[Image provided by user]') and determine the primary task. "
                 "Use the 'select_upstream_model' function to indicate your choice based on the following criteria:\n"
                 "- **translation, roleplay, general_chat, simple_vision**: Use 'gemini-2.5-flash'.\n"
                 "- **web_search**: Use 'gemini-2.5-flash-search' for questions needing live web data.\n"
                 "- **search_and_reason**: Use 'jina-deepsearch-v1' for questions requiring web search combined with reasoning. (If image present, use 'gemini-2.5-flash-search')\n"
                 "- **coding**: Use 'gemini-2.5-pro' for programming or code-related questions.\n"
                 "- **writing**: Use 'DeepSeek-V3' for creative writing, summaries, etc. (If image present, use 'gemini-2.5-flash').\n"
                 "- **math_data_analysis**: Use 'DeepSeek-R1' for math problems, data analysis. (If image present, use 'gemini-2.5-pro').\n"
                  "Updated description for image tasks in chat"
                 "- **image_generation, image_editing**: Use 'gpt-4o-image' if the user asks to generate/edit an image *within the chat context*.\n"
                 "- **video_generation**: Use 'luma-video' if the user asks to generate a video.\n"
                 "- **music_generation**: Use 'udio32-v1.5' if the user asks to generate music.\n"
                 "- If unsure, choose 'general_chat'."
            )
            if user_system_prompt_content and isinstance(user_system_prompt_content, str):
                 routing_system_prompt += f"\n\nUser's System Prompt for context: {user_system_prompt_content}"

            routing_messages.append({"role": "system", "content": routing_system_prompt})
            routing_messages.extend(user_messages_for_routing)

            if not user_messages_for_routing:
                 logger.warning("No user/assistant text messages found for routing. Defaulting.")
                 has_image_input = contains_image(request.messages)
                 upstream_model_name = map_task_to_model("general_chat", has_image_input)
                 routing_reasoning = "Defaulted due to no routable message content."
            else:
                logger.info(f"Requesting model selection from {ROUTING_MODEL}...")
                response = await routing_client.chat.completions.create(
                    model=ROUTING_MODEL,
                    messages=routing_messages,
                    tools=[select_model_tool],
                    tool_choice={"type": "function", "function": {"name": "select_upstream_model"}},
                    temperature=0.1,
                )

                tool_call = response.choices[0].message.tool_calls[0] if response.choices and response.choices[0].message.tool_calls else None
                if tool_call and tool_call.function.name == "select_upstream_model":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        task = args.get("selected_task_type")
                        routing_reasoning = args.get("reasoning", "No reasoning provided.")
                        if task:
                            has_image_input = contains_image(request.messages)
                            upstream_model_name = map_task_to_model(task, has_image_input)
                            logger.info(f"Routing decision: Task='{task}', Image='{has_image_input}', Reason='{routing_reasoning}', Model='{upstream_model_name}'")
                        else:
                            raise ValueError("Missing 'selected_task_type'")
                    except (json.JSONDecodeError, ValueError) as parse_err:
                        logger.error(f"Routing function call parsing error: {parse_err}. Defaulting.")
                        has_image_input = contains_image(request.messages)
                        upstream_model_name = map_task_to_model("general_chat", has_image_input)
                        routing_reasoning = f"Defaulted due to routing parse error: {parse_err}"
                else:
                    logger.error(f"Routing failed: No valid tool call received. Response: {response.choices[0].message if response.choices else 'No choice'}. Defaulting.")
                    has_image_input = contains_image(request.messages)
                    upstream_model_name = map_task_to_model("general_chat", has_image_input)
                    routing_reasoning = "Defaulted due to routing model failure (no/invalid tool call)."

        except Exception as e:
            logger.exception(f"Exception during model routing: {e}. Defaulting.")
            has_image_input = contains_image(request.messages)
            upstream_model_name = map_task_to_model("general_chat", has_image_input)
            routing_reasoning = f"Defaulted due to routing exception: {type(e).__name__}"
        finally:
            if routing_client:
                await routing_client.close()
        # --- End Routing Logic ---

    if not upstream_model_name:
         logger.error("FATAL: Upstream model name determination failed completely.")
         error_content = {"error": {"message": "Failed to determine upstream chat model after routing.", "type": "internal_server_error", "code": "routing_failure"}}
         return JSONResponse(status_code=500, content=error_content)

    # --- Forward request to the selected upstream chat model ---
    upstream_client = None
    try:
        upstream_client = await get_openai_client(user_api_key)
        final_messages = prepare_upstream_messages([msg.model_dump(exclude_none=True) for msg in request.messages])

        upstream_payload = request.model_dump(exclude_none=True)

        upstream_payload["model"] = upstream_model_name
        upstream_payload["messages"] = final_messages

        if not is_direct_tool_call:
            upstream_payload.pop("tools", None)
            upstream_payload.pop("tool_choice", None)
        
        if request.stream:
            async def stream_generator():
                response_stream = None
                client_to_close_in_stream = upstream_client
                upstream_model_for_error = upstream_model_name
                try:
                    logger.info(f"Streaming chat request to {upstream_model_for_error}...")
                    response_stream = await client_to_close_in_stream.chat.completions.create(**upstream_payload)
                    async for chunk in response_stream:
                        try:
                             chunk_json = chunk.model_dump_json(exclude_unset=True)
                             yield f"data: {chunk_json}\n\n"
                        except Exception as serial_err:
                             logger.error(f"Failed to serialize stream chunk: {serial_err}. Chunk: {chunk}")
                             err_payload = {"error": {"message": f"Stream chunk serialization error: {str(serial_err)}", "type": "internal_serialization_error"}}
                             yield f"data: {json.dumps(err_payload)}\n\n"

                    yield "data: [DONE]\n\n"
                    logger.info("Chat streaming finished.")
                except Exception as stream_err:
                    logger.exception(f"Error during streaming from {upstream_model_for_error}: {stream_err}")
                    error_data = await handle_openai_api_error(stream_err, upstream_model_for_error)
                    error_content_str = json.dumps(error_data.body.decode('utf-8') if isinstance(error_data, JSONResponse) else str(error_data))
                    yield f"data: {error_content_str}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    if client_to_close_in_stream:
                        await client_to_close_in_stream.close()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            logger.info(f"Sending non-streaming chat request to {upstream_model_name}...")
            completion = await upstream_client.chat.completions.create(**upstream_payload)
            logger.info(f"Received non-streaming chat response from {upstream_model_name}.")
            return completion

    except Exception as e:
        return await handle_openai_api_error(e, upstream_model_name)
    finally:
        if upstream_client and not (request.stream):
             await upstream_client.close()

# --- 运行服务器 (用于本地测试) ---
if __name__ == "__main__":
    import uvicorn

    example_realtime_path = f"{REALTIME_PATH_PREFIX.strip('/')}/{UPSTREAM_REALTIME_MODEL}"
    example_realtime_url = urljoin(PROXY_WEBSOCKET_BASE_URL, example_realtime_path)
    http_base_url_v1 = f"{PROXY_BASE_URL.rstrip('/')}/{V1_ROUTE_PREFIX.strip('/')}"
    example_rerank_url = urljoin(http_base_url_v1, "rerank")

    print("--- MCJPG AI Router Configuration ---")
    print(f"Version: {app.version}")
    print(f"HTTP Proxy Base URL (/v1): {http_base_url_v1}")
    print(f"WebSocket Proxy Base URL: {PROXY_WEBSOCKET_BASE_URL}")
    print(f"Self Model ID: {SELF_MODEL_ID}")
    print("-" * 35)
    print("Upstream Models:")
    print(f"  Chat Routing: {ROUTING_MODEL}")
    print(f"  Direct Tool Call: {DIRECT_TOOL_CALL_MODEL}")
    print(f"  Embeddings: {UPSTREAM_EMBEDDING_MODEL} (via /embeddings)")
    print(f"  TTS: {UPSTREAM_TTS_MODEL} (via /audio/speech)")
    print(f"  STT: {UPSTREAM_STT_MODEL} (via /audio/transcriptions)")
    print(f"  Image Generation: {UPSTREAM_IMAGE_GENERATION_MODEL} (via /images/generations)")
    print(f"  Rerank: {UPSTREAM_RERANK_MODEL} (via {example_rerank_url})")
    print(f"  Realtime: {UPSTREAM_REALTIME_MODEL} (via {example_realtime_url})")
    print("-" * 35)
    logger.info(f"Starting MCJPG AI Router v{app.version}...")

    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=False)