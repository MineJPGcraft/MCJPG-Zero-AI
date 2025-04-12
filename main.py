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
from openai.types.create_embedding_response import CreateEmbeddingResponse, Embedding
from openai.types.audio import Transcription
from dotenv import load_dotenv
import httpx # Used by openai client and for error handling
from urllib.parse import urljoin # For safely joining URL parts

# --- 配置 ---
load_dotenv() # 加载 .env 文件 (如果存在)

PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "https://proxy.mcjpg.org:29678")
if not PROXY_BASE_URL.endswith('/'):
    PROXY_BASE_URL += '/'
V1_ROUTE_PREFIX = "v1"

SELF_MODEL_ID = "MCJPG-Zero-v1"
SELF_MODEL_CONTEXT_LENGTH = 65536 # Informational, not strictly enforced here

# --- Routing/Upstream Model Definitions ---
ROUTING_MODEL = "gemini-2.0-flash"
DIRECT_TOOL_CALL_MODEL = "gpt-4o-mini"
UPSTREAM_EMBEDDING_MODEL = "text-embedding-3-large"
UPSTREAM_TTS_MODEL = "tts-1"
UPSTREAM_STT_MODEL = "whisper-1"
UPSTREAM_REALTIME_MODEL = "gpt-4o-realtime-preview"

# --- Realtime Proxy Configuration ---
# Base URL for WebSocket connections via the proxy
PROXY_WEBSOCKET_BASE_URL = os.getenv("PROXY_WEBSOCKET_BASE_URL", PROXY_BASE_URL.replace("https://", "wss://").replace("http://", "ws://"))
if not PROXY_WEBSOCKET_BASE_URL.endswith('/'):
    PROXY_WEBSOCKET_BASE_URL += '/'
# Path prefix on the proxy for realtime endpoints
REALTIME_PATH_PREFIX = os.getenv("REALTIME_PATH_PREFIX", "v1/realtime") # Example: "v1/realtime" -> wss://proxy.../v1/realtime/model-name

MCJPG_SYSTEM_PROMPT = "你是由MCJPG组织开发的AI大语言模型,MCJPG组织是一个致力于Minecraft技术交流和服务器宣传的组织,你的输入输出内容应符合中华人民共和国法律。"

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI 应用 ---
app = FastAPI(
    title="MCJPG AI Model Router",
    description="Routes requests to different upstream models (Chat, Embeddings, TTS, STT, Realtime) based on content and handles tool calls.",
    version="1.3.0", # Version bump for Realtime support
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

# Routing Function Definition
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
    # (Mapping logic remains the same as before)
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
    return AsyncOpenAI(api_key=api_key, base_url=http_base_url)


async def handle_openai_api_error(e: Exception, upstream_model_name: str):
    """Handles errors from upstream API calls"""
    logger.exception(f"Error during request to upstream model {upstream_model_name}: {e}")
    status_code = 500
    error_detail = f"Error communicating with upstream model '{upstream_model_name}': {str(e)}"

    if isinstance(e, httpx.HTTPStatusError):
        status_code = e.response.status_code
        try:
            error_body = e.response.json()
            error_detail = error_body.get("error", {}).get("message", e.response.text)
        except json.JSONDecodeError:
            error_detail = e.response.text
        error_detail = f"Upstream API error ({status_code}) for model '{upstream_model_name}': {error_detail}"
    elif hasattr(e, 'status_code'): # Handle potential errors from openai library itself
         status_code = getattr(e, 'status_code')
         error_detail = f"Upstream API error ({status_code}) for model '{upstream_model_name}': {str(e)}"

    raise HTTPException(status_code=status_code, detail=error_detail)


# --- API Endpoints ---

@app.get(f"/{V1_ROUTE_PREFIX}/models", response_model=ModelList)
async def list_models():
    """
    提供模型列表,包含我们自定义的模型ID.
    Note: Listing MCJPG-Zero-v1 implies it can be used with compatible endpoints:
    - /chat/completions (routed based on content)
    - /embeddings (forwarded to text-embedding-3-large)
    - /audio/speech (forwarded to tts-1)
    - /audio/transcriptions (forwarded to whisper-1)
    - Realtime API interaction (initiated via /realtime/connection_info, forwarded to gpt-4o-realtime-preview)
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
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if request.model != SELF_MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Invalid model. Only '{SELF_MODEL_ID}' supported.")

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
        await handle_openai_api_error(e, UPSTREAM_EMBEDDING_MODEL)
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
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if request.model != SELF_MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Invalid model. Only '{SELF_MODEL_ID}' supported.")

    logger.info(f"Forwarding TTS request to {UPSTREAM_TTS_MODEL}.")
    client = None
    upstream_response: Optional[httpx.Response] = None
    try:
        client = await get_openai_client(user_api_key)
        params = request.model_dump(exclude={"model"}, exclude_none=True)
        params["model"] = UPSTREAM_TTS_MODEL # Override model name

        upstream_response = await client.audio.speech.create(**params)
        upstream_response.raise_for_status() # Check for upstream errors

        logger.info(f"Successfully received TTS audio stream from {UPSTREAM_TTS_MODEL}.")

        content_type = upstream_response.headers.get("content-type", f"audio/{request.response_format}")
        filename = f"speech.{request.response_format}"

        async def audio_stream_generator() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
                logger.info("TTS audio streaming finished.")
            finally:
                # Ensure upstream response is closed
                if upstream_response:
                    await upstream_response.aclose()
                # Close client *after* streaming is done
                if client:
                    await client.close()
                    # client = None # Prevent double close in outer finally

        return StreamingResponse(
             audio_stream_generator(),
             media_type=content_type,
             headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        # Close resources if error occurred before streaming generator took over
        if upstream_response:
             await upstream_response.aclose()
        if client:
             await client.close()
        # Let error handler raise HTTPException
        await handle_openai_api_error(e, UPSTREAM_TTS_MODEL)
    # Note: No finally block needed here as resources are closed within try/except or by generator

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
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Form(None) # Note: SDK expects List type
):
    """Handles STT requests using form-data, forwarding to upstream."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if model != SELF_MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Invalid model ('{model}'). Only '{SELF_MODEL_ID}' supported.")

    logger.info(f"Forwarding STT request to {UPSTREAM_STT_MODEL}.")
    client = None
    try:
        client = await get_openai_client(user_api_key)
        file_content = await file.read() # Read file content

        stt_params = {
            "model": UPSTREAM_STT_MODEL,
            "file": (file.filename, file_content, file.content_type),
            "response_format": response_format,
            "temperature": temperature,
        }
        # Add optional parameters if provided by user
        if language: stt_params["language"] = language
        if prompt: stt_params["prompt"] = prompt
        if timestamp_granularities: stt_params["timestamp_granularities"] = timestamp_granularities # Pass as list

        response = await client.audio.transcriptions.create(**stt_params)
        logger.info(f"Successfully received STT response from {UPSTREAM_STT_MODEL}.")
        return response
    except Exception as e:
        await handle_openai_api_error(e, UPSTREAM_STT_MODEL)
    finally:
        await file.close() # Ensure uploaded file handle is closed
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
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if request.model != SELF_MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Invalid model ('{request.model}'). Only '{SELF_MODEL_ID}' supported.")

    logger.info(f"Preparing Realtime connection info for upstream {UPSTREAM_REALTIME_MODEL}.")

    # Construct the upstream WebSocket URL via the configured proxy base and path
    # Example: wss://proxy.mcjpg.org:29678/v1/realtime/gpt-4o-realtime-preview
    # Use urljoin for robust path combination
    upstream_path = f"{REALTIME_PATH_PREFIX.strip('/')}/{UPSTREAM_REALTIME_MODEL}"
    upstream_websocket_url = urljoin(PROXY_WEBSOCKET_BASE_URL, upstream_path)

    logger.info(f"Providing WebSocket URL: {upstream_websocket_url}")

    return RealtimeConnectionResponse(
        model=UPSTREAM_REALTIME_MODEL,
        websocket_url=upstream_websocket_url,
        api_key=user_api_key, # Client needs this key to authenticate the WS connection
        protocol_hint="websocket"
    )

# --- Chat Completions Endpoint ---
@app.post(f"/{V1_ROUTE_PREFIX}/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """Handles chat requests: routes or forwards direct tool calls."""
    user_api_key = get_user_api_key(authorization)
    if not user_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if request.model != SELF_MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Invalid model. Only '{SELF_MODEL_ID}' supported.")

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
            routing_client = await get_openai_client(user_api_key) # Use same base URL/proxy
            routing_messages: List[ChatCompletionMessageParam] = []
            user_system_prompt_content = None
            user_messages_for_routing = []
            has_image_for_routing = False # Track if image placeholder was added

            # Prepare messages for the routing model (text only)
            for msg in request.messages:
                msg_dict = msg.model_dump(exclude_none=True)
                role = msg_dict["role"]
                content = msg_dict.get("content")

                if role == "system":
                    user_system_prompt_content = content # Capture user system prompt text/list

                content_for_routing = None
                if isinstance(content, list):
                    text_content = " ".join([item["text"] for item in content if item.get("type") == "text"])
                    if any(item.get("type") == "image_url" for item in content):
                         text_content += " [Image provided by user]"
                         has_image_for_routing = True # Needed for routing logic
                    content_for_routing = text_content
                elif isinstance(content, str):
                    content_for_routing = content

                # Add msg if it has suitable content for routing analysis
                if content_for_routing and role != "system": # Exclude system here, add custom below
                    user_messages_for_routing.append({"role": role, "content": content_for_routing})

            # Build the routing prompt
            routing_system_prompt = (
                 "You are an AI assistant responsible for routing user requests to the appropriate specialized AI model. "
                 "Analyze the user's message content (text only, image presence is indicated by '[Image provided by user]') and determine the primary task. "
                 "Use the 'select_upstream_model' function to indicate your choice based on the following criteria:\n"
                # (Criteria remain the same as before)
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
            # Vision system prompts are ignored for routing for simplicity

            routing_messages.append({"role": "system", "content": routing_system_prompt})
            routing_messages.extend(user_messages_for_routing)

            # Handle cases with no user messages for routing (e.g., only system prompt)
            if not user_messages_for_routing:
                 logger.warning("No user/assistant text messages found for routing. Defaulting.")
                 has_image_input = contains_image(request.messages) # Check original messages for image
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

                # Process routing result or default
                tool_call = response.choices[0].message.tool_calls[0] if response.choices and response.choices[0].message.tool_calls else None
                if tool_call and tool_call.function.name == "select_upstream_model":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        task = args.get("selected_task_type")
                        routing_reasoning = args.get("reasoning", "No reasoning provided.")
                        if task:
                            # Check original messages for image presence when mapping
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
            # Log routing error but default instead of failing the request
            logger.exception(f"Exception during model routing: {e}. Defaulting.")
            has_image_input = contains_image(request.messages)
            upstream_model_name = map_task_to_model("general_chat", has_image_input)
            routing_reasoning = f"Defaulted due to routing exception: {type(e).__name__}"
        finally:
            if routing_client:
                await routing_client.close()
        # --- End Routing Logic ---

    if not upstream_model_name:
         # Should be covered by defaults, but as a final fallback
         logger.error("FATAL: Upstream model name determination failed completely.")
         raise HTTPException(status_code=500, detail="Failed to determine upstream chat model after routing.")

    # --- Forward request to the selected upstream chat model ---
    upstream_client = None
    try:
        upstream_client = await get_openai_client(user_api_key)
        # Prepare final messages including the MCJPG system prompt
        final_messages = prepare_upstream_messages([msg.model_dump(exclude_none=True) for msg in request.messages])

        # Prepare payload, only include tools/choice if it was a direct tool call request
        upstream_payload = {
            "model": upstream_model_name,
            "messages": final_messages,
            "stream": request.stream,
            "tools": request.tools if is_direct_tool_call else None,
            "tool_choice": request.tool_choice if is_direct_tool_call else None,
            # Pass through other standard parameters
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
                client_to_close_in_stream = upstream_client # Capture client instance
                try:
                    logger.info(f"Streaming chat request to {upstream_model_name}...")
                    response_stream = await client_to_close_in_stream.chat.completions.create(**upstream_payload)
                    async for chunk in response_stream:
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                    yield "data: [DONE]\n\n"
                    logger.info("Chat streaming finished.")
                except Exception as stream_err:
                    logger.exception(f"Error during streaming from {upstream_model_name}: {stream_err}")
                    # Send error details in the stream if possible (non-standard)
                    err_payload = {"error": {"message": f"Upstream stream error: {str(stream_err)}", "type": "upstream_error"}}
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    yield "data: [DONE]\n\n" # Still need DONE message
                finally:
                    # Close the client *after* the stream is fully consumed or errors out
                    if client_to_close_in_stream:
                        await client_to_close_in_stream.close()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # --- Handle Non-Streaming Response ---
            logger.info(f"Sending non-streaming chat request to {upstream_model_name}...")
            completion = await upstream_client.chat.completions.create(**upstream_payload)
            logger.info(f"Received non-streaming chat response from {upstream_model_name}.")
            # Optionally add routing info (non-standard)
            # completion_dict = completion.model_dump(exclude_unset=True)
            # completion_dict["routing_info"] = {"model_selected": upstream_model_name, "reasoning": routing_reasoning}
            # return completion_dict
            return completion.model_dump(exclude_unset=True) # Return standard response

    except Exception as e:
        # Handle errors during request setup or non-streaming call
        await handle_openai_api_error(e, upstream_model_name)
    finally:
        # Close client only if it wasn't passed to a successful stream generator
        if upstream_client and not request.stream:
             await upstream_client.close()


# --- 运行服务器 (用于本地测试) ---
if __name__ == "__main__":
    import uvicorn

    # Construct example Realtime URL for logging
    example_realtime_path = f"{REALTIME_PATH_PREFIX.strip('/')}/{UPSTREAM_REALTIME_MODEL}"
    example_realtime_url = urljoin(PROXY_WEBSOCKET_BASE_URL, example_realtime_path)
    http_base_url = f"{PROXY_BASE_URL.rstrip('/')}/{V1_ROUTE_PREFIX.strip('/')}"


    print("--- MCJPG AI Router Configuration ---")
    print(f"Version: {app.version}")
    print(f"HTTP Proxy Base URL: {http_base_url}")
    print(f"WebSocket Proxy Base URL: {PROXY_WEBSOCKET_BASE_URL}")
    print(f"Self Model ID: {SELF_MODEL_ID}")
    print("-" * 35)
    print("Upstream Models:")
    print(f"  Chat Routing: {ROUTING_MODEL}")
    print(f"  Direct Tool Call: {DIRECT_TOOL_CALL_MODEL}")
    print(f"  Embeddings: {UPSTREAM_EMBEDDING_MODEL}")
    print(f"  TTS: {UPSTREAM_TTS_MODEL}")
    print(f"  STT: {UPSTREAM_STT_MODEL}")
    print(f"  Realtime: {UPSTREAM_REALTIME_MODEL} (via {example_realtime_url})")
    print("-" * 35)
    logger.info(f"Starting MCJPG AI Router v{app.version}...")

    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)