# MCJPG Zero AI

[![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) <!-- 确保你有一个 LICENSE 文件 -->
![Python 版本](https://img.shields.io/badge/python-3.8+-blue.svg)
![框架](https://img.shields.io/badge/Framework-FastAPI-green.svg)

一个 FastAPI 应用，作为处理多种 AI 任务（聊天、视觉识别、深度思考、实时对话、嵌入、文本转语音、语音转文本、图像生成、重排序）的智能路由。它提供了一个统一的模型接口 (`MCJPG-Zero-v1`)，同时根据请求类型或内容分析，将请求定向到最优的上游 OpenAI 兼容模型。由 MCJPG 组织开发。

## 概述

本项目旨在解决与不同任务的多个专用 AI 模型交互的复杂性。用户无需了解应调用哪个具体模型（例如，`gpt-5-mini` 用于工具调用，`tts-1` 用于语音合成，`dall-e-3` 用于图像生成，`jina-reranker-v2-base-multilingual` 用于重排序，`gpt-4o-realtime-preview` 用于实时交互），此路由提供了一个单一、一致的模型 ID：`MCJPG-Zero-v1`。

该路由智能地将请求转发到部署在代理（如 `aihubmix.com`,我们默认使用[ AIHubmix ](https://aihubmix.com?aff=vba5) 作为AI提供商,支持所有模型，同时为使用此项目的所有用户提供了 **10%** 的全场优惠,您也可以使用[ MCJPG API ](https://chatapi.mcjpg.org)来支持我们的开发）后的适当上游后端模型，透传用户提供的令牌进行验证，简化了 API 的使用，并允许灵活管理后端模型，而无需更改面向客户端的接口。

## 主要特性

*   **统一模型 ID:** 使用单一的 `MCJPG-Zero-v1` 模型 ID 与所有支持的模态进行交互。
*   **OpenAI API 兼容:** 遵循 OpenAI API 请求/响应格式，可与现有工具和库无缝集成（HTTP 部分）。
*   **智能聊天路由:** 分析聊天消息内容（仅文本），为编码、写作、网络搜索、通用聊天等任务选择最合适的上游模型。
*   **直接请求转发:** 将特定类型的请求直接路由到指定的上游模型：
    *   包含 `tools` 或 `tool_choice` 的聊天请求将发送到支持工具调用的模型（例如 `gpt-4.1-mini`）。
    *   嵌入请求 (`/v1/embeddings`) 将发送到文本嵌入模型（例如 `text-embedding-3-large`）。
    *   文本转语音 (TTS) 请求 (`/v1/audio/speech`) 将发送到 TTS 模型（例如 `tts-1`）。
    *   语音转文本 (STT) 请求 (`/v1/audio/transcriptions`) 将发送到 STT 模型（例如 `whisper-1`）。
    *   图像生成请求 (`/v1/images/generations`) 将发送到图像生成模型（例如 `dall-e-3`）。
    *   重排序请求 (`/v1/rerank`) 将发送到重排序模型（例如 `jina-reranker-v2-base-multilingual`）。
*   **实时 API 支持 (Realtime):** 提供连接信息 (`/v1/realtime/connection_info` 端点)，使客户端能通过代理与上游实时模型（例如 `gpt-4o-realtime-preview`）建立 WebSocket 连接。**注意：本路由不直接代理 WebSocket 流量，而是提供连接所需的信息。**
*   **流式支持:** 处理聊天完成（Chat Completions）和文本转语音（Text-to-Speech）的 HTTP 流式响应。
*   **系统提示词注入:** 自动在聊天请求前添加可配置的系统提示词（例如，定义 AI 源自 MCJPG），同时尊重用户提供的系统提示词。
*   **可配置:** 用于路由/转发的上游代理 URL (HTTP 和 WebSocket) 和特定上游模型名称可以通过环境变量轻松配置。

## 工作原理

1.  **接收请求:** 用户向路由发送 API 请求（聊天、嵌入、TTS、STT、**图像生成**、**重排序**、**实时连接信息**），指定 `model: "MCJPG-Zero-v1"`。请求必须包含有效的 `Authorization: Bearer <用户API密钥>` 头。
2.  **端点识别:** 路由识别目标端点（例如 `/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`, `/v1/rerank`, `/v1/realtime/connection_info`）。
3.  **路由/转发逻辑:**
    *   **嵌入、TTS、STT、图像生成:** 请求直接映射到预先配置的上游模型 (`UPSTREAM_EMBEDDING_MODEL`, `UPSTREAM_TTS_MODEL`, `UPSTREAM_STT_MODEL`, `UPSTREAM_IMAGE_GENERATION_MODEL`)，并通过 `PROXY_BASE_URL` (HTTP) 转发。
    *   **重排序:** 请求映射到预先配置的 `UPSTREAM_RERANK_MODEL`，并通过 **直接 HTTP 调用** 发送到代理服务器上的 `/v1/rerank` 路径。
    *   **聊天完成:**
        *   如果存在 `tools` 或 `tool_choice`，请求将转发到 `DIRECT_TOOL_CALL_MODEL`。
        *   否则，路由提取文本内容，调用 `ROUTING_MODEL`（例如 `gemini-2.0-flash`）并使用特定的函数调用指令 (`select_upstream_model`) 来确定任务类型（编码、写作等）。
        *   路由根据确定的任务类型（以及输入是否包含图像）和预定义规则，映射到合适的上游聊天模型。
        *   如果路由失败，则默认为通用的聊天模型。
        *   聊天请求通过 `PROXY_BASE_URL` (HTTP) 转发。
    *   **实时连接信息 (`/v1/realtime/connection_info`):**
        *   路由验证模型 ID (`MCJPG-Zero-v1`) 和用户 API 密钥。
        *   使用配置的 `PROXY_WEBSOCKET_BASE_URL` 和 `REALTIME_PATH_PREFIX` 以及 `UPSTREAM_REALTIME_MODEL` 构造目标 WebSocket URL。
        *   返回包含目标 WebSocket URL、上游模型名称 (`UPSTREAM_REALTIME_MODEL`) 和用户 API 密钥的 JSON 响应。
4.  **请求准备 (HTTP):** 对于 HTTP 请求，路由为选定的上游模型构建最终的请求负载。对于聊天，它会注入 `MCJPG_SYSTEM_PROMPT` 并使用原始消息。对于其他类型，它使用映射后的上游模型 ID。对于重排序，它使用原始请求体但替换模型 ID。
5.  **上游调用 (HTTP):** 路由通过配置的 `PROXY_BASE_URL` (或 `/v1/rerank` 路径) 将准备好的 HTTP 请求发送到上游模型/服务，并传递用户的 API 密钥。
6.  **响应处理:**
    *   **HTTP:** 路由从上游模型接收响应，并将其流式传输或直接返回给原始用户。
    *   **Realtime:** 路由将连接信息返回给客户端。**客户端需要负责使用返回的 URL 和 API 密钥，通过 WebSocket 协议连接到指定的上游实时模型（经由代理）。**

## 技术栈

*   **Python:** 核心编程语言。
*   **FastAPI:** 用于构建 API 的高性能 Web 框架。
*   **Pydantic:** 数据验证和设置管理。
*   **OpenAI Python SDK:** 用于与上游 OpenAI 兼容 API 交互（包括类型定义）。
*   **HTTPX:** OpenAI SDK 使用的底层异步 HTTP 客户端，也用于直接调用 Rerank API。
*   **python-dotenv:** 用于从 `.env` 文件加载环境变量。
*   **uvicorn:** 用于运行 FastAPI 应用程序的 ASGI 服务器。
*   **python-multipart:** 处理文件上传（STT）所需。
*   **websockets (或类似库):** *客户端*需要使用此库来处理 Realtime API 的 WebSocket 连接 (本路由服务本身不直接依赖)。

## 安装与设置

1.  **先决条件:**
    *   Python 3.8 或更高版本
    *   pip (Python 包安装器)
    *   Git (用于克隆仓库)

2.  **克隆仓库:**
    ```bash
    git clone [你的 GitHub 仓库链接]
    cd mcjpg-ai-router # 或者你的仓库目录名
    ```

3.  **安装依赖:**
    ```bash
    pip install "fastapi[all]" uvicorn python-dotenv openai httpx python-multipart
    # 或者，如果你创建了 requirements.txt:
    # pip install -r requirements.txt
    ```
    *(建议创建一个 `requirements.txt` 文件以便于依赖管理)*

4.  **配置环境变量:**
    在项目的根目录下创建一个名为 `.env` 的文件。添加以下变量：

    ```dotenv
    # --- 基础配置 (必需) ---
    # 上游 OpenAI 兼容代理的 HTTP 基础 URL (必须以 / 结尾)
    # 例如：PROXY_BASE_URL=https://proxy.mcjpg.org:29678/
    PROXY_BASE_URL=https://proxy.mcjpg.org:29678/

    # --- 实时 API 配置 (可选) ---
    # 上游 WebSocket 代理的基础 URL (用于实时 API, 必须以 / 结尾)
    # 如果未设置，将尝试从 PROXY_BASE_URL 推断 (https -> wss, http -> ws)
    # 例如：PROXY_WEBSOCKET_BASE_URL=wss://proxy.mcjpg.org:29678/
    # PROXY_WEBSOCKET_BASE_URL=

    # 代理上实时 WebSocket 端点的路径前缀 (不含模型名称, 不以 / 开头或结尾)
    # 默认值: "v1/realtime"
    # 最终 URL 示例: wss://proxy.../v1/realtime/gpt-4o-realtime-preview
    # REALTIME_PATH_PREFIX=v1/realtime

    # --- 上游模型名称配置 (可选, 覆盖代码中的默认值) ---
    # 用于内容路由的决策模型
    # ROUTING_MODEL=gemini-2.5-flash-preview-04-17
    # 处理工具调用的聊天模型
    # DIRECT_TOOL_CALL_MODEL=gpt-4.1-mini
    # 嵌入模型
    # UPSTREAM_EMBEDDING_MODEL=text-embedding-3-large
    # 文本转语音 (TTS) 模型
    # UPSTREAM_TTS_MODEL=tts-1
    # 语音转文本 (STT) 模型
    # UPSTREAM_STT_MODEL=whisper-1
    # 图像生成模型
    # UPSTREAM_IMAGE_GENERATION_MODEL=dall-e-3
    # 重排序模型
    # UPSTREAM_RERANK_MODEL=jina-reranker-v2-base-multilingual
    # 实时 API 模型
    # UPSTREAM_REALTIME_MODEL=gpt-4o-realtime-preview
    ```
    *   `PROXY_BASE_URL` **至关重要**。
    *   `PROXY_WEBSOCKET_BASE_URL` 和 `REALTIME_PATH_PREFIX` 对 Realtime API 很重要。
    *   **你可以通过设置上述 `UPSTREAM_...` 变量来更改路由将请求转发到的具体上游模型名称，而无需修改 Python 代码。** 例如，如果你想使用 `dall-e-2` 而不是 `dall-e-3` 进行图像生成，只需在 `.env` 文件中添加 `UPSTREAM_IMAGE_GENERATION_MODEL=dall-e-2`。

5.  **运行服务器:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8005 --reload
    ```
    路由现在应该运行在 `http://<你的服务器IP>:8005`。

## 使用方法

使用标准的 OpenAI API 客户端库或 `curl` 与路由进行交互（HTTP 部分），始终指定 `model: "MCJPG-Zero-v1"` 并在 `Authorization` 头中提供你的上游 API 密钥。

**重要:** 路由本身不处理身份验证，它只是要求 `Authorization` 头，并将其直接传递给上游 `PROXY_BASE_URL` (HTTP) 或在 Realtime 连接信息中返回给客户端。请确保你的代理正确处理身份验证。

**API 调用示例:**

将 `<你的API密钥>` 替换为你的 `PROXY_BASE_URL` 的有效 API 密钥，并将 `<路由URL>` 替换为你的路由运行的地址（例如 `http://127.0.0.1:8005`）。

**1. 聊天 (简单请求):**
```bash
curl -X POST "<路由URL>/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "messages": [
         {"role": "user", "content": "你好，你是谁？"}
       ]
     }'
```

**2. 聊天 (流式输出):**
```bash
curl -X POST "<路由URL>/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "messages": [
         {"role": "user", "content": "写一个关于Minecraft中红石基础的简短教程"}
       ],
       "stream": true
     }'
```

**3. 嵌入向量 (Embeddings):**
```bash
curl -X POST "<路由URL>/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "input": "我的世界服务器"
     }'
```

**4. 文本转语音 (TTS):**
```bash
curl -X POST "<路由URL>/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "input": "欢迎来到MCJPG！",
       "voice": "alloy"
     }' \
     --output speech.mp3
```

**5. 语音转文本 (STT):**
```bash
curl -X POST "<路由URL>/v1/audio/transcriptions" \
     -H "Authorization: Bearer <你的API密钥>" \
     -F file=@"/path/to/your/audio.mp3" \
     -F model="MCJPG-Zero-v1"
```

**6. 图像生成 (Image Generation):**
```bash
curl -X POST "<路由URL>/v1/images/generations" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "prompt": "一只可爱的Minecraft风格苦力怕在草地上",
       "n": 1,
       "size": "1024x1024"
     }'
```

**7. 重排序 (Rerank):**
```bash
curl -X POST "<路由URL>/v1/rerank" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "query": "最佳的Minecraft红石教程",
       "documents": [
         "一个关于早期红石电路的教程",
         "Minecraft红石入门指南",
         "高级红石技巧和自动化农场",
         "如何在Minecraft中建造简单的门"
       ],
       "top_n": 3
     }'
```

**8. 获取实时 API 连接信息 (Realtime):**

*   **第一步：调用路由获取连接信息**

    ```bash
    curl -X POST "<路由URL>/v1/realtime/connection_info" \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer <你的API密钥>" \
         -d '{
           "model": "MCJPG-Zero-v1"
         }'
    ```

*   **预期响应示例:** (URL 和模型名称取决于你的配置)

    ```json
    {
      "model": "gpt-4o-realtime-preview",
      "websocket_url": "wss://proxy.mcjpg.org:29678/v1/realtime/gpt-4o-realtime-preview",
      "api_key": "<你的API密钥>",
      "protocol_hint": "websocket"
    }
    ```

*   **第二步：客户端使用返回的信息建立 WebSocket 连接**

    客户端应用程序（例如 Python 脚本、Web 前端）需要：
    1.  解析上述 JSON 响应。
    2.  使用 `websocket_url` 作为目标地址。
    3.  在 WebSocket 连接握手时，通常需要在 HTTP Header 中（或根据上游实时 API 的具体要求）传递 `api_key` 进行身份验证 (例如，可能放在 `Authorization: Bearer <api_key>` 或自定义 Header 中，具体取决于上游代理和模型的实现)。
    4.  一旦连接建立，客户端就可以根据上游实时 API 的协议（例如发送音频流、接收文本转录）进行双向通信。

    **注意:** `curl` 本身不适合直接进行 WebSocket 交互。你需要使用支持 WebSocket 的编程语言库（如 Python 的 `websockets` 库）来实现客户端逻辑。

## API 端点摘要

| 端点                             | 方法    | 请求模型                     | 描述                                                            | 上游模型（默认）                                         |
| :------------------------------- | :----- | :-------------------------- | :---------------------------------------------------------------| :--------------------------------------------------------|
| `/v1/models`                     | GET    | -                           | 列出可用的模型 ID (`MCJPG-Zero-v1`)                              | N/A                                                      |
| `/v1/chat/completions`           | POST   | `ChatCompletionRequest`     | 处理聊天请求，根据内容进行路由                                    | `gemini-2.5-flash`, `gpt-5-mini`, `claude...`, 等|
| `/v1/embeddings`                 | POST   | `EmbeddingRequest`          | 创建文本嵌入向量                                                 | `text-embedding-3-large`                                  |
| `/v1/audio/speech`               | POST   | `TTSRequest`                | 从文本生成语音 (TTS)                                             | `tts-1`                                                   |
| `/v1/audio/transcriptions`       | POST   | `STTRequest`                  | 将音频转录为文本 (STT)                                            | `whisper-1`                                              |
| `/v1/images/generations`         | POST   | `ImageGenerationRequest`    | 生成图像                                                         | `dall-e-3`                                               |
| `/v1/rerank`                     | POST   | `RerankRequest`             | 对文档列表进行重排序                                              | `jina-reranker-m0`                     |
| `/v1/realtime/connection_info`   | POST   | `RealtimeConnectionRequest` | 获取用于建立 WebSocket 连接到实时 API 的信息（URL, Key）           | `gpt-4o-realtime-preview` (间接)                          |

## 贡献指南

欢迎贡献！如果你想做出贡献，请遵循以下步骤：
*   在 GitHub 上 **Fork** 本仓库。
*   在本地 **克隆** 你的 Fork (`git clone git@github.com:你的用户名/mcjpg-ai-router.git`)。
*   为你的更改 **创建新分支** (`git checkout -b feature/你的特性名称`)。
*   **进行更改**，确保代码质量并在适用的情况下添加测试。
*   **提交你的更改** (`git commit -am '添加某个特性'`)。
*   **将分支推送到** 你的 Fork (`git push origin feature/你的特性名称`)。
*   在 GitHub 上 **创建新的 Pull Request**，将你的分支与主仓库的 `main` 分支进行比较。

对于重大的更改或报告错误，请先创建一个 Issue 进行讨论。

## 许可证

本项目采用 [MIT](./LICENSE) 许可证授权

## 致谢

*   由 [MCJPG](https://mcjpg.org/) 组织开发和维护。
*   基于优秀的 [FastAPI](https://fastapi.tiangolo.com/) 框架构建。
*   依赖 [OpenAI Python SDK](https://github.com/openai/openai-python)。
