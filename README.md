

# MCJPG AI 模型

[![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]({占位符}[指向许可证文件的链接]) 
![Python 版本](https://img.shields.io/badge/python-3.8+-blue.svg)
![框架](https://img.shields.io/badge/Framework-FastAPI-green.svg)

一个 FastAPI 应用，作为处理多种 AI 任务（聊天、嵌入、文本转语音、语音转文本）的智能路由。它提供了一个统一的模型接口 (`MCJPG-Zero-v1`)，同时根据请求类型或内容分析，将请求定向到最优的上游 OpenAI 兼容模型。由 MCJPG 组织开发。

## 概述

本项目旨在解决与不同任务的多个专用 AI 模型交互的复杂性。用户无需了解应调用哪个具体模型（例如，`gpt-4o-mini` 用于工具调用，`tts-1` 用于语音合成，`gemini-2.0-flash` 用于通用聊天），此路由提供了一个单一、一致的模型 ID：`MCJPG-Zero-v1`。

该路由智能地将请求转发到部署在代理（如 `proxy.mcjpg.org`）后的适当上游后端模型，简化了 API 的使用，并允许灵活管理后端模型，而无需更改面向客户端的接口。

## 主要特性

* **统一模型 ID:** 使用单一的 `MCJPG-Zero-v1` 模型 ID 与所有支持的模态进行交互。
* **OpenAI API 兼容:** 遵循 OpenAI API 请求/响应格式，可与现有工具和库无缝集成。
* **智能聊天路由:** 分析聊天消息内容（仅文本），为编码、写作、网络搜索、通用聊天等任务选择最合适的上游模型。
* **直接请求转发:** 将特定类型的请求直接路由到指定的上游模型：
    * 包含 `tools` 或 `tool_choice` 的聊天请求将发送到支持工具调用的模型（例如 `gpt-4o-mini`）。
    * 嵌入请求 (`/v1/embeddings`) 将发送到文本嵌入模型（例如 `text-embedding-3-large`）。
    * 文本转语音 (TTS) 请求 (`/v1/audio/speech`) 将发送到 TTS 模型（例如 `tts-1`）。
    * 语音转文本 (STT) 请求 (`/v1/audio/transcriptions`) 将发送到 STT 模型（例如 `whisper-1`）。
* **流式支持:** 处理聊天完成（Chat Completions）和文本转语音（Text-to-Speech）的流式响应。
* **系统提示词注入:** 自动在聊天请求前添加可配置的系统提示词（例如，定义 AI 源自 MCJPG），同时尊重用户提供的系统提示词。
* **可配置:** 用于路由/转发的上游代理 URL 和特定模型名称可以通过环境变量轻松配置。

## 工作原理

1. **接收请求:** 用户向路由发送 API 请求（聊天、嵌入、TTS、STT），指定 `model: "MCJPG-Zero-v1"`。请求必须包含有效的 `Authorization: Bearer <用户API密钥>` 头。
2. **端点识别:** 路由识别目标端点（例如 `/v1/chat/completions`, `/v1/embeddings`）。
3. **路由/转发逻辑:**
    * **嵌入、TTS、STT:** 请求直接映射到预先配置的上游模型 (`UPSTREAM_EMBEDDING_MODEL`, `UPSTREAM_TTS_MODEL`, `UPSTREAM_STT_MODEL`)。
    * **聊天完成:**
        * 如果存在 `tools` 或 `tool_choice`，请求将转发到 `DIRECT_TOOL_CALL_MODEL`。
        * 否则，路由提取文本内容，调用 `ROUTING_MODEL`（例如 `gemini-2.0-flash`）并使用特定的函数调用指令 (`select_upstream_model`) 来确定任务类型（编码、写作等）。
        * 路由根据确定的任务类型（以及输入是否包含图像）和预定义规则，映射到合适的上游聊天模型。
        * 如果路由失败，则默认为通用的聊天模型。
4. **请求准备:** 路由为选定的上游模型构建最终的请求负载。对于聊天，它会注入 `MCJPG_SYSTEM_PROMPT` 并使用原始消息。对于其他类型，它使用映射后的上游模型 ID。
5. **上游调用:** 路由通过配置的 `PROXY_BASE_URL` 将准备好的请求发送到上游模型，并传递用户的 API 密钥。
6. **响应处理:** 路由从上游模型接收响应，并将其流式传输或直接返回给原始用户。

## 技术栈

* **Python:** 核心编程语言。
* **FastAPI:** 用于构建 API 的高性能 Web 框架。
* **Pydantic:** 数据验证和设置管理。
* **OpenAI Python SDK:** 用于与上游 OpenAI 兼容 API 交互（包括类型定义）。
* **HTTPX:** OpenAI SDK 使用的底层异步 HTTP 客户端（如果需要，也可用于直接流式传输）。
* **python-dotenv:** 用于从 `.env` 文件加载环境变量。
* **uvicorn:** 用于运行 FastAPI 应用程序的 ASGI 服务器。
* **python-multipart:** 处理文件上传（STT）所需。

## 安装与设置

1. **先决条件:**
    
    * Python 3.8 或更高版本
    * pip (Python 包安装器)
    * Git (用于克隆仓库)
2. **克隆仓库:**
    
    ```bash
    git clone [你的 GitHub 仓库链接]
    cd mcjpg-ai-router # 或者你的仓库目录名
    ```
3. **安装依赖:**
    
    ```bash
    pip install "fastapi[all]" uvicorn python-dotenv openai httpx python-multipart
    # 或者，如果你创建了 requirements.txt:
    # pip install -r requirements.txt
    ```
    
    *(建议创建一个 `requirements.txt` 文件以便于依赖管理)*
4. **配置环境变量:**
    在项目的根目录下创建一个名为 `.env` 的文件。添加以下变量：
    
    ```dotenv
    # 必需：上游 OpenAI 兼容代理的基础 URL
    # 示例：PROXY_BASE_URL=https://api.openai.com/ (如果直接使用 OpenAI)
    # 示例：PROXY_BASE_URL=https://proxy.mcjpg.org:29678/
    PROXY_BASE_URL=https://proxy.mcjpg.org:29678/
    
    # 可选：如果需要，覆盖默认的模型名称
    # ROUTING_MODEL=gemini-2.0-flash
    # DIRECT_TOOL_CALL_MODEL=gpt-4o-mini
    # UPSTREAM_EMBEDDING_MODEL=text-embedding-3-large
    # UPSTREAM_TTS_MODEL=tts-1
    # UPSTREAM_STT_MODEL=whisper-1
    ```
    
    * `PROXY_BASE_URL` **至关重要**。它必须指向你的上游 API 提供商的基础 URL（`/v1/...` 之前的部分）。确保它以 `/` 结尾。
    * 其他模型名称默认为代码中显示的值。仅在需要更改它们时才将它们添加到 `.env` 文件中。
5. **运行服务器:**
    
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8005 --reload
    ```
    
    * `--host 0.0.0.0` 使服务器可在你的本地网络上访问。仅本地访问使用 `127.0.0.1`。
    * `--port 8005` 指定端口（如果需要，可以更改）。
    * `--reload` 在开发期间启用自动重新加载（生产环境请移除）。

路由现在应该运行在 `http://<你的服务器IP>:8005`。

## 使用方法

使用标准的 OpenAI API 客户端库或 `curl` 与路由进行交互，始终指定 `model: "MCJPG-Zero-v1"` 并在 `Authorization` 头中提供你的上游 API 密钥。

**重要:** 路由本身不处理身份验证，它只是要求 `Authorization` 头，并将其直接传递给上游 `PROXY_BASE_URL`。请确保你的代理正确处理身份验证。

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
       ],
       "temperature": 0.7
     }'
```

**(路由将添加 MCJPG 系统提示词并选择一个合适的上游模型，如 **gemini-2.0-flash**)**

**2. 聊天 (流式输出):**

```
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


**(路由将选择一个上游模型，如 **claude-3-7-sonnet-20250219** 或 **DeepSeek-V3**，并流式传输响应)**

**3. 嵌入向量 (Embeddings):**

```
curl -X POST "<路由URL>/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "input": "我的世界服务器"
     }'
```

**(路由将通过代理将此请求转发到 **text-embedding-3-large**)**

**4. 文本转语音 (TTS):**

```
curl -X POST "<路由URL>/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <你的API密钥>" \
     -d '{
       "model": "MCJPG-Zero-v1",
       "input": "欢迎来到MCJPG！",
       "voice": "alloy",
       "response_format": "mp3"
     }' \
     --output speech.mp3
```


**(路由将通过代理将此请求转发到 **tts-1**，并流式传回 MP3 音频)**

**5. 语音转文本 (STT):**

```
curl -X POST "<路由URL>/v1/audio/transcriptions" \
     -H "Authorization: Bearer <你的API密钥>" \
     -F file=@"/path/to/your/audio.mp3" \
     -F model="MCJPG-Zero-v1" \
     -F response_format="json"
```

**(路由将通过代理将音频文件和参数转发到 **whisper-1**)**

## API 端点摘要

| **端点**                     | **方法** | **请求模型**              | **描述**                                  | **上游模型（默认）**                                     |
| ------------------------------ | ---------- | --------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| **/v1/models**               | **GET**  | **-**                     | **列出可用的模型 ID (**MCJPG-Zero-v1**)** | **N/A**                                                  |
| **/v1/chat/completions**     | **POST** | **ChatCompletionRequest** | **处理聊天请求，根据内容进行路由**        | **gemini-2.0-flash**,**gpt-4o-mini**,**claude-3...**, 等 |
| **/v1/embeddings**           | **POST** | **EmbeddingRequest**      | **创建文本嵌入向量**                      | **text-embedding-3-large**                               |
| **/v1/audio/speech**         | **POST** | **TTSRequest**            | **从文本生成语音 (TTS)**                  | **tts-1**                                                |
| **/v1/audio/transcriptions** | **POST** | **(表单数据)**            | **将音频转录为文本 (STT)**                | **whisper-1**                                            |

## 贡献指南

**欢迎贡献！如果你想做出贡献，请遵循以下步骤：**

* **在 GitHub 上 **Fork** 本仓库。**
* **在本地 **克隆** 你的 Fork (**git clone git@github.com:你的用户名/mcjpg-ai-router.git**)。**
* **为你的更改 **创建新分支** (**git checkout -b feature/你的特性名称**)。**
* **进行更改**，确保代码质量并在适用的情况下添加测试。
* **提交你的更改** (**git commit -am '添加某个特性'**)。
* **将分支推送到** 你的 Fork (**git push origin feature/你的特性名称**)。
* **在 GitHub 上 **创建新的 Pull Request**，将你的分支与主仓库的 **main** 分支进行比较。**

**对于重大的更改或报告错误，请先创建一个 Issue 进行讨论。**

## 致谢

* **由** [MCJPG](https://mcjpg.org/) 组织开发和维护。
* **基于优秀的** [FastAPI](https://fastapi.tiangolo.com/) 框架构建。
* **依赖** [OpenAI Python SDK](https://github.com/openai/openai-python)。
