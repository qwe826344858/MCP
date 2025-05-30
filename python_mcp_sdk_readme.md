以下是 [README.md](file://D:\Python_code\python-sdk\README.md) 文件内容的中文翻译：

---

# MCP Python SDK

<div align="center">

<strong>Model Context Protocol（MCP）的 Python 实现</strong>

[![PyPI][pypi-badge]][pypi-url]
[![MIT 许可证][mit-badge]][mit-url]
[![Python 版本][python-badge]][python-url]
[![文档][docs-badge]][docs-url]
[![规范][spec-badge]][spec-url]
[![GitHub 讨论][discussions-badge]][discussions-url]

</div>

<!-- omit in toc -->
## 目录

- [MCP Python SDK](#mcp-python-sdk)
  - [概述](#overview)
  - [安装](#installation)
    - [将 MCP 添加到你的 Python 项目中](#adding-mcp-to-your-python-project)
    - [运行独立的 MCP 开发工具](#running-the-standalone-mcp-development-tools)
  - [快速入门](#quickstart)
  - [什么是 MCP？](#what-is-mcp)
  - [核心概念](#core-concepts)
    - [服务器](#server)
    - [资源](#resources)
    - [工具](#tools)
    - [提示词](#prompts)
    - [图像](#images)
    - [上下文](#context)
  - [运行你的服务器](#running-your-server)
    - [开发模式](#development-mode)
    - [Claude 桌面集成](#claude-desktop-integration)
    - [直接执行](#direct-execution)
    - [挂载到现有的 ASGI 服务器](#mounting-to-an-existing-asgi-server)
  - [示例](#examples)
    - [回声服务器](#echo-server)
    - [SQLite 浏览器](#sqlite-explorer)
  - [高级用法](#advanced-usage)
    - [低级服务器](#low-level-server)
    - [编写 MCP 客户端](#writing-mcp-clients)
    - [MCP 原语](#mcp-primitives)
    - [服务器能力](#server-capabilities)
  - [文档](#documentation)
  - [贡献](#contributing)
  - [许可证](#license)

[pypi-badge]: https://img.shields.io/pypi/v/mcp.svg
[pypi-url]: https://pypi.org/project/mcp/
[mit-badge]: https://img.shields.io/pypi/l/mcp.svg
[mit-url]: https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/mcp.svg
[python-url]: https://www.python.org/downloads/
[docs-badge]: https://img.shields.io/badge/docs-modelcontextprotocol.io-blue.svg
[docs-url]: https://modelcontextprotocol.io
[spec-badge]: https://img.shields.io/badge/spec-spec.modelcontextprotocol.io-blue.svg
[spec-url]: https://spec.modelcontextprotocol.io
[discussions-badge]: https://img.shields.io/github/discussions/modelcontextprotocol/python-sdk
[discussions-url]: https://github.com/modelcontextprotocol/python-sdk/discussions

## 概述

Model Context Protocol 允许应用程序以标准化的方式为大语言模型（LLMs）提供上下文，将提供上下文与实际 LLM 交互的关注点分离。这个 Python SDK 实现了完整的 MCP 规范，使你能够轻松地：

- 构建可以连接任何 MCP 服务器的 MCP 客户端
- 创建暴露资源、提示词和工具的 MCP 服务器
- 使用标准传输协议如 stdio、SSE 和可流式 HTTP
- 处理所有 MCP 协议消息和生命周期事件

## 安装

### 将 MCP 添加到你的 Python 项目中

我们推荐使用 [uv](https://docs.astral.sh/uv/) 来管理你的 Python 项目。

如果你还没有创建一个 uv 管理的项目，请先创建一个：

```bash
uv init mcp-server-demo
cd mcp-server-demo
```


然后将 MCP 添加到你的项目依赖中：

```bash
uv add "mcp[cli]"
```


或者，对于使用 pip 的项目：
```bash
pip install "mcp[cli]"
```


### 运行独立的 MCP 开发工具

要使用 uv 运行 mcp 命令：

```bash
uv run mcp
```


## 快速入门

让我们创建一个简单的 MCP 服务器，它暴露一个计算器工具和一些数据：

```python
# server.py
from mcp.server.fastmcp import FastMCP

# 创建一个 MCP 服务器
mcp = FastMCP("Demo")


# 添加一个加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# 添加一个动态问候资源
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"
```


你可以通过以下命令在 [Claude Desktop](https://claude.ai/download) 中安装此服务器并立即与其互动：
```bash
mcp install server.py
```


或者，你可以使用 MCP Inspector 测试它：
```bash
mcp dev server.py
```


## 什么是 MCP？

[Model Context Protocol (MCP)](https://modelcontextprotocol.io) 允许你构建服务器，以安全且标准化的方式向 LLM 应用程序公开数据和功能。可以将其视为一种 Web API，但专门为 LLM 交互设计。MCP 服务器可以：

- 通过 **资源** 暴露数据（这些类似于 GET 终结点；它们用于将信息加载到 LLM 的上下文中）
- 通过 **工具** 提供功能（类似于 POST 终结点；它们用于执行代码或其他产生副作用的操作）
- 通过 **提示词** 定义交互模式（LLM 交互的可重用模板）
- 更多！

## 核心概念

### 服务器

FastMCP 服务器是你与 MCP 协议交互的核心接口。它处理连接管理、协议合规性和消息路由：

```python
# 添加对启动/关闭支持以及强类型
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fake_database import Database  # 替换为你实际的 DB 类型

from mcp.server.fastmcp import FastMCP

# 创建一个命名服务器
mcp = FastMCP("My App")

# 为部署和开发指定依赖项
mcp = FastMCP("My App", dependencies=["pandas", "numpy"])


@dataclass
class AppContext:
    db: Database


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """使用类型安全的上下文管理应用生命周期"""
    # 在启动时初始化
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        # 在关闭时清理
        await db.disconnect()


# 将生命周期传递给服务器
mcp = FastMCP("My App", lifespan=app_lifespan)


# 在工具中访问类型安全的生命周期上下文
@mcp.tool()
def query_db() -> str:
    """使用初始化资源的工具"""
    ctx = mcp.get_context()
    db = ctx.request_context.lifespan_context["db"]
    return db.query()
```


### 资源

资源是向 LLM 暴露数据的方式。它们类似于 REST API 中的 GET 终结点——它们提供数据但不应进行重大计算或有副作用：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")


@mcp.resource("config://app")
def get_config() -> str:
    """静态配置数据"""
    return "App configuration here"


@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """动态用户数据"""
    return f"Profile data for user {user_id}"
```


### 工具

工具让 LLM 可以通过你的服务器采取行动。与资源不同，工具预期会进行计算并产生副作用：

```python
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")


@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """根据以千克为单位的体重和以米为单位的身高计算 BMI"""
    return weight_kg / (height_m**2)


@mcp.tool()
async def fetch_weather(city: str) -> str:
    """获取城市的当前天气"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
```


### 提示词

提示词是可重用的模板，帮助 LLM 有效地与你的服务器互动：

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("My App")


@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
```


### 图像

FastMCP 提供了一个 `Image` 类，自动处理图像数据：

```python
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

mcp = FastMCP("My App")


@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """从图像创建缩略图"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")
```


### 上下文

上下文对象为你的工具和资源提供对 MCP 功能的访问：

```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("My App")


@mcp.tool()
async def long_task(files: list[str], ctx: Context) -> str:
    """处理多个文件并进行进度跟踪"""
    for i, file in enumerate(files):
        ctx.info(f"Processing {file}")
        await ctx.report_progress(i, len(files))
        data, mime_type = await ctx.read_resource(f"file://{file}")
    return "Processing complete"
```


### 认证

认证可以用于希望暴露访问受保护资源工具的服务器。

`mcp.server.auth` 实现了一个 OAuth 2.0 服务器接口，服务器可以通过提供 `OAuthAuthorizationServerProvider` 协议的实现来使用它。

```python
from mcp import FastMCP
from mcp.server.auth.provider import OAuthAuthorizationServerProvider
from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
    RevocationOptions,
)


class MyOAuthServerProvider(OAuthAuthorizationServerProvider):
    # 示例实现在 `examples/servers/simple-auth`
    ...


mcp = FastMCP(
    "My App",
    auth_server_provider=MyOAuthServerProvider(),
    auth=AuthSettings(
        issuer_url="https://myapp.com",
        revocation_options=RevocationOptions(
            enabled=True,
        ),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["myscope", "myotherscope"],
            default_scopes=["myscope"],
        ),
        required_scopes=["myscope"],
    ),
)
```


详情请参阅 [OAuthAuthorizationServerProvider](src/mcp/server/auth/provider.py)。

## 运行你的服务器

### 开发模式

测试和调试服务器的最快方式是使用 MCP Inspector：

```bash
mcp dev server.py

# 添加依赖项
mcp dev server.py --with pandas --with numpy

# 挂载本地代码
mcp dev server.py --with-editable .
```


### Claude 桌面集成

一旦你的服务器准备就绪，可以在 Claude Desktop 中安装它：

```bash
mcp install server.py

# 自定义名称
mcp install server.py --name "My Analytics Server"

# 环境变量
mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
mcp install server.py -f .env
```


### 直接执行

对于自定义部署等高级场景：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

if __name__ == "__main__":
    mcp.run()
```


运行它：
```bash
python server.py
# 或者
mcp run server.py
```


注意：`mcp run` 或 `mcp dev` 仅支持使用 FastMCP 的服务器，不支持低级服务器变体。

### 可流式 HTTP 传输

> **注意**：对于生产部署，可流式 HTTP 传输正在取代 SSE 传输。

```python
from mcp.server.fastmcp import FastMCP

# 有状态服务器（维护会话状态）
mcp = FastMCP("StatefulServer")

# 无状态服务器（没有会话持久性）
mcp = FastMCP("StatelessServer", stateless_http=True)

# 无状态服务器（没有会话持久性，也没有带支持客户端的 SSE 流）
mcp = FastMCP("StatelessServer", stateless_http=True, json_response=True)

# 使用 streamable_http 传输运行服务器
mcp.run(transport="streamable-http")
```


你可以将多个 FastMCP 服务器挂载到一个 FastAPI 应用中：

```python
# echo.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="EchoServer", stateless_http=True)


@mcp.tool(description="一个简单的回声工具")
def echo(message: str) -> str:
    return f"Echo: {message}"
```
```python
# math.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="MathServer", stateless_http=True)


@mcp.tool(description="一个简单的加法工具")
def add_two(n: int) -> int:
    return n + 2
```
```python
# main.py
import contextlib
from fastapi import FastAPI
from mcp.echo import echo
from mcp.math import math


# 创建一个组合的生命周期以管理两个会话管理器
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(echo.mcp.session_manager.run())
        await stack.enter_async_context(math.mcp.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/echo", echo.mcp.streamable_http_app())
app.mount("/math", math.mcp.streamable_http_app())
```


对于低级服务器的可流式 HTTP 实现，请参见：
- 有状态服务器: [`examples/servers/simple-streamablehttp/`](examples/servers/simple-streamablehttp/)
- 无状态服务器: [`examples/servers/simple-streamablehttp-stateless/`](examples/servers/simple-streamablehttp-stateless/)

可流式 HTTP 传输支持：
- 有状态和无状态操作模式
- 使用事件存储的恢复性
- JSON 或 SSE 响应格式
- 更好的多节点部署扩展性

### 挂载到现有的 ASGI 服务器

> **注意**：SSE 传输正被 [Streamable HTTP 传输](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) 取代。

默认情况下，SSE 服务器挂载在 `/sse`，而 Streamable HTTP 服务器挂载在 `/mcp`。你可以通过以下方法自定义这些路径。

你可以使用 [sse_app](file://D:\Python_code\python-sdk\src\mcp\server\fastmcp\server.py#L656-L767) 方法将 SSE 服务器挂载到现有的 ASGI 服务器上。这允许你将 SSE 服务器与其他 ASGI 应用集成。

```python
from starlette.applications import Starlette
from starlette.routing import Mount, Host
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("My App")

# 将 SSE 服务器挂载到现有的 ASGI 服务器
app = Starlette(
    routes=[
        Mount('/', app=mcp.sse_app()),
    ]
)

# 或者动态挂载为主机
app.router.routes.append(Host('mcp.acme.corp', app=mcp.sse_app()))
```


当在不同的路径下挂载多个 MCP 服务器时，你可以通过几种方式配置挂载路径：

```python
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.fastmcp import FastMCP

# 创建多个 MCP 服务器
github_mcp = FastMCP("GitHub API")
browser_mcp = FastMCP("Browser")
curl_mcp = FastMCP("Curl")
search_mcp = FastMCP("Search")

# 方法 1: 通过设置配置挂载路径（推荐用于持久化配置）
github_mcp.settings.mount_path = "/github"
browser_mcp.settings.mount_path = "/browser"

# 方法 2: 直接传入挂载路径到 sse_app（适用于临时挂载）
# 这种方法不会永久修改服务器的设置

# 创建 Starlette 应用并挂载多个服务器
app = Starlette(
    routes=[
        # 使用基于设置的配置
        Mount("/github", app=github_mcp.sse_app()),
        Mount("/browser", app=browser_mcp.sse_app()),
        # 使用直接传入挂载路径的方法
        Mount("/curl", app=curl_mcp.sse_app("/curl")),
        Mount("/search", app=search_mcp.sse_app("/search")),
    ]
)

# 方法 3: 对于直接执行，你也可以在 run() 中传递挂载路径
if __name__ == "__main__":
    search_mcp.run(transport="sse", mount_path="/search")
```


有关在 Starlette 中挂载应用的更多信息，请参见 [Starlette 文档](https://www.starlette.io/routing/#submounting-routes)。

## 示例

### 回声服务器

一个展示资源、工具和提示词的简单服务器：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo")


@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """将消息作为资源回声"""
    return f"Resource echo: {message}"


@mcp.tool()
def echo_tool(message: str) -> str:
    """将消息作为工具回声"""
    return f"Tool echo: {message}"


@mcp.prompt()
def echo_prompt(message: str) -> str:
    """创建一个回声提示词"""
    return f"Please process this message: {message}"
```


### SQLite 浏览器

一个更复杂的数据库集成示例：

```python
import sqlite3

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SQLite Explorer")


@mcp.resource("schema://main")
def get_schema() -> str:
    """将数据库模式作为资源提供"""
    conn = sqlite3.connect("database.db")
    schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(sql[0] for sql in schema if sql[0])


@mcp.tool()
def query_data(sql: str) -> str:
    """安全地执行 SQL 查询"""
    conn = sqlite3.connect("database.db")
    try:
        result = conn.execute(sql).fetchall()
        return "\n".join(str(row) for row in result)
    except Exception as e:
        return f"Error: {str(e)}"
```


## 高级用法

### 低级服务器

为了获得更多的控制权，你可以直接使用低级服务器实现。这为你提供了对协议的完全访问权限，并允许你自定义服务器的每一个方面，包括通过生命周期 API 管理生命周期：

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fake_database import Database  # 替换为你实际的 DB 类型

from mcp.server import Server


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """管理服务器启动和关闭的生命周期"""
    # 在启动时初始化资源
    db = await Database.connect()
    try:
        yield {"db": db}
    finally:
        # 在关闭时清理
        await db.disconnect()


# 将生命周期传递给服务器
server = Server("example-server", lifespan=server_lifespan)


# 在处理器中访问生命周期上下文
@server.call_tool()
async def query_db(name: str, arguments: dict) -> list:
    ctx = server.request_context
    db = ctx.lifespan_context["db"]
    return await db.query(arguments["query"])
```


生命周期 API 提供：
- 初始化服务器启动时的资源并在服务器停止时清理它们的方法
- 在处理器中通过请求上下文访问初始化资源
- 生命周期和请求处理器之间的类型安全上下文传递

```python
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# 创建一个服务器实例
server = Server("example-server")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="example-prompt",
            description="An example prompt template",
            arguments=[
                types.PromptArgument(
                    name="arg1", description="Example argument", required=True
                )
            ],
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name != "example-prompt":
        raise ValueError(f"Unknown prompt: {name}")

    return types.GetPromptResult(
        description="Example prompt",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text="Example prompt text"),
            )
        ],
    )


async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="example",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```


警告：`mcp run` 和 `mcp dev` 不支持低级服务器。

### 编写 MCP 客户端

SDK 提供了一个高级客户端接口，用于使用各种 [传输协议](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) 连接到 MCP 服务器：

```python
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# 创建 stdio 连接的服务器参数
server_params = StdioServerParameters(
    command="python",  # 可执行文件
    args=["example_server.py"],  # 可选的命令行参数
    env=None,  # 可选的环境变量
)


# 可选：创建一个采样回调
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # 初始化连接
            await session.initialize()

            # 列出可用的提示词
            prompts = await session.list_prompts()

            # 获取一个提示词
            prompt = await session.get_prompt(
                "example-prompt", arguments={"arg1": "value"}
            )

            # 列出可用的资源
            resources = await session.list_resources()

            # 列出可用的工具
            tools = await session.list_tools()

            # 读取一个资源
            content, mime_type = await session.read_resource("file://some/path")

            # 调用一个工具
            result = await session.call_tool("tool-name", arguments={"arg1": "value"})


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```


客户端还可以使用 [可流式 HTTP 传输协议](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) 进行连接：

```python
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


async def main():
    # 连接到一个可流式 HTTP 服务器
    async with streamablehttp_client("example/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # 使用客户端流创建一个会话
        async with ClientSession(read_stream, write_stream) as session:
            # 初始化连接
            await session.initialize()
            # 调用一个工具
            tool_result = await session.call_tool("echo", {"message": "hello"})
```


### 客户端的 OAuth 认证

SDK 包括用于连接受保护的 MCP 服务器的 [授权支持](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization)：

```python
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken


class CustomTokenStorage(TokenStorage):
    """简单的内存令牌存储实现"""
    
    async def get_tokens(self) -> OAuthToken | None:
        pass

    async def set_tokens(self, tokens: OAuthToken) -> None:
        pass

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        pass

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        pass


async def main():
    # 设置 OAuth 认证
    oauth_auth = OAuthClientProvider(
        server_url="https://api.example.com",
        client_metadata=OAuthClientMetadata(
            client_name="My Client",
            redirect_uris=["http://localhost:3000/callback"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=CustomTokenStorage(),
        redirect_handler=lambda url: print(f"Visit: {url}"),
        callback_handler=lambda: ("auth_code", None),
    )

    # 与可流式 HTTP 客户端一起使用
    async with streamablehttp_client(
        "https://api.example.com/mcp", auth=oauth_auth
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 准备好经过认证的会话
```


有关完整的工作示例，请参见 [`examples/clients/simple-auth-client/`](examples/clients/simple-auth-client/)。

### MCP 原语

MCP 协议定义了服务器可以实现的三个核心原语：

| 原语 | 控制               | 描述                                         | 示例用途                  |
|------|--------------------|----------------------------------------------|---------------------------|
| 提示词 | 用户控制       | 用户选择调用的交互模板                     | 斜杠命令、菜单选项       |
| 资源 | 应用控制     | 由客户端应用程序管理的上下文数据         | 文件内容、API 响应      |
| 工具 | 模型控制     | 暴露给 LLM 以采取行动的功能              | API 调用、数据更新       |

### 服务器能力

MCP 服务器在初始化期间声明其能力：

| 能力  | 特性标志                 | 描述                        |
|-------|--------------------------|-----------------------------|
| `prompts`   | `listChanged`                | 提示词模板管理         |
| `resources` | `subscribe`<br/>`listChanged`| 资源暴露和更新         |
| `tools`     | `listChanged`                | 工具发现和执行         |
| `logging`   | -                            | 服务器日志配置         |
| `completion`| -                            | 参数完成建议           |

## 文档

- [Model Context Protocol 文档](https://modelcontextprotocol.io)
- [Model Context Protocol 规范](https://spec.modelcontextprotocol.io)
- [官方支持的服务器列表](https://github.com/modelcontextprotocol/servers)

## 贡献

我们热衷于支持各个经验水平的贡献者，并非常希望看到你参与该项目。请参阅 [贡献指南](CONTRIBUTING.md) 以开始。

## 许可证

该项目采用 MIT 许可证 —— 请参阅 LICENSE 文件了解详细信息。