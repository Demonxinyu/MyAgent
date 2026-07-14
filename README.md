# 电商客服 Agent

基于 LangGraph 构建的智能电商客服系统，支持 **RAG 知识库问答**、**退货流程自动化** 和 **人工客服转接**。

内置 Web 测试控制台，启动后浏览器打开 `http://localhost:8000` 即可交互式测试。

## 架构概览

```
用户 ─→ FastAPI ─→ LangGraph Agent ─→ LLM (OpenAI 兼容)
                      │    │    │
                      │    │    └── HTTP ─→ 外部 RAG 知识库
                      │    │
                      │    ├── general_qa ──→ RAG 检索 → 生成回答
                      │    ├── return_request → 退货多轮对话 (6步状态机)
                      │    └── human_support ─→ 转人工队列
                      │
                      └── SQLite (会话 + 图状态持久化)
```

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入必要配置：

```env
# LLM（必填 — 支持任何 OpenAI 兼容 API，包括 DeepSeek）
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=deepseek-chat

# 外部 RAG 服务
RAG_BASE_URL=http://localhost:8080
RAG_API_KEY=

# 其他（可选，以下为默认值）
HOST=0.0.0.0
PORT=8000
DB_PATH=./data/agent.db
SESSION_TTL_MINUTES=30
MAX_RETURN_ATTEMPTS=2
```

### 3. 启动

```bash
uvicorn main:app --reload
```

浏览器访问 `http://localhost:8000`，进入测试控制台。也可直接调用 API：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","message":"我要退货"}'
```

### 健康检查

```http
GET /health

{
  "status": "ok|degraded",
  "checks": {
    "llm": {"ok": true,  "url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "rag": {"ok": false, "url": "http://localhost:8080"}
  }
}
```

## 项目结构

```
MyAgent/
├── main.py                     # FastAPI 入口 + 全局异常处理
├── config.py                   # 环境变量配置（pydantic-settings）
├── requirements.txt
├── .env.example
├── static/
│   └── index.html              # Web 测试控制台
├── agent/
│   ├── state.py                # AgentState 状态定义
│   ├── graph.py                # LangGraph 主图（异步编译 + 条件路由）
│   └── nodes/
│       ├── classify.py         # 意图分类（structured output + 关键词快路径）
│       ├── rag.py              # HTTP 调用外部 RAG 服务
│       ├── generate.py         # 结合 RAG 上下文生成最终回答
│       ├── return_flow.py      # 退货 6 步状态机（多轮对话）
│       └── handoff.py          # 转人工（保存上下文 + 排队通知）
├── api/
│   ├── schemas.py              # Pydantic 请求/响应模型
│   └── routes.py               # REST API 路由 + 图懒加载
├── services/
│   ├── llm.py                  # LLM 封装（structured output 自动降级）
│   ├── rag_client.py           # 外部 RAG HTTP 客户端（httpx）
│   └── session_store.py        # SQLite 会话持久化
└── tools/
    ├── order.py                # 订单查询（mock 数据，生产替换）
    └── return_policy.py        # 退货政策检查 + 创建退货单（mock）
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 重定向到测试控制台 |
| `GET` | `/health` | 健康检查（含 LLM/RAG 连通性） |
| `POST` | `/api/chat` | 对话接口（支持多轮 session） |
| `GET` | `/api/history/{session_id}` | 查看会话历史 |
| `GET` | `/api/handoff/queue` | 待处理转人工队列 |
| `POST` | `/api/handoff/{session_id}/pickup` | 人工坐席接起 |
| `DELETE` | `/api/session/{session_id}` | 关闭会话 |

### 对话接口

```http
POST /api/chat
Content-Type: application/json

{
  "user_id": "user_001",
  "message": "这款手机支持5G吗？",
  "session_id": "可选的已有会话ID"
}
```

```json
{
  "session_id": "a1b2c3d4e5f6g7h8",
  "response": "您好，XPhone X1 支持 5G 双模...",
  "intent": "general_qa",
  "need_handoff": false,
  "handoff_reason": null
}
```

## 三大核心功能

### 1. RAG 知识库问答

```
用户提问 → classify_intent (structured output 或 prompt fallback)
         → rag_retrieve (POST {RAG_BASE_URL}/api/retrieve)
         → generate_response (LLM 结合检索结果生成回答)
```

外部 RAG 接口约定：

```
POST {RAG_BASE_URL}/api/retrieve
Body:   {"query": "用户问题", "top_k": 5}
Return: {"documents": [{"content": "...", "score": 0.95}, ...]}
```

RAG 服务不可用时，系统生成基础 LLM 回答并建议用户联系人工客服；严重异常时自动转人工。

### 2. 退货流程

6 步状态机，多轮对话驱动。每轮结束后若需等待用户输入，图路由到 `END` 并持久化当前进度；下轮从断点恢复继续：

```
第1轮 ─ return_start            提取或询问订单号
          └── 无订单号 → waiting_order_id → END (输出: "请提供订单号")

第2轮 ─ return_start            识别到订单号 → order_extracted
          → return_validate_order 验证订单归属
          → return_check_policy   检查退货资格
          → return_collect_reason 收集或询问退货原因
               └── 无原因 → collecting_reason → END (输出: "请告诉我退货原因")

第3轮 ─ return_start            collecting_reason → need_reason
          → return_collect_reason 提取到原因 → reason_collected
          → return_initiate       创建退货单
          → return_confirm        确认并展示后续步骤 → confirmed → END
```

**异常处理：**
- 订单验证失败 → 允许重试（次数由 `MAX_RETURN_ATTEMPTS` 控制），超限自动转人工
- 不符合退货政策 → `not_eligible`，告知原因
- 创建退货单失败 → `failed`，转人工

### 3. 转人工策略

6 种触发条件：

| 触发条件 | 说明 |
|---|---|
| 关键词匹配 | "转人工""人工客服""找真人""我要投诉"等 → 跳过 LLM 直接转 |
| 意图分类 | LLM 判定为 `human_support`（投诉、强烈不满情绪） |
| RAG 不可用 | 外部知识库服务健康检查失败 |
| 订单验证超限 | 连续 N 次失败（由 `MAX_RETURN_ATTEMPTS` 控制） |
| 退货流程异常 | 创建退货单失败等后端错误 |
| LLM/系统异常 | LLM 调用失败、超时，或代码异常（全局异常处理器兜底） |

转人工后：会话标记 `need_handoff=True` → 存入 SQLite → 进入排队队列 → 返回预估等待时间 → 人工坐席通过 `/api/handoff/{id}/pickup` 接起。

### 意图分类：双层策略

系统自动检测 LLM 提供商能力，选择最优分类方式：

```
首次分类请求
    │
    ├─ 尝试 Native (with_structured_output → function calling)
    │     └─ 成功 → 永久使用此模式（OpenAI/Claude 等）
    │
    └─ 失败 → 自动降级为 Fallback (PydanticOutputParser → prompt 注入 → JSON 提取)
                └─ 永久使用此模式（DeepSeek 等不支持 function calling 的模型）
```

两种模式都返回经过 Pydantic 校验的 `IntentClassification` 对象，**永远不会出现 JSON 解析错误**。

## Web 测试控制台

启动后访问 `http://localhost:8000`，内嵌的测试页面提供：

- 实时聊天界面（用户消息 + Agent 回复 + 状态芯片）
- 意图标签和退货步骤进度条
- 4 个快捷测试按钮（商品咨询 / 退货 / 带订单号退货 / 转人工）
- Session ID 管理（创建/复制/加载历史）
- Enter 发送，Shift+Enter 换行

## 数据持久化

使用 **SQLite**（WAL 模式），服务重启数据不丢失：

```
data/agent.db
├── sessions 表       对话历史、退货状态、转人工标记
└── checkpoint_* 表   LangGraph 图执行状态（断点续传，AsyncSqliteSaver）
```

- 会话在 `SESSION_TTL_MINUTES` 分钟后自动过期（惰性淘汰）
- LangGraph checkpoint 保证退货流程任意步骤中断后可从断点恢复
- 退货流程结束后 (`confirmed`/`not_eligible`/`failed`) 自动清理 session 中的 return 字段，避免后续消息被误判为退货

## 对接外部系统

当前 `tools/order.py` 和 `tools/return_policy.py` 使用 mock 数据。对接真实系统时替换以下函数：

```python
# tools/order.py
async def lookup_order(order_id: str, user_id: str) -> dict | None:
    # 替换为：调用订单服务 HTTP API 或查询数据库

# tools/return_policy.py
async def check_return_eligibility(order_id: str) -> tuple[bool, str]:
    # 替换为：调用退货政策引擎

async def create_return(order_id, user_id, reason) -> dict:
    # 替换为：调用退货服务创建退货单
```

## 技术栈

| 层 | 选型 |
|---|---|
| Agent 框架 | LangGraph (StateGraph + AsyncSqliteSaver) |
| Web 框架 | FastAPI + uvicorn |
| LLM 调用 | langchain-openai（支持 OpenAI / DeepSeek 等所有兼容 API） |
| 结构化输出 | `with_structured_output` (native) / `PydanticOutputParser` (fallback) |
| HTTP 客户端 | httpx（异步） |
| 数据校验 | Pydantic v2 |
| 配置管理 | pydantic-settings |
| 持久化 | SQLite WAL mode (aiosqlite) |
