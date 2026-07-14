# 电商客服 Agent

基于 LangGraph 构建的智能电商客服系统，支持 **RAG 知识库问答**、**退货流程自动化** 和 **人工客服转接**。

## 架构概览

```
用户 ─→ FastAPI ─→ LangGraph Agent ─→ LLM (OpenAI 兼容)
                      │    │    │
                      │    │    └── HTTP ─→ 外部 RAG 知识库
                      │    │
                      │    ├── general_qa ──→ RAG 检索 → 生成回答
                      │    ├── return_request → 退货子图 (6步)
                      │    └── human_support ─→ 转人工队列
                      │
                      └── SQLite (会话 + 图状态持久化)
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.11+
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入必要配置：

```env
# LLM（必填 — 支持任何 OpenAI 兼容 API）
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4o

# 外部 RAG 服务（必填）
RAG_BASE_URL=http://localhost:8080
RAG_API_KEY=your-rag-api-key

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

服务启动后访问 `http://localhost:8000`。

## 项目结构

```
MyAgent/
├── main.py                     # FastAPI 入口
├── config.py                   # 环境变量配置（pydantic-settings）
├── requirements.txt
├── .env.example
├── agent/
│   ├── state.py                # AgentState 状态定义
│   ├── graph.py                # LangGraph 主图（意图路由 + 条件分支）
│   └── nodes/
│       ├── classify.py         # 意图分类（LLM + 关键词快路径）
│       ├── rag.py              # HTTP 调用外部 RAG 服务
│       ├── generate.py         # 结合 RAG 上下文生成最终回答
│       ├── return_flow.py      # 退货流程 6 个节点
│       └── handoff.py          # 转人工（排队通知）
├── api/
│   ├── schemas.py              # Pydantic 请求/响应模型
│   └── routes.py               # REST API 路由
├── services/
│   ├── llm.py                  # LLM 调用封装（OpenAI 兼容）
│   ├── rag_client.py           # 外部 RAG HTTP 客户端（httpx）
│   └── session_store.py        # SQLite 会话持久化
└── tools/
    ├── order.py                # 订单查询（含 mock 数据，生产对接真实 API）
    └── return_policy.py        # 退货政策检查 + 创建退货单
```

## API 端点

### 对话

```http
POST /api/chat
Content-Type: application/json

{
  "user_id": "user_001",
  "message": "这款手机支持5G吗？",
  "session_id": "可选的已有会话ID"
}
```

响应：

```json
{
  "session_id": "a1b2c3d4e5f6g7h8",
  "response": "您好，XPhone X1 支持 5G 双模...",
  "intent": "general_qa",
  "need_handoff": false,
  "handoff_reason": null
}
```

### 查看会话历史

```http
GET /api/history/{session_id}
```

### 查看转人工队列

```http
GET /api/handoff/queue
```

### 人工坐席接起

```http
POST /api/handoff/{session_id}/pickup
Content-Type: application/json

{
  "agent_name": "客服张三"
}
```

### 关闭会话

```http
DELETE /api/session/{session_id}
```

### 健康检查

```http
GET /health
```

## 三大核心功能

### 1. RAG 知识库问答

```
用户提问 → classify_intent (识别为 general_qa)
         → rag_retrieve (POST 请求外部 RAG 服务)
         → generate_response (LLM 结合检索结果生成回答)
```

外部 RAG 接口约定：

```
POST {RAG_BASE_URL}/api/retrieve
Body:   {"query": "用户问题", "top_k": 5}
Return: {"documents": [{"content": "...", "score": 0.95}, ...]}
```

RAG 服务不可用时，系统自动降级为基础 LLM 回答或转人工。

### 2. 退货流程

6 步状态机，支持多轮对话，中断可恢复（SQLite 持久化）：

```
return_start          询问订单号（自动从消息中提取）
    │
return_validate_order 验证订单是否存在且属于当前用户
    │                 └── 失败重试，超限自动转人工
return_check_policy   检查退货资格（7天无理由、类目限制等）
    │                 └── 不可退则告知原因并结束
return_collect_reason 收集退货原因
    │
return_initiate       创建退货单
    │                 └── 失败则转人工
return_confirm        返回退货单号、取件时间、退款流程
```

### 3. 转人工策略

以下 6 种情况自动触发转人工：

| 触发条件 | 说明 |
|---|---|
| 关键词匹配 | "转人工""人工客服""找真人""我要投诉" 等 → 跳过 LLM 直接转 |
| 意图分类 | LLM 判定用户有强烈不满、投诉情绪 |
| RAG 不可用 | 外部知识库服务健康检查失败 |
| 订单验证失败 | 连续 N 次（由 `MAX_RETURN_ATTEMPTS` 控制） |
| 退货流程异常 | 创建退货单失败等后端错误 |
| LLM 异常 | LLM 调用超时或返回异常 |

转人工后：会话存入 SQLite → 进入排队队列 → 返回预估等待时间 → 人工坐席通过 `/api/handoff/{id}/pickup` 接起。

## 数据持久化

使用 **SQLite**（WAL 模式），服务重启数据不丢失：

```
data/agent.db
├── sessions 表       对话历史、退货状态、转人工标记
└── checkpoint_* 表   LangGraph 图执行状态（断点续传）
```

- 会话在 `SESSION_TTL_MINUTES` 分钟后自动过期（惰性淘汰）
- LangGraph checkpoint 保证退货流程任意步骤中断后可从断点恢复

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
| Agent 框架 | LangGraph (StateGraph + SqliteSaver) |
| Web 框架 | FastAPI + uvicorn |
| LLM 调用 | langchain-openai（兼容任何 OpenAI API） |
| HTTP 客户端 | httpx（异步） |
| 数据校验 | Pydantic v2 |
| 配置管理 | pydantic-settings |
| 持久化 | SQLite (WAL mode) |
