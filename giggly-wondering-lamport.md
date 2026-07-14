# 电商客服 Agent 系统 — 实现计划

## Context

在空目录 `D:\code\MyAgent` 下从零构建一个基于 LangGraph 的电商智能客服系统。核心需求：
1. 通过 HTTP 接入外部 RAG 知识库，回答商品/政策类问题
2. 识别用户意图，引导完成退货流程
3. 遇到无法处理的问题或用户要求时，转人工客服

## 技术栈

| 层 | 选型 |
|---|---|
| Agent 框架 | LangGraph (StateGraph) |
| Web 框架 | FastAPI + uvicorn |
| LLM 调用 | OpenAI 兼容 SDK (可对接任何兼容 API) |
| HTTP 客户端 | httpx (异步) |
| 数据校验 | Pydantic v2 |
| 会话存储 | 内存 dict (生产可换 Redis) |

## 项目结构

```
MyAgent/
├── requirements.txt
├── .env.example
├── config.py                  # 配置管理（环境变量）
├── main.py                    # FastAPI 入口 + 启动
├── agent/
│   ├── __init__.py
│   ├── state.py               # AgentState 定义
│   ├── graph.py               # 主图构建（节点+路由）
│   └── nodes/
│       ├── __init__.py
│       ├── classify.py        # 意图分类节点
│       ├── rag.py             # RAG 检索节点
│       ├── generate.py        # 答案生成节点
│       ├── return_flow.py     # 退货子图（5步）
│       └── handoff.py         # 转人工节点
├── api/
│   ├── __init__.py
│   ├── routes.py              # /chat, /history, /handoff 路由
│   └── schemas.py             # 请求/响应模型
├── services/
│   ├── __init__.py
│   ├── llm.py                 # LLM 调用封装
│   ├── rag_client.py          # 外部 RAG HTTP 客户端
│   └── session_store.py       # 会话管理
└── tools/
    ├── __init__.py
    ├── order.py               # 订单查询工具
    └── return_policy.py       # 退货政策检查工具
```

## 架构图 (Agent Graph)

```
                  ┌──────────────┐
                  │  用户消息输入  │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │ classify_intent│  (LLM 识别意图)
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────────┐
          │              │                  │
   general_qa      return_request    human_support
          │              │                  │
   ┌──────▼──────┐  ┌───▼────────┐   ┌─────▼──────┐
   │rag_retrieve │  │return_flow │   │human_handoff│
   │ (HTTP→RAG)  │  │ (子图5步)   │   │ (记录+响应) │
   └──────┬──────┘  └───┬────────┘   └────────────┘
          │              │
   ┌──────▼──────┐       │
   │generate_resp│       │
   │ (LLM 生成)  │       │
   └──────┬──────┘       │
          │              │
          └──────┬───────┘
                 │
          ┌──────▼──────┐
          │    END      │
          └─────────────┘
```

## 退货子图 (Return Flow Subgraph)

```
return_start (询问订单号)
    │
return_validate_order (调用订单API验证)
    │
    ├── 订单无效 ──→ human_handoff
    │
return_check_policy (检查退货政策)
    │
    ├── 不可退 ──→ generate_response (告知原因)
    │
return_collect_reason (收集退货原因)
    │
return_initiate (创建退货单)
    │
    ├── 失败 ──→ human_handoff
    │
return_confirm (返回退货信息给用户)
    │
    END
```

## 核心数据结构 (AgentState)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 对话历史
    user_id: str
    intent: str                               # general_qa | return_request | human_support
    rag_context: str                          # RAG 检索结果
    return_order_id: str                      # 退货订单号
    return_reason: str                        # 退货原因
    return_step: str                          # 退货当前步骤
    return_eligible: bool                     # 是否可退
    handoff_reason: str                       # 转人工原因
    final_response: str                       # 最终回复
```

## 意图分类 Prompt

```
你是电商客服意图分类器。分析用户消息，输出JSON：
{
  "intent": "general_qa|return_request|human_support",
  "reason": "分类理由"
}

规则：
- general_qa: 商品咨询、政策询问、使用帮助等知识类问题
- return_request: 退货、退款、换货相关请求
- human_support: 用户明确要求转人工，或包含投诉、情绪激动内容
```

## 外部 RAG 接口约定

```
POST {RAG_BASE_URL}/api/retrieve
Body: {"query": "...", "top_k": 5}
Response: {"documents": [{"content": "...", "score": 0.95}, ...]}
```

## 转人工策略

以下情况自动转人工：
1. 用户明确说"转人工"/"人工客服"/"找真人"
2. 意图分类识别为 `human_support`
3. 退货流程中订单查询失败 2 次
4. LLM 返回异常或超时
5. RAG 服务不可用
6. 检测到投诉/辱骂关键词

转人工后：保存当前会话上下文 → 返回排队信息 → 通知人工坐席

## API 设计

```
POST /api/chat          # 对话接口
  Body: { "user_id": "...", "message": "...", "session_id": "..." }
  Response: { "response": "...", "intent": "...", "need_handoff": false }

GET  /api/history/{session_id}  # 获取历史

POST /api/handoff/{session_id}  # 管理后台接管会话
```

## 实现步骤

1. 创建项目骨架：requirements.txt, config.py, .env.example
2. 实现 services 层：llm.py, rag_client.py, session_store.py
3. 实现 agent state 和所有节点
4. 构建主图和退货子图
5. 实现 API 层 (FastAPI routes)
6. 编写 main.py 启动入口
7. 编写测试用例并验证

## 验证方式

1. 启动服务: `uvicorn main:app --reload`
2. 测试普通问答: `curl -X POST http://localhost:8000/api/chat -d '{"user_id":"u1","message":"这款手机支持5G吗？"}'`
3. 测试退货: `curl -X POST http://localhost:8000/api/chat -d '{"user_id":"u1","message":"我要退货"}'`
4. 测试转人工: `curl -X POST http://localhost:8000/api/chat -d '{"user_id":"u1","message":"转人工"}'`
