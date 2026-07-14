"""Response-generation node — produces the final answer sent to the user."""

from langchain_core.messages import AIMessage

from agent.state import AgentState
from services.llm import llm_generate

SYSTEM_PROMPT = """你是专业的电商客服助手。请根据提供的知识库内容回答用户问题。

要求：
1. 回答准确、简洁、友好，使用中文
2. 如果知识库包含相关信息，优先基于知识库回答，不要编造
3. 如果知识库没有相关信息，诚实告知用户，并建议联系人工客服获取更准确的答案
4. 回复末尾不要添加"还有什么可以帮您"之类的冗余收尾
5. 涉及价格、库存等实时信息时，提醒用户以页面显示为准
"""

SYSTEM_PROMPT_NO_RAG = """你是专业的电商客服助手。

当前知识库暂时无法访问，请：
1. 先向用户致歉
2. 对于一般性问题，基于你的知识尽量回答
3. 对于订单/退货/投诉等需要系统操作的问题，告知用户正在转接人工客服
4. 回复简洁、友好，使用中文
"""


async def generate_response(state: AgentState) -> dict:
    """Generate the final customer-facing response."""
    messages = state["messages"]
    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    rag_context = state.get("rag_context", "")

    if rag_context:
        prompt = SYSTEM_PROMPT
        context = rag_context
    else:
        prompt = SYSTEM_PROMPT_NO_RAG
        context = "知识库暂时不可用"

    response_text = await llm_generate(prompt, context, user_text)

    return {
        "final_response": response_text,
        "messages": [AIMessage(content=response_text)],
    }
