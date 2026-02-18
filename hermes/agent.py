"""LangChain agent with unified STM/LTM tool access.

This module provides:
- `HermesAgent`: a LangChain AgentExecutor wired with two tools:
  - `stm_tool`
  - `ltm_tool`
"""

from __future__ import annotations
from typing import List, Literal
from memory import ShortTermMemory, LongTermMemory


def _require(pkg: str) -> None:
    raise SystemExit(
        f"Missing dependency: {pkg}.\n"
        "Install with: pip install langchain langchain-openai"
    )


try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import StructuredTool
except Exception:  # pragma: no cover
    _require("langchain")

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    _require("langchain-openai")


class HermesAgent:
    """LangChain tool-calling agent with one STM tool and one LTM tool."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        stm_capacity: int = 20,
    ) -> None:
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory()

        llm = ChatOpenAI(model=model, temperature=temperature)
        tools = self._build_tools()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Hermes, an agent that solves user tasks while managing memory. "
                    "Use `stm_tool` for immediate context (Retain/Discard/Retrieve/Summary/Filter). "
                    "Use `ltm_tool` for long-term storage (add/Retrieve/Update/Delete/get/list/clear). "
                    "Choose memory operations deliberately based on task relevance.",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    def _build_tools(self) -> List[StructuredTool]:
        stm_tool = StructuredTool.from_function(
            name="stm_tool",
            description=(
                "Operate on short-term memory. "
                "Actions: retain, discard, retrieve, summary, filter."
            ),
            func=self._stm_tool,
        )
        ltm_tool = StructuredTool.from_function(
            name="ltm_tool",
            description=(
                "Operate on long-term memory. "
                "Actions: add, retrieve, update, delete, get, list, clear."
            ),
            func=self._ltm_tool,
        )
        return [stm_tool, ltm_tool]

    def _stm_tool(
        self,
        action: Literal["retain", "discard", "retrieve", "summary", "filter"],
        content: str = "",
        k: int = 5,
    ) -> str:
        if action == "retain":
            return self.stm.retain(content)
        if action == "discard":
            return self.stm.discard(content)
        if action == "retrieve":
            items = self.stm.retrieve(content, k=k)
            return "\n".join(items) if items else "No STM matches."
        if action == "summary":
            return self.stm.summary()
        if action == "filter":
            items = self.stm.filter(content)
            return "\n".join(items) if items else "No STM matches."
        return f"Unsupported STM action: {action}"

    def _ltm_tool(
        self,
        action: Literal["add", "retrieve", "update", "delete", "get", "list", "clear"],
        content: str = "",
        key: str = "",
        k: int = 5,
        tags: Optional[List[str]] = None,
        importance: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        tags_any: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        if action == "add":
            out = self.ltm.add(content=content, tags=tags, meta=meta, importance=importance)
            return f"Added to LTM: {out.get('key')}" if out.get("ok") else f"Add failed: {out.get('error')}"
        
        if action == "retrieve":
            out = self.ltm.retrieve(query=content, k=k, tags_any=tags_any, meta_filter=meta_filter)
            hits = out.get("hits", []) if out.get("ok") else []
            if not hits:
                return "No LTM matches."
            total = out.get("total", len(hits))
            lines = [f"Found {len(hits)}/{total} matches:"]
            lines += [f"- {h['key']} (score={h['score']:.2f}): {h['content']}" for h in hits]
            return "\n".join(lines)
        
        if action == "update":
            out = self.ltm.update(key=key, content=content, tags=tags, meta=meta, importance=importance)
            return f"Updated LTM {key}" if out.get("ok") else f" Update failed: {out.get('error')}"
        
        if action == "delete":
           out = self.ltm.delete(key=key)
           return f"Deleted LTM {key}" if out.get("ok") else f"Delete failed: {out.get('error')}"
        
        if action == "get":
            out = self.ltm.get(key=key)
            if not out.get("ok"):
                return f" Get failed: {out.get('error')}"
            rec = out["record"]
            return f"{rec['key']}: {rec['content']} | tags={rec.get('tags', [])} | importance={rec.get('importance', 0.0)}"

        if action == "list":
            out = self.ltm.list()
            items = out.get("items", [])
            if not items:
                return "LTM is empty."
            return "\n".join([f"- {it['key']}: {it['preview']} (tags={it.get('tags', [])})" for it in items])

        if action == "clear":
            out = self.ltm.clear()
            return "Cleared LTM." if out.get("ok") else "Clear failed."

        return f"Unsupported LTM action: {action}"

    def run(self, task: str) -> str:
        """Execute one user task through the LangChain agent."""
        result = self.executor.invoke({"input": task})
        return str(result.get("output", ""))


if __name__ == "__main__":
    agent = HermesAgent()
    query = (
        "Remember that Miguel likes concise answers, add project goal, "
        "then summarize what you retained."
    )
    print(agent.run(query))
