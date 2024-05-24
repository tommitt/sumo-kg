from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from sumo.agent.prompts import (
    DIRECT_LLM_SYSTEM_PROMPT,
    GENERATE_KG_SYSTEM_PROMPT,
    INVESTIGATE_KG_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from sumo.agent.tools import get_explore_kg_tool
from sumo.schemas import Graph
from sumo.settings import config


def get_llm(
    model: str = config.OPENAI_CHAT_MODEL,
    temperature: float = config.LLM_DEFAULT_TEMPERATURE,
) -> BaseChatModel:
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=config.OPENAI_API_KEY
    )


class CustomOutput(BaseModel):
    type: Literal["str", "pydantic", "tool"]
    tool_calls: list[dict] = None
    pydantic_object: BaseModel = None
    str_generation: str = None


class CustomOutputParser(BaseGenerationOutputParser):
    def __init__(self, pydantic_object: BaseModel | None = None) -> None:
        self.pydantic_object = pydantic_object

    def parse_result(self, result: list[ChatGeneration]) -> CustomOutput:
        if "tool_calls" in result[0].message.additional_kwargs:
            return CustomOutput(type="tool", tool_calls=result[0].message.tool_calls)
        elif self.pydantic_object:
            return CustomOutput(
                type="pydantic",
                pydantic_object=PydanticOutputParser(self.pydantic_object).parse_result(
                    result
                ),
            )
        return CustomOutput(
            type="str", str_generation=StrOutputParser().parse_result(result)
        )


# Router
class RouteQuery(BaseModel):
    """Route a user query to the most relevant action."""

    source: Literal["generate_kg", "investigate_kg", "direct_llm"]


def router_llm() -> RunnableSerializable:
    llm = get_llm()
    structured_llm_router = llm.with_structured_output(RouteQuery)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    return prompt | structured_llm_router


# Generate KG
def generate_kg_llm(kg: Graph) -> RunnableSerializable:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATE_KG_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    llm_with_tools = llm.bind_tools([get_explore_kg_tool(kg)]) if kg.edges else llm
    return prompt | llm_with_tools | CustomOutputParser(Graph)


# Investigate KG
def investigate_kg_llm(kg: Graph) -> RunnableSerializable:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INVESTIGATE_KG_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    llm_with_tools = llm.bind_tools([get_explore_kg_tool(kg)]) if kg.edges else llm
    return prompt | llm_with_tools | CustomOutputParser()


# Direct LLM
def direct_llm() -> RunnableSerializable:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DIRECT_LLM_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    return prompt | llm | StrOutputParser()
