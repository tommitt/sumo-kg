from typing import Literal, Union

from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from sumo.agent.prompts import (
    DIRECT_LLM_SYSTEM_PROMPT,
    GENERATE_KG_SYSTEM_PROMPT,
    GRAPH_FORMAT_INSTRUCTIONS,
    ROUTER_SYSTEM_PROMPT,
)
from sumo.schemas import Graph
from sumo.settings import config


def get_llm(
    model: str = config.OPENAI_CHAT_MODEL,
    temperature: float = config.LLM_DEFAULT_TEMPERATURE,
) -> BaseChatModel:
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=config.OPENAI_API_KEY
    )


# Router
class RouteQuery(BaseModel):
    """Route a user query to the most relevant action."""

    source: Literal["generate_kg", "direct_llm"]


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
class GraphOutputParser(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        return GRAPH_FORMAT_INSTRUCTIONS

    def parse_result(
        self, result: list[ChatGeneration]
    ) -> Union[AIMessage, "GraphOutputParser"]:
        """
        Custom parser that accepts both a tool call or a pydantic-json output

        Args:
            result (list[ChatGeneration]): a list with one item, the generated response

        Returns:
            AIMessage if the result returned a tool call, the pydantic object otherwise
        """
        if "tool_calls" in result[0].message.additional_kwargs:
            return result[0].message
        return super().parse_result(result)


def generate_kg_llm() -> RunnableSerializable:
    llm = get_llm()
    parser = GraphOutputParser(pydantic_object=Graph)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATE_KG_SYSTEM_PROMPT + "\n\n{format_instructions}"),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


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
