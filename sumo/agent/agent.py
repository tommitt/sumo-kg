import logging
from typing import Literal, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from sumo.agent.llms import direct_llm, generate_kg_llm, router_llm
from sumo.agent.tools import get_explore_kg_tool
from sumo.schemas import Graph, Ontology
from sumo.settings import config

logger = logging.getLogger("llm")


class AgentState(TypedDict):
    query: str
    ontology: Ontology
    kg: Graph
    tool_calls: list[dict]
    explorations: list[str]
    generation: str


def input_router_edge(state: AgentState) -> Literal["generate_kg", "direct_llm"]:
    logger.info("---ROUTER---")
    query = state["query"]

    llm = router_llm()
    out = llm.invoke({"query": query})
    source = out["source"]

    logger.info(f"Routing to {source}\n")
    if source in ["generate_kg", "direct_llm"]:
        return source
    else:
        raise Exception(f"Router source {source} is not supported")


def tool_router_edge(state: AgentState) -> Literal["call_tool", "__end__"]:
    tool_calls = state["tool_calls"]

    if tool_calls:
        return "call_tool"
    return "__end__"


def generate_kg_node(state: AgentState) -> AgentState:
    _TEMPLATE_GENERATION = "KG correctly generated"

    logger.info("---GENERATE KG---")
    query = state["query"]
    ontology = state["ontology"]
    kg = state["kg"]
    explorations = state["explorations"]

    llm = generate_kg_llm()
    output = llm.invoke(
        {
            "query": query,
            "ontology": str(ontology.dump()),
            "nodes": kg.get_nodes_list(),
            "explorations": explorations,
        }
    )

    if isinstance(output, Graph):
        kg.merge_edges(output)
        logger.info(f"Query: {query}\nGenerated KG: {output}\n")
        return AgentState(kg=kg, generation=_TEMPLATE_GENERATION, tool_calls=[])

    logger.info(f"Query: {query}\nTool calling: {output.tool_calls}")
    return AgentState(tool_calls=output.tool_calls)


def direct_llm_node(state: AgentState) -> AgentState:
    logger.info("---DIRECT LLM---")
    query = state["query"]

    llm = direct_llm()
    generation = llm.invoke({"query": query})

    logger.info(f"Query: {query}\nAnswer: {generation}\n")
    return AgentState(generation=generation)


def explore_kg_tool_node(state: AgentState) -> AgentState:
    tool_calls = state["tool_calls"]
    kg = state["kg"]

    explorations = []
    explore_kg_tool = get_explore_kg_tool(kg)
    for call in tool_calls:
        result = explore_kg_tool.invoke(call["args"])
        explorations.append(result)

    return AgentState(tool_calls=[], explorations=explorations)


class LlmAgent:
    def __init__(
        self,
        ontology: Ontology = Ontology(labels=[], relationships=[]),
        kg: Graph = Graph(edges=[]),
    ) -> None:
        self._ontology = ontology
        self._kg = kg
        self.graph = self._compile()

    def _compile(self) -> CompiledGraph:
        graph = StateGraph(AgentState)

        # nodes
        graph.add_node("generate_kg", generate_kg_node)
        graph.add_node("direct_llm", direct_llm_node)
        graph.add_node("explore_kg_tool", explore_kg_tool_node)

        # edges
        graph.set_conditional_entry_point(
            input_router_edge,
            {
                "generate_kg": "generate_kg",
                "direct_llm": "direct_llm",
            },
        )
        graph.add_conditional_edges(
            "generate_kg",
            tool_router_edge,
            {"call_tool": "explore_kg_tool", "__end__": END},
        )
        graph.add_edge("explore_kg_tool", "generate_kg")
        graph.add_edge("direct_llm", END)

        return graph.compile()

    def run(self, query: str) -> AgentState:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=config.OPENAI_CHAT_MODEL,
            chunk_size=config.CHUNK_TOKENS_LIMIT,
            chunk_overlap=0,
        )
        queries = text_splitter.split_text(query)

        for i, q in enumerate(queries):
            logger.info(f"Executing agent for query {i+1}/{len(queries)}")
            state = self.graph.invoke(
                {
                    "query": q,
                    "ontology": self._ontology,
                    "kg": self._kg,
                    "explorations": [],
                },
                config={"recursion_limit": config.AGENT_STEPS_LIMIT},
            )
        return state
