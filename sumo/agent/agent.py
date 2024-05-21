import logging
from typing import Literal, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from sumo.agent.llms import direct_llm, generate_kg_llm, router_llm
from sumo.schemas import Graph, Ontology
from sumo.settings import config

logger = logging.getLogger("llm")


class AgentState(TypedDict):
    query: str
    ontology: Ontology
    kg: Graph
    generation: str


def router_edge(state: AgentState) -> Literal["generate_kg", "direct_llm"]:
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


def generate_kg_node(state: AgentState) -> AgentState:
    _TEMPLATE_GENERATION = "KG correctly generated"

    logger.info("---GENERATE KG---")
    query = state["query"]
    ontology = state["ontology"]
    kg = state["kg"]

    llm = generate_kg_llm()
    new_kg = llm.invoke(
        {
            "query": query,
            "ontology": str(ontology.dump()),
            "nodes": kg.get_nodes_list(),
        }
    )
    kg.merge_edges(new_kg)

    logger.info(f"Query: {query}\nGenerated KG: {new_kg}\n")
    return AgentState(kg=kg, generation=_TEMPLATE_GENERATION)


def direct_llm_node(state: AgentState) -> AgentState:
    logger.info("---DIRECT LLM---")
    query = state["query"]

    llm = direct_llm()
    generation = llm.invoke({"query": query})

    logger.info(f"Query: {query}\nAnswer: {generation}\n")
    return AgentState(generation=generation)


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

        # edges
        graph.set_conditional_entry_point(
            router_edge,
            {
                "generate_kg": "generate_kg",
                "direct_llm": "direct_llm",
            },
        )
        graph.add_edge("generate_kg", END)
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
                {"query": q, "ontology": self._ontology, "kg": self._kg},
                config={"recursion_limit": config.AGENT_STEPS_LIMIT},
            )
        return state
