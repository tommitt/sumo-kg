import logging

from sumo.llms import get_extraction_chain
from sumo.schemas import Graph, Ontology

logger = logging.getLogger("kg")


def create_kg(texts: list[str], ontology: Ontology) -> Graph:
    chain = get_extraction_chain()

    graph = Graph(edges=[])
    for i, text in enumerate(texts):
        logger.info(f"Extracting KG from text {i+1}/{len(texts)}")
        subgraph = chain.invoke({"ontology": ontology.dump(), "text": text})
        graph.merge_edges(subgraph)

    return graph
