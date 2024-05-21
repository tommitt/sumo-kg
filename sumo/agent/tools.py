from typing import Annotated

from langchain_core.tools import tool


@tool
def explore_kg_tool(name: Annotated[str, "The name of the entity"]) -> str:
    """Explore all the relationships of a single node of the knowledge graph"""
    # TODO: retrieve relationships from kg
    return ""
