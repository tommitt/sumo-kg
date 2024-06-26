from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from sumo.schemas import Graph


class ExploreKgToolInput(BaseModel):
    name: str = Field(description="The name of the entity")


class ExploreKgTool:
    def __init__(self, kg: Graph) -> None:
        self._kg = kg

    def run(self, name: str) -> dict:
        return self._kg.get_node_relationships(name)


def get_explore_kg_tool(kg: Graph):
    return StructuredTool(
        name="explore_kg_tool",
        description="Explore all the relationships of a single node of the knowledge graph",
        args_schema=ExploreKgToolInput,
        func=ExploreKgTool(kg=kg).run,
        return_direct=False,
    )
