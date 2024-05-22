from typing import Dict, List, Union

import pandas as pd
from pydantic import BaseModel, Field
from pyvis.network import Network


class Ontology(BaseModel):
    """The framework of the knowledge graph, expressing its entities and relationships"""

    labels: List[Union[str, Dict]]
    relationships: List[str]

    def dump(self) -> dict:
        if len(self.relationships) == 0:
            return self.model_dump(exclude=["relationships"])
        else:
            return self.model_dump()


class Node(BaseModel):
    """A node of the graph expressing a single entity"""

    label: str = Field(..., description="label as per the ontology")
    name: str = Field(..., description="name of the entity")


class Edge(BaseModel):
    """An edge of the graph expressing the relationship between two nodes"""

    node_1: Node
    node_2: Node
    relationship: str = Field(
        ...,
        description="the relationship between node_1 and node_2 as per the context",
    )


class Graph(BaseModel):
    edges: List[Edge]

    def merge_edges(self, graph: "Graph") -> None:
        if len(graph.edges) > 0:
            self.edges += graph.edges

    def get_nodes_list(self) -> list[str]:
        nodes = []
        for edge in self.edges:
            nodes.extend([edge.node_1.name, edge.node_2.name])

        if len(nodes) == 0:
            return ["The graph is empty"]
        return list(set(nodes))

    def get_node_relationships(self, name: str) -> list[str]:
        relationships = []
        for edge in self.edges:
            if (edge.node_1.name == name) or (edge.node_2.name == name):
                relationships.append(
                    (
                        f"{edge.node_1.name} ({edge.node_1.label}) - "
                        f"{edge.node_2.name} ({edge.node_2.label}) -> "
                        f"{edge.relationship}"
                    )
                )

        if len(relationships) == 0:
            return [f'Node "{name}" is not present in the graph']
        return relationships

    def to_pandas(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        nodes, edges = [], []
        for edge in self.edges:
            nodes.extend(
                [
                    [edge.node_1.label, edge.node_1.name],
                    [edge.node_2.label, edge.node_2.name],
                ]
            )
            edges.append([edge.node_1.name, edge.node_2.name, edge.relationship])

        df_nodes = (
            pd.DataFrame(columns=["label", "name"], data=nodes)
            .value_counts()
            .to_frame()
            .reset_index()
            .assign(id=lambda df: df.index)
        )
        df_edges = (
            pd.DataFrame(columns=["node_1", "node_2", "relationship"], data=edges)
            .merge(
                df_nodes[["name", "id"]].rename(
                    columns={"name": "node_1", "id": "id_1"}
                ),
                how="left",
                on="node_1",
            )
            .merge(
                df_nodes[["name", "id"]].rename(
                    columns={"name": "node_2", "id": "id_2"}
                ),
                how="left",
                on="node_2",
            )
        )
        return df_nodes, df_edges

    def to_html(self) -> str:
        if len(self.edges) == 0:
            return ""

        df_nodes, df_edges = self.to_pandas()

        # add a color to each label
        _COLORS_ROTATION = [
            "#00bfff",  # DeepSkyBlue
            "#ff7f50",  # Coral
            "#32cd32",  # LimeGreen
            "#ff69b4",  # HotPink
            "#8a2be2",  # BlueViolet
            "#ff4500",  # OrangeRed
            "#2e8b57",  # SeaGreen
            "#dda0dd",  # Plum
            "#ff6347",  # Tomato
            "#4682b4",  # SteelBlue
            "#9acd32",  # YellowGreen
            "#ff1493",  # DeepPink
            "#00ced1",  # DarkTurquoise
            "#7b68ee",  # MediumSlateBlue
            "#dc143c",  # Crimson
        ]
        colors_map = {
            label: _COLORS_ROTATION[i % len(_COLORS_ROTATION)]
            for i, label in enumerate(df_nodes["label"].unique())
        }
        df_nodes["color"] = df_nodes["label"].map(colors_map)

        net = Network(notebook=False, cdn_resources="remote")
        for _, row in df_nodes.iterrows():
            net.add_node(
                n_id=row["id"],
                label=row["name"],
                color=row["color"],
                size=row["count"] * 10,
            )
        for _, row in df_edges.iterrows():
            net.add_edge(source=row["id_1"], to=row["id_2"], label=row["relationship"])

        return net.generate_html()
