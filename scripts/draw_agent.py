from sumo.agent import LlmAgent
from sumo.schemas import Ontology

PNG_OUT_PATH = ".scratchpad/outputs/agent.png"

agent = LlmAgent(ontology=Ontology(labels=[], relationships=[]))
drawing = agent.graph.get_graph().draw_mermaid_png()

with open(PNG_OUT_PATH, "wb") as f:
    f.write(drawing)
