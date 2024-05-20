from sumo.agent import LlmAgent

PNG_OUT_PATH = ".scratchpad/outputs/agent.png"

agent = LlmAgent()
drawing = agent.graph.get_graph().draw_mermaid_png()

with open(PNG_OUT_PATH, "wb") as f:
    f.write(drawing)
