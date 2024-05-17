import pickle

import streamlit as st
import streamlit.components.v1 as components

from sumo.schemas import Graph

st.set_page_config(layout="wide")

if "init" not in st.session_state:
    in_filename = ".scratchpad/outputs/graph.pkl"
    with open(in_filename, "rb") as f:
        graph: Graph = pickle.load(f)
    st.session_state["graph_html"] = graph.to_html()
    st.session_state["init"] = True

st.title("Sumo Knowledge Graph")
left, right = st.columns(2)

_CONTAINER_HEIGHT = 500
with left:
    with st.container(height=_CONTAINER_HEIGHT):
        with st.chat_message("ai"):
            st.write("Hello, here is your KG")
    message = st.chat_input()

with right:
    with st.container(height=_CONTAINER_HEIGHT):
        components.html(st.session_state["graph_html"], height=_CONTAINER_HEIGHT - 40)
