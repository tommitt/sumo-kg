import logging
import pickle

import streamlit as st
import streamlit.components.v1 as components

from sumo.agent import LlmAgent
from sumo.schemas import Graph, Ontology

logging.basicConfig(level=logging.INFO)
st.set_page_config(layout="wide")


if "init" not in st.session_state:
    st.session_state["init"] = True
    st.session_state["ontology_labels"] = ["Person", "Place"]
    st.session_state["ontology_relationships"] = [
        "Any relationship between two entities"
    ]
    st.session_state["messages"] = [
        ("ai", "Hello, can I help you to create a Knowledge Graph?")
    ]
    st.session_state["graph"] = Graph(edges=[])


def st_modifiable_list(name: str, options: list) -> None:
    def modify_ontology():
        st.session_state[f"ontology_{name.lower()}"] = st.session_state[f"{name}_list"]

    st.subheader(name)
    col1, col2 = st.columns(2)
    value = col1.text_input("Value", label_visibility="collapsed", key=name + "_value")
    button = col2.button("Add", key=name + "_add")
    if button and (value not in options + [""]):
        options.append(value)
    st.multiselect(
        "List",
        label_visibility="collapsed",
        options=options,
        default=options,
        key=name + "_list",
        on_change=modify_ontology,
    )


def load_kg():
    upload = st.session_state["uploaded_file"]
    if upload:
        st.session_state["graph"] = pickle.loads(upload.read())


_CONTAINER_HEIGHT = 450
kg: Graph = st.session_state["graph"]
kg_html = kg.to_html()

st.title("Sumo Knowledge Graph")

with st.sidebar:
    st.header("Ontology")
    st_modifiable_list("Labels", st.session_state["ontology_labels"])
    st_modifiable_list("Relationships", st.session_state["ontology_relationships"])

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Chat interface")
    left_container = st.container(height=_CONTAINER_HEIGHT)
    user_query = st.chat_input()

with col_right:
    st.subheader("Graph display")
    right_container = st.container(height=_CONTAINER_HEIGHT)
    col1, col2 = st.columns(2)
    col1.download_button("Download HTML", kg_html, "graph.html", disabled=(not kg_html))
    col2.download_button(
        "Download KG", pickle.dumps(kg), "graph.pkl", disabled=(not kg.edges)
    )
    _ = st.file_uploader("Import KG", on_change=load_kg, key="uploaded_file")

with left_container:
    for message in st.session_state["messages"]:
        st.chat_message(message[0]).markdown(message[1])

    if user_query:
        st.chat_message("human").markdown(user_query)

        with st.spinner("I'm thinking..."):
            agent = LlmAgent(
                ontology=Ontology(
                    labels=st.session_state["ontology_labels"],
                    relationships=st.session_state["ontology_relationships"],
                ),
                kg=kg,
            )
            agent_state = agent.run(user_query)

            st.session_state["messages"] += [
                ("human", user_query),
                ("ai", agent_state["generation"]),
            ]
            if "kg" in agent_state:
                st.session_state["graph"] = agent_state["kg"]

            st.rerun()

with right_container:
    components.html(kg_html, height=_CONTAINER_HEIGHT - 30)
