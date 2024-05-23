import logging

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


def st_chat_containers(
    subtitle_left: str, subtitle_right: str, height: int, html: str | None = None
) -> tuple:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(subtitle_left)
        container_left = st.container(height=height)
        chat_input = st.chat_input()

    with col_right:
        st.subheader(subtitle_right)
        container_right = st.container(height=height)
        if html:
            st.download_button("Download HTML", html, "graph.html")

    return container_left, container_right, chat_input


st.title("Sumo Knowledge Graph")

with st.sidebar:
    st.header("Ontology")
    st_modifiable_list("Labels", st.session_state["ontology_labels"])
    st_modifiable_list("Relationships", st.session_state["ontology_relationships"])

_CONTAINER_HEIGHT = 450
kg_html = st.session_state["graph"].to_html()
left, right, user_query = st_chat_containers(
    "Chat interface", "Graph display", _CONTAINER_HEIGHT, kg_html
)

with left:
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
                kg=st.session_state["graph"],
            )
            agent_state = agent.run(user_query)

            st.session_state["messages"] += [
                ("human", user_query),
                ("ai", agent_state["generation"]),
            ]
            if "kg" in agent_state:
                st.session_state["graph"] = agent_state["kg"]

            st.rerun()

with right:
    components.html(kg_html, height=_CONTAINER_HEIGHT - 30)
