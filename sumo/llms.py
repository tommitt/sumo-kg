from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from sumo.schemas import Graph


def get_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> BaseChatModel:
    return ChatOpenAI(model=model, temperature=temperature)


class GraphOutputParser(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        return (
            "Format your output as a json with the following schema. "
            "Do not add any other comment before or after the json. "
            "Respond ONLY with a well formed json that can be directly read by a program.\n\n"
            "{\n"
            "   edges: [\n"
            "      {\n"
            '         node_1: Required, an entity object with attributes: {"label": "as per the ontology", "name": "Name of the entity"},\n'
            '         node_2: Required, an entity object with attributes: {"label": "as per the ontology", "name": "Name of the entity"},\n'
            "         relationship: Describe the relationship between node_1 and node_2 as per the context, in a few sentences.\n"
            "      },\n"
            "   ]\n"
            "}\n"
        )


def get_extraction_chain() -> RunnableSerializable:
    llm = get_llm()

    parser = GraphOutputParser(pydantic_object=Graph)

    _EXTRACTION_SYSTEM_PROMPT = (
        "You are an expert at creating knowledge graphs. "
        "Consider the following ontology:\n\n{ontology}\n\n"
        "The user will provide you with an input text. "
        "Extract all the entities and relationships from the user-provided text as per the given ontology. "
        "Do not use any previous knowledge about the context. "
        "Remember there can be multiple direct (explicit) or implied relationships between the same pair of nodes. "
        "Be consistent with the given ontology. Use ONLY the labels and relationships mentioned in the ontology."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _EXTRACTION_SYSTEM_PROMPT + "\n{format_instructions}",
            ),
            ("human", "{text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser
