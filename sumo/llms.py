from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from sumo.schemas import Graph


def get_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> BaseChatModel:
    return ChatOpenAI(model=model, temperature=temperature)


def get_extraction_chain() -> RunnableSerializable:
    llm = get_llm()

    parser = PydanticOutputParser(pydantic_object=Graph)

    EXTRACTION_SYSTEM_PROMPT = (
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
                EXTRACTION_SYSTEM_PROMPT + "\n{format_instructions}",
            ),
            ("human", "{text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser
