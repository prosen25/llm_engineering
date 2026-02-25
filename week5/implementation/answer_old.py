# from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Declare global constants
LLM_MODEL = "gpt-5-nano"
DB_NAME = "vector_db"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 5
SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgable, friendly assistant representing the compnay InsureLLM.
You are chatting with user about InsureLLM.
If relevant, use the below context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# load_dotenv(override=True)

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question

    Args:
        question (str): Question from the user
    Returns:
        list[Document]: List of relevant documents from vector store
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    return retriever.invoke(input=question, k=RETRIEVAL_K)

def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine the current question with the existing user questions in history

    Args:
        question (str): Current user question
        history (list[dict]): Old conversation history

    Returns:
        str: Combination of current and previous user questions
    """
    prior = "\n".join(h["content"] for h in history if h["role"] == "user")
    return prior + "\n" + question

def answer_question(question: str, history: list[dict] = []) -> str:
    """
    Answer the given question with RAG; fetch the relevant context
    and invoke llm to get the answer

    Args:
        question (str): User question
        history (list[dict]): Chat history

    Returns:
        str: Content of the response message
    """
    combined = combined_question(question=question, history=history)
    docs = fetch_context(question=combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(messages=history))
    messages.append(HumanMessage(content=question))

    llm = ChatOpenAI(temperature=0, model=LLM_MODEL)
    response = llm.invoke(input=messages)

    return response.content, docs