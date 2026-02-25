from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tenacity import retry, wait_exponential
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv(override=True)

MODEL_LLM = "gpt-5-nano"
MODEL_EMBEDDING = "text-embedding-3-large"
DB_NAME = "vector_db"
RETRIEVAL_K = 20
FINAL_K =10
SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

wait = wait_exponential(multiplier=1, min=10, max=240)

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

def make_rag_messages(question: str, history: list[dict], chunks: list[Document]) -> list[dict]:
    """
    Prepare the messages for final llm call to get the answer of the question

    Args:
        question (str): Question of the user
        history (list[dict]): Conversation history
        chunks (list[Document]): Relevant content chunks

    Returns:
        list[dict]: Messages to call the llm
    """

    context = "\n\n".join(chunk.page_content for chunk in chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(history)
    messages.append(HumanMessage(content=question))

    return messages

@retry(wait=wait)
def rerank(question: str, chunks: list[Document]) -> list[Document]:
    """
    Rerank the list of relevant content according to the user question

    Args:
        question (str): Question of the user
        chunks (list[Document]): Relevant content chunks to be reranked

    Returns:
        list[Document]: Reranked relevant content chunks
    """

    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=user_prompt))
    llm = ChatOpenAI(temperature=0, model=MODEL_LLM).with_structured_output(schema=RankOrder)
    response = llm.invoke(input=messages)

    order = response.order
    return [chunks[i - 1] for i in order]

def merge_chunks(chunks_original: list[Document], chunks_rewritten: list[Document]) -> list[Document]:
    """
    Take two chunks and merge into one by removing duplicate

    Args:
        chunks_original (list[Document]): Relevant content chunks retrieved for original question of the user
        chunks_rewritten (list[Document]): Relevant content chunks retrieved for rewritten question

    Returns:
        list[Document]: Merged chunks after removing duplicate
    """

    chunks_merged = chunks_original[:]
    existing = [chunk.page_content for chunk in chunks_original]
    for chunk in chunks_rewritten:
        if chunk.page_content not in existing:
            chunks_merged.append(chunk)

    return chunks_merged

def fetch_content_unranked(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question

    Args:
        question (str): Question from the user
    Returns:
        list[Document]: List of relevant documents from vector store
    """

    embeddings = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    return retriever.invoke(input=question, k=RETRIEVAL_K)

@retry(wait=wait)
def rewrite_query(question: str, history: list[dict]) -> str:
    """
    Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base.

    Args:
        question (str): Question of the user
        history (list[dict]): Conversation history

    Returns:
        str: Rewritten question to get relevant content from the knowledge base
    """

    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    messages = [SystemMessage(content=message)]
    llm = ChatOpenAI(temperature=0, model=MODEL_LLM)
    response = llm.invoke(input=messages)
    return response.content

def fetch_context(original_question: str, history: list[dict]) -> list[Document]:
    """
    Re-write the original question using llm to retrieve more relevant content from vector database.
    Fetch relevant content for both original and rewritten question.
    Merge the relevant content after removing duplicate.
    Rank the content chunks from most relevant to least relevant using llm.

    Args:
        original_question (str): Question of the user
        history (list[dict]): Conversation history

    Returns:
        list[Document]: List of chunks in order of most relevant to least relevant 
    """

    rewritten_question = rewrite_query(question=original_question, history=history)
    chunks_original = fetch_content_unranked(question=original_question)
    chunks_rewritten = fetch_content_unranked(question=rewritten_question)
    chunks_merged = merge_chunks(chunks_original=chunks_original, chunks_rewritten=chunks_rewritten)
    chunks_reranked = rerank(question=original_question, chunks=chunks_merged)

    return chunks_reranked[:FINAL_K]

@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer the given question with RAG; fetch the relevant context
    and invoke llm to get the answer

    Args:
        question (str): User question
        history (list[dict]): Chat history

    Returns:
        str: Content of the response message
        list: Chunks of duciments
    """

    chunks = fetch_context(original_question=question, history=history)
    messages = make_rag_messages(question=question, history=history, chunks=chunks)

    llm = ChatOpenAI(temperature=0, model=MODEL_LLM)
    response = llm.invoke(input=messages)

    return response.content, chunks