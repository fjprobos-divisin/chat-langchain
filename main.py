"""Main entrypoint for the app."""
import asyncio
import json
import os
from operator import itemgetter
from typing import AsyncIterator, Dict, List, Optional, Sequence

import langsmith
import weaviate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.vectorstores import Weaviate
from langsmith import Client
from pydantic import BaseModel

from constants import WEAVIATE_DOCS_INDEX_NAME

# This part of the prompt was left behind to avoid problems with citing after sources post processing
# Should be placed after "Do not repeat text."
CITING_PROMPT = """
Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.
"""

RESPONSE_TEMPLATE = """\
You are an expert real estate analyst, tasked with answering any question \
about CHMC reports.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (files, pages and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

# Load json with reports metadata
with open('reports_metadata.json', 'r') as f:
    data_list = json.load(f)
metadata_dict = {item["file"]: item for item in data_list}


client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


def get_retriever() -> BaseRetriever:
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=200),
        by_text=False,
        attributes=["source", "page", "file"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=8))


def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever, use_chat_history: bool
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    if not use_chat_history:
        initial_chain = (itemgetter("question")) | retriever
        return initial_chain
    else:
        condense_question_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
        ).with_config(
            run_name="CondenseQuestion",
        )
        conversation_chain = condense_question_chain | retriever
        return conversation_chain


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def remove_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    use_chat_history: bool = False,
) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm, retriever, use_chat_history
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return _context | response_synthesizer


def create_test_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm, retriever, use_chat_history=False
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return _context | response_synthesizer


async def transform_stream_for_client(
    stream: AsyncIterator[RunLogPatch],
) -> AsyncIterator[str]:
    async for chunk in stream:
        for op in chunk.ops:
            if op["path"] == "/logs/0/final_output":
                seen_sources = set()
                all_sources = []

                for doc in op["value"]["output"]:
                    current_source = doc.metadata["source"]
                    if current_source not in seen_sources and doc.metadata.get("file"):
                        seen_sources.add(current_source)

                        source_info = {
                            "source": current_source,
                            # "page": doc.metadata["page"],
                            "file": doc.metadata["file"],
                            "name": metadata_dict[remove_prefix(doc.metadata["file"], prefix="reports/")].get("name"),
                            "author": metadata_dict[remove_prefix(doc.metadata["file"], prefix="reports/")].get(
                                "author"),
                            "date_published": metadata_dict[remove_prefix(doc.metadata["file"], prefix="reports/")].get(
                                "date_published")
                        }
                        all_sources.append(source_info)

                if all_sources:
                    src = {"sources": all_sources}
                    yield f"{json.dumps(src)}\n"

            elif op["path"] == "/streamed_output/-":
                # Send stream output
                yield f'{json.dumps({"tok": op["value"]})}\n'

            elif not op["path"] and op["op"] == "replace":
                yield f'{json.dumps({"run_id": str(op["value"]["id"])})}\n'


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]]
    conversation_id: Optional[str]


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global trace_url
    trace_url = None
    question = request.message
    chat_history = request.history or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))

    metadata = {
        "conversation_id": request.conversation_id,
    }

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
    )
    retriever = get_retriever()
    answer_chain = create_chain(
        llm,
        retriever,
        use_chat_history=bool(converted_chat_history),
    )
    stream = answer_chain.astream_log(
        {
            "question": question,
            "chat_history": converted_chat_history,
        },
        config={"metadata": metadata},
        include_names=["FindDocs"],
        include_tags=["FindDocs"],
    )
    return StreamingResponse(transform_stream_for_client(stream))


@app.post("/feedback")
async def send_feedback(request: Request):
    data = await request.json()
    run_id = data.get("run_id")
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    key = data.get("key", "user_score")
    vals = {**data, "key": key}
    client.create_feedback(**vals)
    return {"result": "posted feedback successfully", "code": 200}


@app.patch("/feedback")
async def update_feedback(request: Request):
    data = await request.json()
    feedback_id = data.get("feedback_id")
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=data.get("score"),
        comment=data.get("comment"),
    )
    return {"result": "patched feedback successfully", "code": 200}

@app.get("/ping")
def pong():
    return {"ping": "pong!"}


# TODO: Update when async API is available
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


@app.post("/get_trace")
async def get_trace(request: Request):
    data = await request.json()
    run_id = data.get("run_id")
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(run_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
