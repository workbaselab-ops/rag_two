from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROMPT = PromptTemplate.from_template(
    """아래 [문서]만 근거로 질문에 한국어로 답하세요.
[문서]에 답이 없으면 "문서에서 찾을 수 없습니다"라고 답하세요.

[문서]
{context}

[질문]
{question}

[답변]"""
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain(file_path="./data/virtual_tech_internal_handbook.pdf"):
    """PDF 경로를 받아 1부와 같은 RAG 체인을 만들어 돌려준다."""
    docs = PyMuPDFLoader(file_path).load()                                   # 1 로드
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)                              # 2 분할
    vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())       # 3·4 임베딩·저장
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})             # 5 리트리버
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)                     # 7 LLM
    return (                                                                 # 8 체인
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
