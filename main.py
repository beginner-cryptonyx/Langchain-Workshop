# Document Loading
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./technova_policy.pdf")
policy_document = loader.load()


# Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
chunks = splitter.split_documents(policy_document)


# Document Embedding
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Creating Vector Storeage
from langchain_community.vectorstores import FAISS

VectorDataBase = FAISS.from_documents(chunks, embedding_model)


# Prompt Template
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

Template = """
Use the following context to answer the question.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""


Prompt = PromptTemplate.from_template(Template)


# Making retriver for the 3 most relavent chunks
Retriever = VectorDataBase.as_retriever(search_kwargs={"k": 3})


# Function to pass retrieved data into prompt
def FormatRelaventChunks(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Defining the LanguageModel
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)


CompanyRAGChain = (
    {"context": Retriever | FormatRelaventChunks, "question": RunnablePassthrough()}
    | Prompt
    | llm
    | StrOutputParser()
)

Queries = [
    "How many paid annual leave days does a full-time employee receive per year?",
    "An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?",
    "What is the company's parental leave policy?",
]

for query in Queries:
    print(CompanyRAGChain.invoke(query))
    
# A full‑time employee receives **18 paid annual leave days** each calendar year.

# The policy permits remote work on Wednesdays, provided the employee emails the manager at least one working day beforehand. Friday is also a designated remote‑work day, but the employee is requesting leave, which is not covered by the remote‑work rules. Therefore, the Wednesday request can be approved under the remote‑work policy, while the Friday leave must follow the company’s separate leave‑approval process.

# The provided context does not include any information about a parental leave policy.