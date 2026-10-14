# Document Loading
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./technova_policy.pdf")
policy_document = loader.load()



# Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=15)
chunks = splitter.split_documents(policy_document)



# Document Embedding
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)



# Creating Vector Storeage
from langchain_community.vectorstores import FAISS

VectorDataBase = FAISS.from_documents(chunks, embedding_model)
