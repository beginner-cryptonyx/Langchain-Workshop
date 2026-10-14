# Document Loading
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./technova_policy.pdf")
policy_document = loader.load()



# Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=15)
chunks = splitter.split_documents(policy_document)



