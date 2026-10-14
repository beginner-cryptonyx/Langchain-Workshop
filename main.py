# Document Loading
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("./technova_policy.pdf")
policy_document = loader.load()


