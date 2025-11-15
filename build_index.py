# # build_index.py
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# loader = TextLoader("knowledge.md", encoding="utf-8")
# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
# chunks = splitter.split_documents(docs)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# store = FAISS.from_documents(chunks, embeddings)
# store.save_local("vector-store")