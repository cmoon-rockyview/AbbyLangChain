import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Define embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Define the path for ChromaDB persistence
DB_PATH = r"D:\Work\Projects\RAG\chroma_db"

# Initialize Chroma vector store
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,  # Use the defined embeddings object
    collection_name="EmailHistory",
)

# Perform similarity search
query = """
park is good
"""

retriever = db.as_retriever(search_type="similarity")

result = retriever.invoke(query)
print(result)

# search_result = retriever.get_relevant_documents(query)
# print(search_result)

# results = db.similarity_search(query, k=1)
# results_score = db.similarity_search_with_score(query, k=1)

# for result in results_score:
#     document, score = result  # Unpacking the tuple
#     print(f"score: {score} Docu: {document.page_content}")


# # Print results
# for result in results:
#     with open("./results.txt", "w") as file:
#         for result in results:
#             file.write(result.page_content + "\n")
#             file.write("--------------------------------------------------\n")
