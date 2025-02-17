import pandas as pd
import urllib
from sqlalchemy import create_engine
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

# Database connection parameters
server = "GDBDEV"
database = "AbbyCC"
table_name = "EHC23"

# Define connection string using SQLAlchemy
params = urllib.parse.quote_plus(
    f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Read data into Pandas DataFrame
query = f"SELECT * FROM {table_name}"
df = pd.read_sql(query, engine)

# Load data using LangChain's DataFrameLoader
loader = DataFrameLoader(df, page_content_column="Body")
raw_data = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Define ChromaDB persistence directory (use a folder, not a .db file)
db = FAISS.from_documents(raw_data, embeddings)
db.save_local(folder_path="./faiss_db", index_name="EHC23")

# # Retrieve the FAISS index
# index = db.index

# # Get the IDs of the vectors
# ids = index.id_map

# # Print the IDs
# print(ids)
