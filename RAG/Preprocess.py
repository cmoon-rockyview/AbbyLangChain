import pandas as pd
import urllib
from sqlalchemy import create_engine
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

# Database connection parameters
server = "GDBDEV"
database = "AbbyCC"
table_name = "EHC"

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

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Define ChromaDB persistence directory (use a folder, not a .db file)
DB_PATH = "./chroma_db"
persist_db = Chroma.from_documents(
    raw_data, embeddings, persist_directory=DB_PATH, collection_name="EHC"
)


#####Retrieve collection metadata
all_ids = persist_db.get()["ids"]
collection_name = persist_db._collection.name

# Print the collection name to confirm
print(collection_name)
