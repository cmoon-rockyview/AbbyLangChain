import streamlit as st
import pandas as pd
import pyodbc  # Or use pymssql

# SQL Server Connection Configuration
server = "GDBDEV"
database = "AbbyCC"

# Establish connection using OSA (Windows Authentication)
conn_str = (
    f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
)
conn = pyodbc.connect(conn_str)

# SQL Query
query = """SELECT 
       [category]
      ,[reason_category]
      ,[person]
      ,[subject]
      ,[email_sender]
      ,[email_recipient]
      ,[email_date]
      ,[summary]
      ,[content]
  FROM [AbbyCC].[dbo].[EmailSummary]
"""


# Fetch Data
@st.cache_data
def fetch_data():
    df = pd.read_sql(query, conn)

    # Convert email_date to readable format
    df["email_date"] = pd.to_datetime(df["email_date"]).dt.strftime("%Y-%m-%d")

    # Truncate long texts for better display
    df["summary"] = df["summary"].apply(
        lambda x: x[:100] + "..." if len(x) > 100 else x
    )
    df["content"] = df["content"].apply(
        lambda x: x[:100] + "..." if len(x) > 100 else x
    )

    return df


# Streamlit UI
st.title("📩 Council Correspondence Data")
st.write("Displaying email correspondence data in a well-structured table.")

# Load Data
try:
    df = fetch_data()

    # Apply enhanced table styling
    st.markdown(
        """
        <style>
        /* Table Styling */
        .dataframe tbody tr:nth-child(odd) { background-color: #f9f9f9; }
        .dataframe tbody tr:hover { background-color: #f1f1f1 !important; }
        .dataframe thead { background-color: #007bff; color: white; text-align: center; font-size: 16px; }
        .dataframe td, .dataframe th { padding: 10px; border: 1px solid #ddd; }
        .dataframe { border-radius: 10px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display Data with Styled Table
    st.dataframe(
        df.style.set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#007bff"),
                        ("color", "white"),
                        ("font-size", "16px"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(odd)",
                    "props": [("background-color", "#f9f9f9")],
                },
                {
                    "selector": "tbody tr:hover",
                    "props": [("background-color", "#f1f1f1")],
                },
            ]
        ).set_properties(
            **{
                "border": "1px solid #ddd",
                "font-size": "14px",
                "text-align": "left",
                "padding": "10px",
            }
        )
    )

except Exception as e:
    st.error(f"Error fetching data: {e}")

# Close connection
conn.close()
