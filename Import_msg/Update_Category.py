import pyodbc
from llm_service import initialize_llm, load_email_prompt, process_email_content

# Define database connection details using Windows Authentication (Integrated Security)
DB_CONFIG = {
    "server": "GDBDEV",
    "database": "AbbyCC",
}


def update_category_ai(category_value, Id):
    """
    Updates the 'CategoryAI' column in the 'EmailHistoryT' table with a given value.

    Args:
        category_value (str): The new value for CategoryAI (default is 'ISRI').
        condition (str): Optional WHERE condition to filter specific rows (default is None).

    Returns:
        int: Number of rows updated.
    """

    try:
        # Establish connection using Windows Authentication
        conn = pyodbc.connect(
            f"DRIVER={{SQL Server}};SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};Trusted_Connection=yes;"
        )
        cursor = conn.cursor()

        # Build the UPDATE SQL query
        sql_query = f"""
        UPDATE [dbo].[EmailHistoryT]
        SET [CategoryAI3] = ?
        WHERE [Id] = ?
        """

        # Execute the query
        cursor.execute(sql_query, (category_value, Id))
        rows_affected = cursor.rowcount  # Get number of updated rows
        conn.commit()  # Commit changes

        print(f"✅ {rows_affected} rows updated successfully.")

        # Close the connection
        cursor.close()
        conn.close()
        return rows_affected

    except Exception as e:
        print(f"❌ Error updating database: {e}")
        return 0


def get_record_by_id(record_id):
    """
    Retrieves a record from the 'EmailHistoryT' table by ID.

    Args:
        record_id (int): The ID of the record to retrieve.

    Returns:
        dict: A dictionary containing the record details, or None if not found.
    """
    try:
        # Establish connection using Windows Authentication
        conn = pyodbc.connect(
            f"DRIVER={{SQL Server}};SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};Trusted_Connection=yes;"
        )
        cursor = conn.cursor()

        # Define SQL query to fetch record by ID
        sql_query = """
        SELECT [Category], [Body1], [CategoryAI], [Category1]
        FROM [dbo].[EmailHistoryT]
        WHERE ID = ?
        """

        # Execute the query with the provided ID
        cursor.execute(sql_query, (record_id,))
        row = cursor.fetchone()  # Fetch one record

        # Close the database connection
        cursor.close()
        conn.close()

        if row:
            # Convert row to dictionary
            record = {
                "Category": row[0],
                "Body1": row[1],
                "CategoryAI": row[2],
                "Category1": row[3],
            }
            return record
        else:
            print(f"⚠️ No record found with ID: {record_id}")
            return None

    except Exception as e:
        print(f"❌ Error retrieving record: {e}")
        return None


def main():
    for i in range(0, 10):
        # Load email content
        email_content = get_record_by_id(i)

        if email_content:
            # Initialize LLM and Parser
            llm, parser = initialize_llm()

            # Load Email Summary Prompt
            prompt_email = load_email_prompt(parser)

            # Process Email Content
            structured_response = process_email_content(
                llm, parser, prompt_email, email_content
            )

            update_category_ai(structured_response.category, i)


if __name__ == "__main__":
    main()
