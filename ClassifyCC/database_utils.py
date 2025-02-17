import pyodbc
from dateutil import parser as util_parser
from config import DB_CONFIG


def get_db_connection():
    """Establishes a database connection."""
    conn_str = f"DRIVER={{{DB_CONFIG['driver']}}};SERVER={DB_CONFIG['server']};DATABASE={DB_CONFIG['database']};Trusted_Connection={DB_CONFIG['trusted_connection']}"
    return pyodbc.connect(conn_str)


def save_to_database(email_summary, email_content):
    """Saves the structured email summary to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    email_date = util_parser.parse(email_summary.email_date)
    email_date_str = email_date.strftime("%Y-%m-%d %H:%M:%S")

    incident_id = (
        f"{email_summary.email_sender}_{email_summary.subject}_{email_date_str}".replace(
            ",", ""
        )
        .replace(".", "")
        .replace("<", "")
        .replace(">", "")
    )

    cursor.execute(
        """
        INSERT INTO EmailSummary (category, reason_category, incident_id, person, subject, email_sender, email_recipient, email_date, summary, content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        email_summary.category,
        email_summary.reason_category,
        incident_id,
        email_summary.person,
        email_summary.subject,
        email_summary.email_sender,
        email_summary.email_recipient,
        email_date_str,
        email_summary.summary,
        email_content,
    )

    conn.commit()
    cursor.close()
    conn.close()
