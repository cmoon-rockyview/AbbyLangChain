import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DB_CONFIG = {
    "driver": "ODBC Driver 17 for SQL Server",
    "server": os.getenv("DB_SERVER", "gdbdev"),
    "database": os.getenv("DB_NAME", "AbbyCC"),
    "trusted_connection": "yes",
}

# SMTP Configuration
SMTP_CONFIG = {
    "server": os.getenv("SMTP_SERVER", "smtp-exchange.abbotsford.loc"),
    "port": int(os.getenv("SMTP_PORT", 25)),
    "sender_email": os.getenv("SENDER_EMAIL", "Abby_CC@abbotsford.ca"),
    "receiver_email": os.getenv("RECEIVER_EMAIL", "chmoon@abbotsford.ca"),
}
