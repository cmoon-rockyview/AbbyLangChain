import os
import extract_msg
import pyodbc
import re


def extract_without_first_email(email_thread: str) -> str:
    """Removes the first email from a series of emails in a given email thread and returns the rest."""

    # Define a regex pattern to detect email headers, assuming 'From:' starts each email
    email_split_pattern = r"(?=From:\s[\w\s]+<[^>]+>)"

    # Split emails in the thread
    emails = re.split(email_split_pattern, email_thread, flags=re.MULTILINE)

    # If multiple emails exist, remove the first one and return the rest
    if len(emails) > 1:
        modified_thread = "".join(
            emails[1:]
        )  # Reconstruct the thread without the first email
    else:
        modified_thread = ""  # If only one email exists, return an empty string

    return modified_thread


def extract_email_data(msg_file_path):
    """Extracts email data from an MSG file."""
    msg = extract_msg.Message(msg_file_path)
    ##msg.load()

    subject = msg.subject
    body = msg.body
    body1 = extract_without_first_email(msg.body)
    sender = msg.sender
    recipient = msg.to
    copy_to = msg.cc
    email_date = msg.date
    file_name = os.path.basename(msg_file_path)

    return subject, sender, recipient, body, body1, copy_to, email_date, file_name


def store_email_data_in_sql(connection_string, email_data):
    """Stores extracted email data into SQL Server."""
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO EmailHistory21(subject, sender, recipient, Body, Body1, CopyTo, email_date, file_name)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.execute(insert_query, email_data)
    conn.commit()
    conn.close()


def process_folder(folder_path, connection_string):
    """Processes all .msg files in the given folder and imports them into SQL Server."""
    for file in os.listdir(folder_path):
        if file.endswith(".msg"):
            msg_path = os.path.join(folder_path, file)
            email_data = extract_email_data(msg_path)
            store_email_data_in_sql(connection_string, email_data)


if __name__ == "__main__":
    folder_path = (
        r"D:\Work\EMails\History\2021 Correspondence"  # Change this to your folder path
    )
    connection_string = (
        "DRIVER={SQL Server};SERVER=gdbdev;DATABASE=AbbyCC;Trusted_Connection=yes;"
    )
    process_folder(folder_path, connection_string)
    print("All MSG files imported successfully.")
