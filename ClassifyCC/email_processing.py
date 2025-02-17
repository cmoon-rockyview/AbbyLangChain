import os, re
import extract_msg


def import_outlook_msg(file_path):
    """
    Parses an Outlook .msg file and returns an object with its metadata and content.

    Args:
        file_path (str): The path to the .msg file.

    Returns:
        dict: A dictionary containing the subject, body, sender, recipients, and attachments.
    """
    # Load the .msg file
    msg = extract_msg.Message(file_path)
    # msg.load()  # Ensure the message is fully loaded

    # Extract relevant information
    msg_data = {
        "subject": msg.subject,
        "body": msg.body,
        "sender": msg.sender,
        "to": msg.to,
        "cc": msg.cc,
        "bcc": msg.bcc,
        "date": msg.date,
        # "attachments": []
    }

    # Extract attachments if present
    # for attachment in msg.attachments:
    #     attachment_name = attachment.longFilename or attachment.shortFilename
    #     if attachment_name:
    #         attachment_path = os.path.join(os.getcwd(), attachment_name)
    #         attachment.save(customPath=attachment_path)  # Save attachment
    #         msg_data["attachments"].append({
    #             "filename": attachment_name,
    #             "filepath": attachment_path
    #         })

    return msg_data


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


def load_email_content_from_msg(folder_name: str, file_name: str) -> str:
    """Loads email content from an Outlook .msg file."""
    file_path = os.path.join(os.path.dirname(__file__), folder_name, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The email content file at {file_path} does not exist."
        )
    email_body = import_outlook_msg(file_path)["body"]
    return extract_without_first_email(email_body)


def load_email_content_from_text(folder_name: str, file_name: str) -> str:
    """Loads email content from a file."""
    file_path = os.path.join(os.path.dirname(__file__), folder_name, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The email content file at {file_path} does not exist."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
