import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from config import SMTP_CONFIG
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


def send_email_notification(email_summary, email_content):
    """Sends an email notification with an enhanced email body format, emphasizing category."""
    msg = MIMEMultipart()
    msg["From"] = SMTP_CONFIG["sender_email"]
    msg["To"] = SMTP_CONFIG["receiver_email"]
    msg["Subject"] = f"📢 {email_summary.category.upper()} - {email_summary.subject}"

    email_body_html = f"""
    <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <p style="font-size:18px;">To those engaged in {email_summary.category},</p>
            <p>Below is a summary of the email received:</p>

            <!-- Emphasized Category Section -->
            <div style="background-color: #007bff; color: white; padding: 15px; text-align: center; font-size: 23px; font-weight: bold; border-radius: 5px;">
                🚀 CATEGORY: {email_summary.category.upper()}
            </div>

            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd; background-color: #f8f9fa;"><strong>👤 Sender:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{email_summary.email_sender}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>📩 Subject:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{email_summary.subject}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd; background-color: #f8f9fa;"><strong>📅 Date:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{email_summary.email_date}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>📜 Summary:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{email_summary.summary}</td>
                </tr>
            </table>

            <p style="margin-top: 20px;"><strong>📧 Original Email Content:</strong></p>
            <pre style="border-left: 3px solid #007bff; padding: 10px; color: #555; white-space: pre-wrap; word-wrap: break-word;">
                {email_content}
            </pre>

            <p>If you need further details, please feel free to reach out to IT Department.</p><br/>
            <p style="font-size:18px;">Best regards,<br><strong> Abbotsford AI Team </strong></p>
        </body>
    </html>
    """

    # Attach HTML content
    msg.attach(MIMEText(email_body_html, "html"))

    try:
        with smtplib.SMTP(SMTP_CONFIG["server"], SMTP_CONFIG["port"]) as server:
            server.sendmail(
                SMTP_CONFIG["sender_email"],
                SMTP_CONFIG["receiver_email"],
                msg.as_string(),
            )
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
