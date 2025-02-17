from email_processing import load_email_content_from_msg
from database_utils import save_to_database
from notification_service import send_email_notification
from llm_service import (
    initialize_llm,
    load_email_prompt,
    process_email_content,
    initialize_llm_str,
)


def main():
    # Load email content
    # email_content = load_email_content_from_msg(
    #     "Emails_msg",
    #     "2024-06-26 Email - Doreen McMillan re Homeless camp in Babich Park.msg",
    # )
    email_content = """
    From: Katherine Treloar <KTreloar@abbotsford.ca>
Sent: Thursday, February 6, 2025 11:56 AM
To: sonashaukath768@gmail.com
Cc: Council Correspondence <CouncilCorrespondence@abbotsford.ca>
Subject: RE: 2025-02-04 Email - Sona Shaukath re Request for Assistance in Expedited Processing of BC PNP Application

Hello Sona,

Mayor Siemens forwarded your email to me to respond to you directly.

Thank you for your email of February 4 and for your kind words about the City of Abbotsford. Abbotsford City Council aims to foster an inclusive and connected community, and I’m happy to hear your experiences in Abbotsford have been positive. I appreciate your contributions to our community and your dedication to making British Columbia your home.

Regarding your BC PNP application, unfortunately, I am unable to assist you as this program is administered by the Government of BC and not the City of Abbotsford. You may wish to consider contacting your local MLA<https://www.leg.bc.ca/members/mla-by-community>, who will be best positioned to address this as a local representative for the Province.

I hope you receive the necessary updates on your application soon and that you can visit your grandfather without undue stress.

Sincerely,

Katherine

Katherine Treloar, MBA
Pronouns: she/her
General Manager, Innovation, Strategy and Intergovernmental Relations
City of Abbotsford
T: 604-557-4421 E: ktreloar@abbotsford.ca<mailto:ktreloar@abbotsford.ca>     www.abbotsford.ca<http://www.abbotsford.ca/>
[City-Email-Signature]
The information transmitted herein is confidential and may contain privileged information.  It is intended solely for the person(s) or entity(s) to which it is addressed.  Any review, re-transmission, dissemination, taking of any action in reliance upon, or other use of this information by persons or entities other than the intended recipient(s) is prohibited.  If you received this in error, please notify the sender and delete or destroy all copies.

From: sona shaukath <sonashaukath768@gmail.com<mailto:sonashaukath768@gmail.com>>
Sent: Tuesday, February 4, 2025 2:15 PM
To: Ross Siemens <RSiemens@abbotsford.ca<mailto:RSiemens@abbotsford.ca>>
Subject: Request for Assistance in Expedited Processing of BC PNP Application

?Dear Mayor Ross Siemens,

I hope this email finds you well. I want to begin by expressing my admiration for the work you and your office do in supporting the community of Abbotsford. Over the past year of living here, I have come to appreciate the city's strong sense of community and the opportunities it provides for newcomers like myself to contribute meaningfully.

I am reaching out to you with an urgent request regarding my BC PNP application. My file number is BCSA-24-20595, and I submitted my application on June 18, 2024. I have applied for BC PNP through the Skills Immigration- International Post Graduate (Non- Express Entry) stream. My full name is SONA SHAUKATH and my date of birth : 16-01-1999(dd-mm-yyyy). Unfortunately, I have yet to receive any updates on its status. While I understand that these processes take time, I am in a difficult situation as my grandfather has been hospitalized in my home country, and I am in urgent need of visiting him. However, without clarity on my application status, I am unable to make necessary travel arrangements.

I completed my Master of Science in Data Science from Thompson Rivers University, Kamloops, in 2024, and since then, I have been actively seeking a technical job in British Columbia to further contribute to the province’s growth in the technology sector. In the meantime, I have been dedicated to serving the Abbotsford community by working as a permanent part-time Customer Service and Post Office Specialist at London Drugs for the past three years (two years in Kamloops, one in Abbotsford). Additionally, I have actively volunteered in community initiatives to give back to the city that has welcomed me.

I am truly committed to building my future in British Columbia and contributing to its economy and community. I kindly request your support in helping me obtain an update on my BC PNP application so that I can make informed decisions regarding my urgent need to travel. I would deeply appreciate any guidance or assistance you can provide in facilitating communication with the relevant authorities.

Thank you for taking the time to read my request. I greatly appreciate your leadership and all that you do for Abbotsford. I hope to continue contributing to this wonderful city in even greater ways once I secure a role in my field. Please let me know if you need any additional details regarding my application.

Looking forward to your kind support.

Best regards,
Sona Shaukath
sonashaukath768@gmail.com<mailto:sonashaukath768@gmail.com>
7828823388
34882 2nd Ave, Abbotsford, BC
V2S 0A8

    """

    # Initialize LLM and Parser
    llm, parser = initialize_llm()
    # Load Email Summary Prompt
    prompt_email = load_email_prompt(parser)

    # Process Email Content
    structured_response = process_email_content(
        llm, parser, prompt_email, email_content
    )

    print(structured_response)

    # Save structured response to database
    # save_to_database(structured_response, email_content)

    # # Send email notification
    # send_email_notification(structured_response, email_content)


if __name__ == "__main__":
    main()
