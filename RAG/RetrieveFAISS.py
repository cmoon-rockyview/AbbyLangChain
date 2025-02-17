import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from AbbyUtils import load_prompt

# Load environment variables
load_dotenv()


def retrieveMulti():
    # Define embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Define the path for FAISS database
    DB_PATH = r"D:\AbbyLangchain\RAG\faiss_db"

    # Load both FAISS indexes
    vectorstore_1 = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC23"
    )
    vectorstore_2 = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC24"
    )

    # Merge the two FAISS indexes
    vectorstore_1.merge_from(vectorstore_2)  # This combines both indexes into one

    # Create a retriever from the merged FAISS vector store
    retriever = vectorstore_1.as_retriever(search_kwargs={"top_k": 1})

    return retriever


def retrieve():
    # Define embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Define the path for ChromaDB persistence
    DB_PATH = r"D:\AbbyLangchain\RAG\faiss_db"

    # Initialize Chroma vector store
    vectorstore = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True, index_name="EHC24"
    )

    # Create a retriever from the FAISS vector store
    retriever = vectorstore.as_retriever(kwargs={"top_k": 1})

    return retriever


def create_chain(retriever, model_name="gpt-4o-mini"):

    prompt = load_prompt("prompts/EmailCat01.yaml", encoding="utf-8")

    llm = ChatOpenAI(model_name=model_name, temperature=0)

    chain = (
        {"context": retriever, "email_content": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


retrieve = retrieveMulti()
chain = create_chain(retrieve, model_name="gpt-4o-mini")

query = """    
From: Council Correspondence <CouncilCorrespondence@abbotsford.ca>
Sent: Thursday, January 23, 2025 3:05 PM
To: Les Barkman <LBarkman@abbotsford.ca>
Cc: Peter Sparanese <PSparanese@abbotsford.ca>; Council Correspondence <CouncilCorrespondence@abbotsford.ca>
Subject: 2025-01-16 and 20 Email - Councillor Barkman re Speeding and Truck Traffic on Gladwin Road

Hello Councillor Barkman,

For your information, below are the responses from staff to your emails.



AbbyPD – Sgt. Kelly’s response on January 23rd at 1:03pm - Regarding truck traffic - safety and speeding on Gladwin:

“Traffic Enforcement Unit (TEU)  is not back until tomorrow. I will be setting up our covert speed reading device and report back with speeds after a week of data recording.

TEU has also been tasked with ongoing truck traffic enforcement on Maclure Rd which has tied up enforcement in other areas. I will be looking at prioritizing our enforcement strategies accordingly starting tomorrow.”


ENG -  Gordon Botha’s response on January 21st at 9:31am - Regarding Non-Truck Route signage:

“We understand this to be an enforcement issue. Once trucks reach Gladwin at Haida or Gladwin at Downes, they are already a considerable distance away from a truck route. Trucks leaving Maclure Rd (a truck route) at Gladwin Rd or at Trethewey St (to get to Haida Dr) are passing by signs to indicate that they are leaving the truck route. In addition, we do have NO TRUCKS signs on Downes Rd at Gladwin Rd.

If the dump trucks are hauling from within the city urban containment boundary to a site at the corner of Gladwin Road and Townshipline Road (outside of the urban containment boundary), they should be heading east to Hwy 11, turning north onto Hwy 11, and turning left onto Harris Road and then left onto Gladwin Road to head south on Gladwin Road to Townshipline Road.”


BYLAWS – Magda Laljee’s response on January 21st at 3:59pm, following an update at 4:22pm - Regarding truck parking in a residential area as well as the illegal dump site at Gladwin and Townshipline:


  *   The illegal dump site at Gladwin and Townshipline is linked to a long-standing court case, file number 32863 at Townshipline, which is currently still before the courts. This case involves City of Abbotsford v. Randhawa (Provincial Court File No. 96616-1). It is my understanding that Council has received a legal update regarding this case.


  *   Regarding commercial truck parking in the residential area, Bylaw Services addressed the concern related to the black truck (see attached photo). Bylaw Services received the service request from Councillor Barkman on January 9, 2025, and compliance was achieved by January 15, 2025. This file is now closed.



As for the red truck (see below photo), no file has been created yet. I have asked staff to conduct a quick inspection of the area, but it would be helpful if Councillor Barkman could provide specific addresses where concerns have been raised.
[cid:image007.png@01DB6C20.1F138D90]






























An update on January 21st at 4:22pm:

Regarding the red commercial vehicle. Staff were able to find the property identified with the red commercial vehicle but no commercial vehicle was observed on site.  We will provide a warning and education to the residents of the property regarding the City’s commercial vehicle parking regulations.

Thank you,
Muhsina

Muhsina Haq, BBA, Pilot
Administrative Assistant, Executive Office
Tel:   604-853-2281 Ext. 5793
Email: mhaq@abbotsford.ca<mailto:mhaq@abbotsford.ca>
[City-Email-Signature]


From: Council Correspondence
Sent: Monday, January 20, 2025 12:39 PM
To: Les Barkman <LBarkman@abbotsford.ca<mailto:LBarkman@abbotsford.ca>>
Cc: Council Correspondence <CouncilCorrespondence@abbotsford.ca<mailto:CouncilCorrespondence@abbotsford.ca>>
Subject: 2025-01-16 and 20 Email - Councillor Barkman re Speeding and Truck Traffic on Gladwin Road

Hello Councillor,
We are in receipt of your emails of January 16 and 20.  and will be directed to the appropriate staff including AbbyPD for their review and follow-up.
An update will be provided to you.
With thanks,
Muhsina Haq, BBA, Pilot
Administrative Assistant, Executive Office
Tel:   604-853-2281 Ext. 5793
Email: mhaq@abbotsford.ca<mailto:mhaq@abbotsford.ca>
[City-Email-Signature]
From: Les Barkman <LBarkman@abbotsford.ca<mailto:LBarkman@abbotsford.ca>>
Date: January 20, 2025 at 9:06:00 AM PST
To: Peter Sparanese <PSparanese@abbotsford.ca<mailto:PSparanese@abbotsford.ca>>
Subject: FYI
?
Morning Peter:
These pics are to supplement my email sent January 16. 2025 re: truck traffic to an illegal dump site on Townshipline and Gladwin Rd. These pics were taken a minute apart shortly after 7:00 AM January 20, 2025.
Thanks for your time
Councillor Barkman

[IMG_8719.jpg]


[IMG_8720.jpg]

Sent from my iPad

From: Les Barkman <LBarkman@abbotsford.ca<mailto:LBarkman@abbotsford.ca>>
Date: January 16, 2025 at 2:45:07 PM PST
To: Peter Sparanese <PSparanese@abbotsford.ca<mailto:PSparanese@abbotsford.ca>>, Council Correspondence <CouncilCorrespondence@abbotsford.ca<mailto:CouncilCorrespondence@abbotsford.ca>>
Subject: FYI
?
Hi Peter:
Commercial truck traffic from Maclure to Downs Rd needs some attention for a few reasons. Excessive speeds and long skid marks at Haida and Gladwin confirm what I’ve observed. There are no “No truck route” signs at Gladwin and Haida or prior to Gladwin and Downs. Gladwin still being used to travel to illegal dump site at Gladwin and Townshipline. Four schools on Gladwin are affected by Commerical truck non truck route traffic and is a safety concern. By-laws is aware of residential parking.
Thank you for your time.
CB



"""
response = chain.stream(query)

final_answer = ""
for token in response:
    final_answer += token
    print(token, end="", flush=True)
