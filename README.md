# Chatbot_Assignment
This is my project where I built an AI-powered chatbot that can answer questions from a given document set.
It uses a Retrieval-Augmented Generation (RAG) pipeline with a vector database (Chroma) and an open-source LLM.

# What it does
Takes a document (e.g., Terms & Conditions, Privacy Policy, etc.)

Splits it into smaller chunks (100–300 words)

Creates embeddings for each chunk and stores them in ChromaDB

Lets you ask natural language questions

Retrieves the most relevant chunks

Passes them to the LLM to generate an answer

Shows the source chunks used to answer

# Tech stack
Language model: stabilityai/stablelm-zephyr-3b

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector DB: Chroma (persistent mode)

Interface: Streamlit

Environment: Google Colab


1. Clone repo & install requirements

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

2. Unzip the vector DB
unzip vectordb/chroma_db.zip -d .

3. Run the Streamlit app
streamlit run app.py

How it works
Retriever
Uses Chroma to search for the most relevant document chunks based on your query.

Generator
Feeds those chunks + your question to the LLM with a custom prompt template.

Pipeline
Combines retrieval + generation and streams the answer back to the chat.

 Notes
If the answer is not in the document, the bot will say so.

Larger models may be slower in Colab, so I used a smaller, faster model.

The vector DB is already saved so you don’t need to reprocess the document every time.

iam attaching a video and screenshots of the chatbot functioanlity which shows how chatbot take question and provide the asnwers.

Here is the drive link for the video :
https://drive.google.com/drive/folders/1bJDoPbxt_wRfgY2psrlK-hQjf3UyyF2e?usp=sharing

OUTPUTS:
<img width="1366" height="768" alt="ss1" src="https://github.com/user-attachments/assets/9b14a443-1bea-4158-942d-6a95929e0137" />
<img width="1366" height="768" alt="ss2" src="https://github.com/user-attachments/assets/32deffa7-e622-455a-b5ac-ebc9b2dcddbb" />
<img width="1366" height="768" alt="ss3" src="https://github.com/user-attachments/assets/5bbcf066-cea8-4827-aded-8151dc005800" />
<img width="1366" height="768" alt="ss4" src="https://github.com/user-attachments/asset<img width="1366" height="768" alt="ss5" src="https://github.com/user-attachments/assets/8b598757-2f42-4d13-bb3d-755dd93e7dd8" />
s/5ac21e88-603e-442f-965b-63893ccd7371" />
