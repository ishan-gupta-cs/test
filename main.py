from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import dotenv
import os
import subprocess
import uuid
import webvtt

dotenv.load_dotenv()
key = os.getenv("api_key")
genai.configure(api_key = key)
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
    parsed_url = urlparse(youtube_url)
    query = parse_qs(parsed_url.query)
    if "v" in query:
        return query["v"][0]
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip('/')
    return None

def get_transcript(youtube_url):
    video_id = extract_video_id(youtube_url)
    if(not video_id):
        return "invalid youtube url"
    print("extracted video id")
    print("extracting transcript")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        full_text = "".join(t["text"] for t in transcript)
        print("transcript extracted")
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None, None
    return video_id, full_text


# def get_transcript(url):
#     if not url:
#         return "No URL provided"

#     # Unique ID for temp files
#     video_id = str(uuid.uuid4())
#     subtitle_file = f"{video_id}.en.vtt"

#     # yt-dlp command to download subtitles
#     command = [
#         "yt-dlp",
#         "--write-auto-sub",
#         "--sub-lang", "en",
#         "--skip-download",
#         "-o", video_id,
#         url
#     ]

#     try:
#         subprocess.run(command, check=True)

#         # Parse VTT file
#         transcript = ""
#         for caption in webvtt.read(subtitle_file):
#             transcript += caption.text + " "

#         # Clean up
#         os.remove(subtitle_file)

#         return transcript.strip()
#     except Exception as e:
#         return None, None




def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
    )
    chunks = splitter.split_text(text)
    return chunks

def store_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_relevant_chunks(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)
    return results

def ask_bot(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text
