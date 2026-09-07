import os
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    YouTubeTranscriptApiException,
    InvalidVideoId,
)
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

embeddings = OllamaEmbeddings(model="bge-m3")


def fetch_transcript(video_id: str) -> str:
    """
    Fetches transcript for a YouTube video ID.
    Attempts to fetch English captions first. If unavailable, translates
    available captions to English. If translation is not supported,
    falls back to native captions.
    """
    yt = YouTubeTranscriptApi()
    try:
        transcript_list = yt.list(video_id)
    except TranscriptsDisabled:
        raise Exception("Subtitles/captions are disabled for this video.")
    except InvalidVideoId:
        raise Exception("Invalid YouTube Video ID provided.")
    except YouTubeTranscriptApiException as e:
        raise Exception(f"Could not retrieve transcript from YouTube: {e}")
    except Exception as e:
        raise Exception(f"Failed to fetch video transcripts: {e}")

    # 1. Try fetching English captions
    transcript_obj = None
    try:
        transcript_obj = transcript_list.find_transcript(["en", "en-US", "en-GB", "en-CA", "en-IN", "en-AU"])
    except NoTranscriptFound:
        # 2. Try translating any caption to English
        for t in transcript_list:
            if t.is_translatable:
                try:
                    transcript_obj = t.translate("en")
                    break
                except Exception:
                    pass
        # 3. Fallback to first available caption in native language
        if not transcript_obj:
            for t in transcript_list:
                transcript_obj = t
                break

    if not transcript_obj:
        raise Exception("No readable captions found for this video.")

    fetched = transcript_obj.fetch()
    return " ".join(chunk.text for chunk in fetched)



def create_retriever(video_id: str):
    os.makedirs("vectors", exist_ok=True)

    # Sanitize video_id for file path
    safe_video_id = "".join(c for c in video_id if c.isalnum() or c in ("-", "_"))
    index_path = os.path.join("vectors", f"faiss_index_{safe_video_id}")

    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Fetching transcript...")
        transcript = fetch_transcript(video_id)

        print("Creating chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript])

        print("Creating FAISS index...")
        vector_store = FAISS.from_documents(
            chunks,
            embedding=embeddings
        )
        vector_store.save_local(index_path)
        print("FAISS index saved.")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20
        }
    )
    return retriever


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)