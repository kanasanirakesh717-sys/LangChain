import os

import fitz

import re

import warnings

from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI



 

warnings.filterwarnings("ignore")


 

# =========================

# 1. Load environment

# =========================

load_dotenv()



import os
import fitz
import re
import warnings
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================
# 1. Load environment
# =========================
load_dotenv()

# =========================
# 2. Text Cleaning
# =========================
def clean_text(text):
    text = re.sub(r'\|.*\|', '', text)  # Remove table-like structures
    text = re.sub(r'\d+\.', '', text)   # Remove numbered lists/page numbers
    text = re.sub(r'[\r\n]+', ' ', text)  # Remove newlines
    return text.strip()

# =========================
# 3. PDF Extraction
# =========================
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))

# =========================
# 4. Summarization with LangChain + Gemini
# =========================
def summarize_with_gemini(chunks):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", transport='rest')
    summaries = []
    for chunk in chunks:
        cleaned = clean_text(chunk)
        if not cleaned:
            continue
        response = model.invoke(f"Summarize the following text in the same language within 100 words:\n\n{cleaned}")
        summaries.append(response.content)
    return " ".join(summaries)

# =========================
# 5. Main Workflow
# =========================
def process_pdf_with_langchain(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    chunks = splitter.split_text(cleaned_text)
    # Summarize with Gemini
    summary = summarize_with_gemini(chunks)
    return summary

# =========================
# 6. Entry Point
# =========================
if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ").strip()
    try:
        output = process_pdf_with_langchain(pdf_path)
        print("\n📌 Final Summary:\n", output)
    except Exception as e:
        print("Error:", e)