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

# =========================
# 2. Text Cleaning
# =========================
def clean_text(text):
    """Removes unwanted characters and structures from text."""
    text = re.sub(r'\|.*\|', '', text)  # Remove table-like structures
    text = re.sub(r'\d+\.', '', text)   # Remove numbered lists/page numbers
    text = re.sub(r'[\r\n]+', ' ', text)  # Remove newlines
    return text.strip()

# =========================
# 3. PDF Extraction
# =========================
def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    doc = fitz.open(pdf_path)
    return "".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))

# =========================
# 4. Summarization with LangChain + Gemini
# =========================
def summarize_with_gemini(chunks):
    """Summarizes a list of text chunks using the Gemini model."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", transport='rest')
    summaries = []

    for chunk in chunks:
        cleaned = clean_text(chunk)
        if not cleaned:
            continue
        
        # Define the prompt for summarization
        prompt = f"Summarize the following text in the same language within 100 words, Keep it concise and informative:\n\n{cleaned}"
        
        try:
            response = model.invoke(prompt)
            summaries.append(response.content)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            # Optionally, you could append a placeholder or just skip
            # summaries.append("[Summarization failed for this chunk]")

    return " ".join(summaries)

# =========================
# 5. Main Workflow
# =========================
def process_pdf_with_langchain(pdf_path):
    """Orchestrates the PDF processing workflow."""
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    # A single clean pass after extraction
    cleaned_text = clean_text(text)
    print(f"Extracted and cleaned {len(cleaned_text)} characters.")

    # Split into chunks
    print("Splitting text into chunks...")
    splitter = CharacterTextSplitter(
        separator=" ",  # Split on spaces
        chunk_size=1000,
        chunk_overlap=30
    )
    chunks = splitter.split_text(cleaned_text)
    print(f"Created {len(chunks)} chunks.")

    # Summarize with Gemini
    print("Summarizing chunks with Gemini...")
    summary = summarize_with_gemini(chunks)
    return summary

# =========================
# 6. Script Execution
# =========================
print("PDF Summarizer Started.")
pdf_path = input("Enter PDF file path: ").strip()

# Basic check for file existence
if not os.path.isfile(pdf_path):
    print(f"Error: File not found at '{pdf_path}'")
else:
    try:
        output = process_pdf_with_langchain(pdf_path)
        print("\n" + "="*20)
        print("📌 Final Summary:")
        print("="*20 + "\n")
        print(output)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
