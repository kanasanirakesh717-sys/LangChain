import os
import fitz  # PyMuPDF
import json
import streamlit as st
import camelot
from io import BytesIO
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
import logging

# ----------------------------------------------------------
# 🧾 Setup Logging
# ----------------------------------------------------------
LOG_FILE = "agent_logs.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 🌍 Load environment and initialize LLM
# ----------------------------------------------------------
load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', transport='rest')

# ----------------------------------------------------------
# 🧩 Agent 1: PDF Table & Image Extractor (Structured)
# ----------------------------------------------------------
def extract_pdf_tables_images(file):
    """Extract structured tables (rows/columns) and image metadata from PDF."""
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(file.read())

    extracted_data = {"tables": [], "images": []}

    # --- TABLE EXTRACTION USING CAMELOT ---
    try:
        tables = camelot.read_pdf(temp_pdf_path, pages="all")
        for i, t in enumerate(tables):
            df = t.df
            if not df.empty:
                headers = list(df.iloc[0])
                rows = df.iloc[1:].values.tolist()
                extracted_data["tables"].append({
                    "table_index": i + 1,
                    "columns": headers,
                    "rows": rows
                })
        logger.info(f"[Agent 1] Extracted {len(extracted_data['tables'])} tables successfully.")
    except Exception as e:
        logger.error(f"[Agent 1] Table extraction failed: {e}")

    # --- IMAGE EXTRACTION USING PyMuPDF ---
    try:
        doc = fitz.open(temp_pdf_path)
        for page_number, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                base_image = doc.extract_image(img[0])
                extracted_data["images"].append({
                    "page": page_number,
                    "image_type": base_image["ext"],
                    "width": base_image["width"],
                    "height": base_image["height"]
                })
        logger.info(f"[Agent 1] Extracted {len(extracted_data['images'])} images successfully.")
    except Exception as e:
        logger.error(f"[Agent 1] Image extraction failed: {e}")

    logger.info(f"[Agent 1] Final Extracted Data: {json.dumps(extracted_data, indent=2)}")
    return extracted_data

# ----------------------------------------------------------
# 🧩 Agent 2: JSON Formatter Agent (LLM)
# ----------------------------------------------------------
def format_to_json(extracted_data):
    """Convert extracted structured data into clean standardized JSON."""
    prompt = f"""
    Clean and standardize the following extracted PDF data into a JSON format.
    Maintain the following structure:
    {{
      "tables": [{{"columns": [...], "rows": [[...]]}}],
      "images": [{{"page": int, "image_type": str, "width": int, "height": int}}]
    }}

    Extracted Data:
    {json.dumps(extracted_data, indent=2)}
    """
    response = llm.invoke(prompt)
    formatted_json = response.content
    logger.info(f"[Agent 2] Final JSON formatted by Gemini:\n{formatted_json}")
    return formatted_json

# ----------------------------------------------------------
# 🧠 Two-Agent Setup
# ----------------------------------------------------------
extract_tool = Tool(
    name="PDF Table & Image Extractor",
    func=extract_pdf_tables_images,
    description="Extracts structured tables (rows/columns) and images from PDF."
)

json_tool = Tool(
    name="JSON Formatter",
    func=format_to_json,
    description="Formats extracted data into standardized clean JSON."
)

agent = initialize_agent(
    tools=[extract_tool, json_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=False
)

# ----------------------------------------------------------
# 🎨 Streamlit Frontend
# ----------------------------------------------------------
st.set_page_config(page_title="📊 PDF → JSON Converter (Structured Tables + Images)", layout="centered")
st.title("🤖 Agentic PDF → JSON Converter (Structured Tables + Images)")

uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.info("✅ PDF uploaded successfully!")

    with st.spinner("🔍 Agent 1: Extracting structured tables and images..."):
        extracted_data = extract_pdf_tables_images(uploaded_file)

    with st.spinner("⚙️ Agent 2: Formatting data into clean JSON..."):
        json_data = format_to_json(extracted_data)

    st.subheader("🧾 Extracted JSON Output:")
    try:
        st.json(json.loads(json_data))
    except:
        st.text(json_data)

    # --- Download JSON ---
    json_bytes = BytesIO(json_data.encode("utf-8"))
    st.download_button(
        label="📥 Download JSON File",
        data=json_bytes,
        file_name=f"{uploaded_file.name.split('.')[0]}_structured_tables.json",
        mime="application/json"
    )

    st.success("✅ Process completed. Check 'agent_logs.log' for detailed logs.")
