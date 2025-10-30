import streamlit as st
import logging, os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================
# 1️⃣ Logging Configuration
# ==============================
logging.basicConfig(
    filename="sdlc_tool.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SDLC_TOOL")

# ==============================
# 2️⃣ Environment Setup & LLM
# ==============================
try:
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", transport="rest")
    logger.info("LLM model initialized successfully.")
except Exception as e:
    logger.error(f"LLM initialization failed: {e}")
    st.error("⚠️ LLM initialization failed. Check API key or connection.")

# ==============================
# 3️⃣ Helper Function
# ==============================
def extract_text(response) -> str:
    """Extract clean text output from the LLM response."""
    if not response:
        return "(No response)"
    return getattr(response, "content", str(response)).strip()

# ==============================
# 4️⃣ Streamlit UI
# ==============================
st.title("🧠 SDLC AI Support Tool")
st.write("""
### Streamlit-based SDLC Automation
- Generate clean, production-ready code from natural language.
- Convert uploaded source files between major languages.
""")

# Task selection
task = st.radio("Choose a task:", ["Code Generation", "Code Conversion"], horizontal=True)

# ==============================
# 5️⃣ Code Generation Section
# ==============================
if task == "Code Generation":
    st.subheader("🧩 Automated Code Generation")
    st.write("Provide a natural language description to generate structured, maintainable code.")

    user_input = st.text_area("Enter your requirement:")
    lang = st.selectbox("Output language:", ["python", "javascript", "java","cpp", "c"])

    if st.button("🚀 Generate Code"):
        if not user_input.strip():
            st.warning("Please enter a requirement first.")
            logger.warning("Code generation attempted with empty input.")
        else:
            try:
                prompt = (
                    f"You are a senior software engineer. Generate {lang} code that:\n"
                    "- Follows SDLC best practices\n"
                    "- Includes exception handling and structured logging\n"
                    f"Requirement:\n{user_input}"
                )
                response = llm.invoke(prompt)
                code_output = extract_text(response)

                st.subheader(f"✅ Generated {lang.upper()} Code")
                st.code(code_output, language=lang)
                logger.info(f"Code generation successful for {lang}.")
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"Code generation failed: {e}")

# ==============================
# 6️⃣ Code Conversion Section
# ==============================
elif task == "Code Conversion":
    st.subheader("🔄 Code Conversion System")
    st.write("Upload a code file to convert it into another programming language.")

    file = st.file_uploader("Upload source file:", type=["py", "js","java", "cpp", "c"])
    target_lang = st.selectbox("Convert to:", ["python", "javascript","java", "cpp", "c"])

    if st.button("🔁 Convert Code"):
        if not file:
            st.warning("Please upload a source file.")
            logger.warning("Conversion attempted without file upload.")
        else:
            try:
                source_code = file.read().decode("utf-8")
                prompt = (
                    f"Convert the following code into {target_lang}:\n"
                    "- Maintain equivalent logic\n"
                    "- Add minimal documentation and comments\n"
                    "- Follow target language best practices\n\n"
                    f"Source Code:\n{source_code}"
                )
                response = llm.invoke(prompt)
                converted_code = extract_text(response)

                st.subheader(f"✅ Converted Code ({target_lang.upper()})")
                st.code(converted_code, language=target_lang)
                logger.info(f"Code conversion to {target_lang} successful.")
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"Code conversion failed: {e}")
