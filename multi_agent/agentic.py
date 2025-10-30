from dotenv import load_dotenv
import logging
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# =============================
# 1️⃣ Environment & Logging Setup
# =============================
load_dotenv()

logging.basicConfig(
   level=logging.INFO,
   filename="app.log",
   filemode="w",
   format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("🔧 System initialized... Environment and logging setup complete.")

# =============================
# 2️⃣ Load Google Gemini LLM
# =============================
try:
   llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', transport='rest')
   logging.info("✅ LLM initialized successfully (Gemini 2.0 Flash).")
except Exception as e:
   logging.error(f"❌ Failed to initialize LLM: {e}")
   raise e

# =============================
# 3️⃣ Agent 1: Research Agent (Data Extraction)
# =============================
logging.info("⚙️ Initializing Research Agent (SerpAPI)...")

search = SerpAPIWrapper()
search_tool = Tool(
   name="SerpAPI Search",
   func=search.run,
   description="Useful for fetching current news, articles, and reports from the web."
)

research_agent = initialize_agent(
   tools=[search_tool],
   llm=llm,
   agent="zero-shot-react-description",
   verbose=True
)
logging.info("✅ Research Agent initialized successfully.")

# =============================
# 4️⃣ Agent 2: Clean & Summarize Agent
# =============================
logging.info("⚙️ Initializing Clean & Summarize Agent...")

def clean_and_summarize(text: str) -> str:
   """Performs text cleaning and summarization in 3 concise lines."""
   logging.info("🧹 [Clean & Summarize Agent] Cleaning and summarizing text...")
   prompt = f"Clean the text for readability and summarize it in 5 lines:\n\n{text}"
   try:
      summary = llm.predict(prompt)
      logging.info("✅ [Clean & Summarize Agent] Summarization complete.")
      return summary
   except Exception as e:
      logging.error(f"❌ [Clean & Summarize Agent] Failed: {e}")
      return "Error during summarization."

clean_summarize_tool = Tool(
   name="Clean & Summarize",
   func=clean_and_summarize,
   description="Cleans and summarizes input text into 3 lines."
)

clean_summarize_agent = initialize_agent(
   tools=[clean_summarize_tool],
   llm=llm,
   agent="zero-shot-react-description",
   verbose=True
)
logging.info("✅ Clean & Summarize Agent initialized successfully.")

# =============================
# 5️⃣ Agent 3: Writer Agent (File Storage)
# =============================
logging.info("⚙️ Initializing Writer Agent...")

def save_to_file(text: str) -> str:
   """Writes summarized text to file and logs the process, with line breaks for readability."""
   file_path = "summary.txt"
   try:
      # --- Clean formatting before writing ---
      formatted_text = text.replace("•", "\n•").replace(" - ", "\n- ").replace(". ", ".\n")
      formatted_text = formatted_text.strip()

      with open(file_path, "w", encoding="utf-8") as f:
         f.write(formatted_text)

      logging.info(f"✅ [Writer Agent] Summary successfully written to {file_path}")
      return f"Summary saved to {file_path}"
   except Exception as e:
      logging.error(f"❌ [Writer Agent] Failed to write file: {e}")
      return "File write operation failed."

writer_tool = Tool(
   name="File Writer",
   func=save_to_file,
   description="Writes the given text into summary.txt (overrides each run)."
)

writer_agent = initialize_agent(
   tools=[writer_tool],
   llm=llm,
   agent="zero-shot-react-description",
   verbose=True
)
logging.info("✅ Writer Agent initialized successfully.")

# =============================
# 6️⃣ Supervisor Agent (Coordinator)
# =============================
def supervisor_task(prompt):
   """Coordinates the Research, Clean&Summarize, and Writer Agents."""
   logging.info("🧠 [Supervisor] Workflow started.")
   logging.info(f"[Supervisor] Received user prompt: {prompt}")

   # --- Step 1: Research Phase ---
   logging.info("[Supervisor] Sending task to Research Agent...")
   news = research_agent.run(prompt)
   logging.info(f"[Supervisor] Research Agent completed. Raw data collected: {news[:300]}...")

   # --- Step 2: Clean & Summarize Phase ---
   logging.info("[Supervisor] Sending data to Clean & Summarize Agent...")
   summary = clean_summarize_agent.run(f"Summarize this: {news}")
   logging.info(f"[Supervisor] Clean & Summarize Agent completed. Summary: {summary}")

   # --- Step 3: Writing Phase ---
   logging.info("[Supervisor] Sending summary to Writer Agent...")
   result = writer_agent.run(summary)
   logging.info(f"[Supervisor] Writer Agent completed. File saved: {result}")

   logging.info("🏁 [Supervisor] Workflow finished successfully.")
   return result

# =============================
# 7️⃣ Main Execution
# =============================
if __name__ == "__main__":
   logging.info("🚀 Multi-Agent System started.")
   prompt = input("You: ")
   output = supervisor_task(prompt)
   print(output)
   logging.info("✅ Execution completed without errors.")
