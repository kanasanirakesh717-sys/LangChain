import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

# Initialize the LLM (Google Gemini via LangChain)
# Ensure GOOGLE_API_KEY is set in environment (.env)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # adjust model name if needed
    temperature=0.0,
    max_output_tokens=2048,
    transport="rest"
)

# Simple function to read profile.txt and return key-value pairs as JSON
def read_profile_and_convert_to_json(file_path: str = "profile.txt") -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = dict(
                line.strip().split("=", 1)
                for line in f if "=" in line
            )
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": str(e)})

# Define the tool
profile_tool = Tool(
    name="Profile Reader",
    func=read_profile_and_convert_to_json,
    description="Reads profile.txt and converts its key-value pairs to JSON."
)

# Initialize the agent
profile_agent = initialize_agent(tools=[profile_tool],llm=llm,agent="zero-shot-react-description",verbose=True)

def validate_profile(json_data: str) -> str:
    try:
        data = json.loads(json_data)
        age = int(data.get("age", 0))
        job_type = data.get("jobType", "")
        salary = int(data.get("salary", 0))
        cibil = int(data.get("cibilscore", 0))
        valid_jobs = {"IT", "Govt", "Bank", "Doctor", "Military"}

        if not (25 < age < 50):
            return "Not eligible: Age criteria failed \n The age should be between 25 and 50."
        if job_type not in valid_jobs:
            return "Not eligible: Job type criteria failed \n The job type should be one of the following:IT,Govt,Bank,Doctor,Military."
        if salary < 450000:
            return "Not eligible: Salary criteria failed \n The salary should be at least 450000    ."
        if cibil < 700:
            return "Not eligible: CIBIL score criteria failed \n The CIBIL score should be at least 700."

        return "This profile is Eligible for 1cr term-insurance."
    except Exception as e:
        return f"Validation error: {e}"

validation_tool = Tool(
    name="Profile Validator",
    func=validate_profile,
    description="Validates JSON profile data for eligibility."
)

validation_agent = initialize_agent(tools=[validation_tool],llm=llm,agent="zero-shot-react-description",verbose=True)

# Agent to fetch best 1cr term plans using SerpAPI
def fetch_best_term_plans(_: str) -> str:
    try:
        serpapi = SerpAPIWrapper()
        query = "1cr term insurance plans providers list in India 2025"
        result = serpapi.results(query)
        return result
    except Exception as e:
        return f"Error fetching term plans: {e}"

term_plan_tool = Tool(
    name="Term Plan Finder",
    func=fetch_best_term_plans,
    description="You are a searching agent in internet using SerpAPI and provide useful data."
)

term_plan_agent = initialize_agent(tools=[term_plan_tool],llm=llm,agent="zero-shot-react-description",verbose=True)

# --- Email Utility ---
def send_email(to_email: str, subject: str, body: str):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("EMAIL_SENDER")
    smtp_password = os.getenv("EMAIL_PASSWORD")

    if not smtp_user or not smtp_password:
        print("email_sender and email_password must be set in your .env file.")
        return

    from_email = smtp_user
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())
            logging.info(f"Email sent to {to_email}")
    except Exception as e:
            logging.error(f"Failed to send email: {e}")

# --- Manager Agent ---
def manager():
    # Step 1: Read and convert profile.txt to JSON
    json_result = profile_agent.run("Read and convert profile.txt to JSON")
    logging.info(f"Profile JSON: {json_result}")
    profile_data = json.loads(json_result)
    to_email = profile_data.get("email")
   
    # Step 2: Validate JSON data
    validation_result = validation_agent.run(json_result)
    logging.info(f"Validation Result: {validation_result}")

    # Step 3: If eligible, fetch best term plans
    if "Eligible" in validation_result:
        plans = term_plan_agent.run("Fetch best 1cr term insurance plans provider company names ?")
        logging.info(f"Best 1cr Term Plan providers: {plans}")
        subject = "Your 1cr Term Insurance Eligibility & Best Plans"
        body = f"Congratulations! You are eligible for 1cr term insurance. Here are the best plan providers:\n\n{plans}"
    else:
        subject = "Your 1cr Term Insurance Eligibility Result"
        body = f"Sorry, you are not eligible for 1cr term insurance. Reason:\n{validation_result}"

    if to_email:
        send_email(to_email, subject, body)
    else:
        logging.warning("No email found in profile. Email not sent.")

if __name__ == "__main__":
    manager()
