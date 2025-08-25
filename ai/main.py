import google.generativeai as genai
import json, re, os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def extract_patient_pdf_data(pdf_path, api_key, model="gemini-2.5-flash"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Configure Gemini with API key
    genai.configure(api_key=api_key)

    # Upload the PDF file
    file = genai.upload_file(path=pdf_path)

    # Generate structured content
    response = genai.GenerativeModel(model).generate_content([
        """You are a clinical assistant. Extract patient details from this PDF 
        into structured JSON with fields: Patient Name, Age, Gender, Diagnosis,
        Medications, Lab Results, Imaging Findings, Doctor Notes, 
        Follow-up Recommendations. Summarize any images like X-rays in 'Imaging Findings'.""",
        file
    ])

    # Clean and parse the JSON response
    text = re.sub(r"^```json\s*|\s*```$", "", response.text.strip())
    return json.loads(text)

# Run the function
try:
    data = extract_patient_pdf_data("sample.pdf", api_key=GEMINI_API_KEY)
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")