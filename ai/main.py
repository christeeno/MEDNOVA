import google.generativeai as genai
import json, re, os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def extract_patient_pdf_data(pdf_path, api_key, model="gemini-2.5-flash"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    genai.configure(api_key=api_key)
    file = genai.upload_file(path=pdf_path)

    # Step 1: Extract info from PDF
    extraction_prompt = """
    You are a clinical assistant. Extract patient details from this PDF 
    into structured JSON with fields: 
    Patient Name, Age, Gender, Symptoms, Diagnosis,
    Medications, Lab Results, Imaging Findings, Doctor Notes, 
    Follow-up Recommendations.
    If images (X-rays, scans) are present, summarize them in 'Imaging Findings'.
    """
    response = genai.GenerativeModel(model).generate_content([extraction_prompt, file])
    text = re.sub(r"^```json\s*|\s*```$", "", response.text.strip())

    try:
        patient_data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse extracted JSON:\n" + response.text)

    # Step 2: Ask user for missing details (Symptoms mandatory for diagnosis)
    required_fields = ["Patient Name", "Age", "Gender", "Symptoms"]
    for field in required_fields:
        if not patient_data.get(field):
            patient_data[field] = input(f"Please enter {field}: ")

    # Step 3: Get diagnosis & medicine suggestions
    diagnosis_prompt = f"""
    Based on the following patient details, provide:
    1. Possible diagnosis.
    2. Recommended medicines (generic names, not brands).
    3. Any further tests needed.

    Patient Data:
    {json.dumps(patient_data, indent=2)}
    """

    diag_response = genai.GenerativeModel(model).generate_content(diagnosis_prompt)

    patient_data["AI_Diagnosis_and_Treatment"] = diag_response.text.strip()

    return patient_data


# Run the pipeline
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "sample.pdf")
try:
    data = extract_patient_pdf_data(pdf_path, api_key=GEMINI_API_KEY)
    print("\n Final Patient Report with AI Diagnosis:\n")
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f" Error: {e}")
