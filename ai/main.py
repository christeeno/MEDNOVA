import google.generativeai as genai
import json, re, os

def extract_patient_pdf_data(pdf_path, api_key, model="gemini-2.5-flash"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    genai.configure(api_key=api_key)
    file = genai.upload_file(path=pdf_path)
    response = genai.GenerativeModel(model).generate_content([
        """You are a clinical assistant. Extract patient details from this PDF 
        into structured JSON with fields: Patient Name, Age, Gender, Diagnosis,
        Medications, Lab Results, Imaging Findings, Doctor Notes, 
        Follow-up Recommendations. Summarize any images like X-rays in 'Imaging Findings'.""",
        file
    ])

    text = re.sub(r"^```json\s*|\s*```$", "", response.text.strip())
    return json.loads(text)

try:
    data = extract_patient_pdf_data("sample.pdf", api_key="AIzaSyA7btJ9Ee4Zvlwl1s9prb-2GOgtKNlfM8g")
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")
