
from fastapi import FastAPI
from pydantic import BaseModel
from parser import extract_candidate_info
from llm_scorer import compute_fit_score

app = FastAPI(title="Smart Resume Parser API")

class ResumeInput(BaseModel):
    resume_text: str

@app.post("/parse_resume")
async def parse_resume(data: ResumeInput):
    extracted = extract_candidate_info(data.resume_text)
    
    # Calculate Fit Score using LLM
    fit_score = compute_fit_score(
        tech_stack=extracted["tech_stack"],
        years_of_experience=extracted["years_experience"]
    )

    response = {
        "candidate_name": extracted["name"],
        "tech_stack": extracted["tech_stack"],
        "years_of_experience": extracted["years_experience"],
        "fit_score": fit_score
    }

    return response
import re
from datetime import datetime

CURRENT_YEAR = datetime.now().year


def extract_name(text: str) -> str:
    # Very simple name extraction: first two words with capital letters
    match = re.match(r"([A-Z][a-z]+)\s([A-Z][a-z]+)", text.strip())
    return match.group(0) if match else "Unknown"


def extract_tech_stack(text: str):
    tech_keywords = [
        "Python", "Django", "Flask", "FastAPI", "NumPy", "Pandas",
        "TensorFlow", "PyTorch", "Machine Learning", "AI",
        "REST API", "SQL", "MongoDB", "Docker", "Kubernetes",
    ]

    stack_found = [tech for tech in tech_keywords if tech.lower() in text.lower()]
    return list(set(stack_found))


def extract_years_of_experience(text: str) -> int:
    # Match patterns like: 2019-2021, 2020 to 2022, 2018 – Present
    patterns = re.findall(r"(\d{4})\s*[-–to]+\s*(\d{4}|Present|present)", text)
    
    total_years = 0

    for start, end in patterns:
        start = int(start)
        end = CURRENT_YEAR if "present" in end.lower() else int(end)
        total_years += max(0, end - start)

    return total_years


def extract_candidate_info(text: str):
    return {
        "name": extract_name(text),
        "tech_stack": extract_tech_stack(text),
        "years_experience": extract_years_of_experience(text)
    }
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def compute_fit_score(tech_stack, years_of_experience):

    if not GEMINI_API_KEY:
        # fallback rule-based score
        score = min(10, len(tech_stack) + years_of_experience // 2)
        return score

    prompt = f"""
    You are an HR evaluator. Score this candidate (0-10) for a Senior Python Developer role.

    Tech Stack: {tech_stack}
    Years of Experience: {years_of_experience}

    Evaluate based on:
    - Python expertise
    - AI/ML skills
    - Backend/API development
    - Senior-level exposure

    Respond with a single integer between 0 and 10.
    """

    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    try:
        return int(response.text.strip())
    except:
        return 5  # fallback
