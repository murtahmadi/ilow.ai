import base64
import io
import json
import os
import re

import anthropic
from pypdf import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="iLow Claims API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"```json\n?", "", text)
    cleaned = re.sub(r"```\n?", "", cleaned)
    return json.loads(cleaned.strip())


async def read_pdf_text(file: UploadFile) -> str:
    data = await file.read()
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)[:10000]
    except Exception:
        return data.decode("utf-8", errors="ignore")[:10000]


@app.post("/api/analyze-policy")
async def analyze_policy(
    policy_pdf: UploadFile = File(...),
    claim_summary: str = Form(...),
):
    policy_text = await read_pdf_text(policy_pdf)

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are an expert insurance claims analyst. Always respond with valid JSON only — no markdown, no explanation outside the JSON.",
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this insurance policy against the submitted claim summary.

INSURANCE POLICY:
{policy_text}

CLAIM SUMMARY:
{claim_summary}

Return a JSON object with these exact keys:
{{
  "covered_items": [{{"item": "string", "citation": "exact policy language"}}],
  "not_covered_items": [{{"item": "string", "reason": "string", "citation": "exact policy language"}}],
  "determination": "Likely Covered | Partially Covered | Likely Not Covered",
  "key_clauses": ["relevant section text"],
  "confidence_score": 0-100,
  "summary": "2-3 sentence plain-English summary"
}}""",
            }
        ],
    )

    try:
        data = extract_json(msg.content[0].text)
    except Exception:
        data = {"raw": msg.content[0].text}

    return data


@app.post("/api/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_b64 = base64.standard_b64encode(image_bytes).decode()
    media_type = image.content_type or "image/jpeg"
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if media_type not in allowed:
        raise HTTPException(400, "Unsupported image type. Upload JPEG, PNG, GIF, or WebP.")

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1536,
        system="You are an expert insurance property damage assessor. Always respond with valid JSON only — no markdown, no explanation outside the JSON.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Assess the property/vehicle damage visible in this image for an insurance claim.

Return a JSON object with these exact keys:
{
  "damage_type": "primary damage type (e.g. Hail, Water Intrusion, Fire, Collision, Wind)",
  "severity": "Minor | Moderate | Severe | Total Loss",
  "severity_explanation": "why this severity rating was assigned",
  "affected_areas": ["list of specific damaged components/areas"],
  "materials": [{"component": "string", "cost_tier": "Low | Medium | High", "notes": "string"}],
  "estimated_scope": "brief scope of repair needed",
  "red_flags": ["any anomalies worth investigating — empty array if none"],
  "confidence_score": 0-100,
  "summary": "2-3 sentence plain-English assessment"
}""",
                    },
                ],
            }
        ],
    )

    try:
        data = extract_json(msg.content[0].text)
    except Exception:
        data = {"raw": msg.content[0].text}

    return data


class FraudClaimData(BaseModel):
    claimant_name: str = ""
    incident_date: str = ""
    claim_date: str = ""
    zip_code: str = ""
    claim_type: str = ""
    claimed_amount: str = ""
    description: str = ""
    prior_claims: str = "0"
    policy_age_months: str = ""


@app.post("/api/analyze-fraud")
async def analyze_fraud(claim: FraudClaimData):
    claim_text = f"""
Claimant Name: {claim.claimant_name}
Incident Date: {claim.incident_date}
Claim Filed Date: {claim.claim_date}
Incident ZIP Code: {claim.zip_code}
Claim Type: {claim.claim_type}
Claimed Amount: ${claim.claimed_amount}
Policy Age: {claim.policy_age_months} months
Prior Claims (3 years): {claim.prior_claims}
Incident Description: {claim.description}
""".strip()

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1536,
        system="You are an expert insurance fraud detection specialist. Always respond with valid JSON only — no markdown, no explanation outside the JSON.",
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this insurance claim for potential fraud indicators.

CLAIM DETAILS:
{claim_text}

Evaluate for: new-policy claims, high-frequency claimants, date inconsistencies, high-risk zip codes, inflated amounts, vague descriptions, suspicious claim patterns.

Return a JSON object with these exact keys:
{{
  "risk_level": "Low | Medium | High | Very High",
  "risk_score": 0-100,
  "red_flags": [{{"flag": "short title", "explanation": "why this is concerning", "severity": "Low | Medium | High"}}],
  "recommended_action": "Auto-Approve | Manual Review | Enhanced Investigation | Deny",
  "investigation_notes": "specific things to verify or follow up on",
  "summary": "2-3 sentence plain-English risk summary"
}}""",
            }
        ],
    )

    try:
        data = extract_json(msg.content[0].text)
    except Exception:
        data = {"raw": msg.content[0].text}

    return data


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
