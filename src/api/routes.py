# src/api/routes.py

import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from core.primary_analysis import analyze_spreadsheet
from core.report_generator import write_markdown_and_html_from_markdown

load_dotenv()
router = APIRouter()


class AnalyzeRequest(BaseModel):
    file_path: str


class AnalyzeResponse(BaseModel):
    success: bool
    errors: list[str]
    analysis: dict
    md: str      # path to report.md
    html: str    # path to report.html


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # 1) Run the core analysis
    analysis_result = analyze_spreadsheet(req.file_path)
    if not analysis_result.get("success"):
        raise HTTPException(status_code=500, detail=analysis_result.get("errors"))
    analysis_data = analysis_result["analysis"]

    # 2) Generate the narrative via OpenAI using JSON context
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    json_context = "```json\n" + json.dumps(analysis_data, indent=2) + "\n```"
    system_prompt = (
        "You are a data storyteller for car shoppers. "
        "Given the JSON analysis above, write only the 'Customer-Centric Highlights' section "
        "in friendly, actionable language. Do NOT include any other sections."
    )
    resp = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json_context},
        ],
        temperature=0.7,
    )
    narrative_md = resp.choices[0].message.content.strip()

    # 3) Load Jinja2 template and render the full report
    try:
        template_dir = os.path.dirname(__file__)
        env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        template = env.get_template("report_skeleton.md")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template loading failed: {e}")

    # 4) Render and write output
    full_md = template.render(analysis=analysis_data, narrative=narrative_md)
    output_dir = os.getenv("REPORT_OUTPUT_DIR", ".")
    md_path, html_path = write_markdown_and_html_from_markdown(full_md, output_dir)

    return AnalyzeResponse(
        success=analysis_result["success"],
        errors=analysis_result["errors"],
        analysis=analysis_data,
        md=md_path,
        html=html_path
    )


# src/api/app.py

from fastapi import FastAPI
from api.routes import router

app = FastAPI(debug=True, title="Data Analysis API")
app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
