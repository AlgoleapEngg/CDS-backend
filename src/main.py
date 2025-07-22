import os
import json
import logging
from dotenv import load_dotenv
from jinja2 import Template
from core.primary_analysis import analyze_spreadsheet
from openai import OpenAI

import markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(file_path):
    # 1. load .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("MODEL", "gpt-4o")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in .env")
        return

    # 2. prepare OpenAI client
    client = OpenAI(api_key=api_key)

    # 3. run your analysis
    result = analyze_spreadsheet(file_path)
    with open("analysis_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if not result.get("success"):
        logger.error("Analysis failed: %s", result["errors"])
        return

    # 4. render template (narrative left blank)
    with open("narrative_template.md", encoding="utf-8") as f:
        template_str = f.read()
    tmpl = Template(template_str)
    partial_md = tmpl.render(analysis=result["analysis"], narrative="")

    # 5. call chat API for Insights
    system_prompt = (
        "You are a data storyteller for car shoppers. "
        "Given the Markdown report above, write only the 'Customer-Centric Highlights' section "
        "in friendly, actionable language. Do NOT modify other sections."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": partial_md}
        ],
        temperature=0.7,
    )
    narrative_md = response.choices[0].message.content.strip()

    # 6. stitch together full report.md
    header, sep, _ = partial_md.partition("## Customer-Centric Highlights")
    final_md = header + sep + "\n\n" + narrative_md + "\n"

    # 7. save final report.md
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(final_md)
    logger.info("✅ report.md generated")

    # 8. convert to HTML and save report.html
    html_body = markdown.markdown(
        final_md,
        extensions=["fenced_code", "tables"]
    )
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Car Data Analysis Report</title>
  <style>
    body {{ font-family: sans-serif; max-width: 800px; margin: 2em auto; line-height:1.6; }}
    pre {{ background:#f4f4f4; padding:1em; overflow-x:auto; }}
    code {{ background:#eee; padding:0.2em 0.4em; }}
    table {{ border-collapse: collapse; width:100%; margin:1em 0; }}
    th, td {{ border:1px solid #ccc; padding:0.5em; text-align:left; }}
    h1,h2,h3 {{ margin-top:1.5em; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info("✅ report.html generated – open this in your browser")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m main <path-to-spreadsheet>")
    else:
        main(sys.argv[1])