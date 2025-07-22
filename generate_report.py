import os
import json
from dotenv import load_dotenv
import openai
from jinja2 import Template

# 1. load env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4")

# 2. load analysis JSON
with open("analysis_output.json") as f:
    analysis = json.load(f)["analysis"]

# 3. load MD template
with open("narrative_template.md") as f:
    raw_template = f.read()

# 4. render the template with a placeholder for narrative
jinja_tmpl = Template(raw_template)
# leave {{ narrative }} empty for the LLM to fill
partial_md = jinja_tmpl.render(analysis=analysis, narrative="")

# 5. ask the LLM to fill in the Insights section
system_prompt = (
    "You are an expert data storyteller. "
    "Fill in the '## 5. Customer-Focused Insights' section below in customer-friendly language, "
    "using the data above. Do not alter other sections."
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": partial_md}
]

resp = openai.ChatCompletion.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
)

filled_md = resp.choices[0].message.content

# 6. write out the final report
with open("report.md", "w") as f:
    f.write(filled_md)

print("✅ report.md generated")
