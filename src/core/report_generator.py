import os
import markdown
import logging

logger = logging.getLogger(__name__)

def write_markdown_and_html_from_markdown(md: str, output_dir: str = ".") -> tuple[str, str]:
    """
    Takes a Markdown string `md`, writes it to report.md in `output_dir`,
    converts it to HTML, writes report.html, and returns (md_path, html_path).
    """
    # Ensure the output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # 1) Write Markdown
    md_path = os.path.join(output_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"✅ Wrote Markdown report to {md_path}")

    # 2) Convert to HTML
    html_body = markdown.markdown(md, extensions=["fenced_code", "tables"])
    full_html = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <title>Data Analysis Report</title>\n"
        "  <style>\n"
        "    body { font-family: sans-serif; max-width:800px; margin:2em auto; line-height:1.6; }\n"
        "    pre  { background:#f4f4f4; padding:1em; overflow-x:auto; }\n"
        "    code { background:#eee; padding:0.2em 0.4em; }\n"
        "    table{ border-collapse:collapse; width:100%; margin:1em 0; }\n"
        "    th, td { border:1px solid #ccc; padding:0.5em; text-align:left; }\n"
        "    h1,h2,h3 { margin-top:1.5em; }\n"
        "  </style>\n"
        "</head>\n<body>\n"
        f"{html_body}\n"
        "</body>\n</html>"
    )

    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info(f"✅ Wrote HTML report to {html_path}")

    return md_path, html_path
