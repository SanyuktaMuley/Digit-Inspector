"""Produce official one-PDF submission: two live URLs, real BITS screenshot, README.

Usage (preview, NOT SUBMITTABLE): python make_submission_pdf.py --draft
Usage (final): python make_submission_pdf.py --name ... --bits-id ...
                --github https://github.com/... --app https://....streamlit.app
                --screenshot evidence/bits_lab_execution.png --output ...pdf

Build requirements: pip install -r requirements-report.txt (not needed on Cloud).
"""
from __future__ import annotations
import argparse
from html import escape
from pathlib import Path
from urllib.parse import urlparse

import mistune
from PIL import Image
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent
PLACEHOLDERS = (
    'REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL',
    'REPLACE_WITH_YOUR_STREAMLIT_APP_URL',
)

CSS = """
@page {size:A4; margin:17mm 17mm 19mm 17mm;
       @bottom-right {content: 'PAGE ' counter(page); font-size:8pt; color:#64748b;}}
* {box-sizing:border-box}
body {font-family:'DejaVu Sans',Arial,sans-serif;color:#1f324a;font-size:9pt;line-height:1.39}
h1,h2,h3,h4 {font-weight:750;line-height:1.22; color:#17364a;break-after:avoid}
h1 {font-size:20pt; margin:0 0 10pt}h2 {font-size:14pt; margin:14pt 0 6pt;border-bottom:1px solid #d5e4e8;padding-bottom:4pt}
h3 {font-size:11pt; margin:12pt 0 6pt}
p {margin:5pt 0 8pt}a {color:#047B83;text-decoration:underline;overflow-wrap:anywhere}
table {border-collapse:collapse;width:100%;margin:9pt 0;table-layout:auto;font-size:7.65pt}
th,td {border:1px solid #cfdae3;padding:6pt 6pt;vertical-align:top;overflow-wrap:anywhere}
th {background:#e8f4f5;color:#193c55;font-weight:700;text-align:left}
tr {break-inside:avoid}thead {display:table-header-group}
code {font-family:'DejaVu Sans Mono',monospace;background:#eef3f7;padding:1pt 2pt;font-size:7.8pt;overflow-wrap:anywhere}
pre {background:#edf4f7;color:#19374d;border-radius:6px;padding:10pt;white-space:pre-wrap;word-wrap:break-word; font-size:7.4pt}
pre code {background:transparent;padding:0}
blockquote {padding:7pt 13pt;border-left:3pt solid #0f878c;background:#f2f8f8;margin:10pt 0}
li {margin:3pt 0}strong {color:#15394c}
.cover {page-break-after:always}
.kicker {text-transform:uppercase;letter-spacing:1.3px;color:#047b83;font-size:9pt;font-weight:700;margin-top:4pt}
.cover-title {font-size:23pt;color:#183447;margin-top:8pt}
.instruction {background:#e8f4f4;border-left:4pt solid #087f85;padding:10pt 12pt;margin:14pt 0}
.warning {color:#9b352d;background:#fff0ed;border:1pt solid #df9d9a;padding:11pt;margin:12pt 0;font-weight:bold}
.detail {font-size:10pt;padding:9pt 0;border-bottom:1pt solid #dee7ea}
.screenshot {border:1pt solid #b9cad3;background:white;padding:4pt;margin:11pt 0;text-align:center}
.screenshot img {max-width:100%; max-height:145mm; object-fit:contain}
.placeholder {height:85mm;border:2pt dashed #9fbac6;border-radius:8pt;display:flex;align-items:center;justify-content:center;color:#667e8d;font-weight:bold}
.label {font-size:10pt;font-weight:bold;color:#133e54;margin-top:15pt}
.readme-title {font-size:18pt;color:#17364a}
hr {border:0;border-top:1pt solid #d8e3e9;margin:14pt 0}
"""


def valid_public_url(url, host):
    p = urlparse(url)
    return p.scheme == "https" and bool(p.netloc) and (p.hostname == host or p.hostname.endswith('.'+host)) and '/YOUR_' not in url and 'PLACEHOLDER' not in url


def main():
    parser = argparse.ArgumentParser(description='Create the faculty-compliant single-PDF ML submission.')
    parser.add_argument('--draft', action='store_true', help='Generate an explicitly non-submittable PDF preview.')
    parser.add_argument('--name', default='[YOUR FULL NAME]')
    parser.add_argument('--bits-id', default='[YOUR BITS ID]')
    parser.add_argument('--github')
    parser.add_argument('--app')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.draft:
        if not (args.github and valid_public_url(args.github, 'github.com')):
            parser.error('--github must be your real https://github.com/<user>/<repo> URL.')
        if len(urlparse(args.github).path.strip('/').split('/')) < 2:
            parser.error('--github must include both owner and repository name.')
        if not (args.app and valid_public_url(args.app, 'streamlit.app')):
            parser.error('--app must be your actual https://...streamlit.app URL.')
        if not args.screenshot or not args.screenshot.is_file():
            parser.error('A real BITS Lab screenshot file is mandatory (--screenshot).')
        if '[' in args.name or '[' in args.bits_id or not args.name.strip() or not args.bits_id.strip():
            parser.error('Provide your actual --name and --bits-id.')
        try:
            with Image.open(args.screenshot) as im:
                im.verify()
        except Exception as e:
            parser.error(f'--screenshot is not a valid PNG/JPEG image: {e}')
    github = args.github or '[ADD YOUR VERIFIED GITHUB URL]'
    app = args.app or '[ADD YOUR VERIFIED STREAMLIT URL]'
    image_path = (args.screenshot.resolve().as_uri() if args.screenshot and args.screenshot.is_file() else None)
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    if args.github:
        readme = readme.replace('[REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL]', args.github)
    if args.app:
        readme = readme.replace('[REPLACE_WITH_YOUR_STREAMLIT_APP_URL]', args.app)
    if not args.draft:
        # The generated report's README must have real author and links, regardless
        # of whether README.md has yet been edited locally.
        readme = readme.replace('[YOUR NAME]', args.name).replace('[YOUR BITS ID]', args.bits_id)
        if any(marker in readme for marker in PLACEHOLDERS):
            parser.error('Some repository/app link placeholders remain in the README.')
    md = mistune.create_markdown(plugins=['table', 'url'])
    readme_html = md(readme)
    pdf_name = args.output or Path('ML_Assignment2_PREVIEW_NOT_FOR_SUBMISSION.pdf' if args.draft else 'ML_Assignment2_FINAL.pdf')
    if not pdf_name.is_absolute():
        pdf_name = ROOT / pdf_name
    pdf_name.parent.mkdir(exist_ok=True, parents=True)
    link_github = f'<a href="{escape(github,quote=True)}">{escape(github)}</a>' if args.github else escape(github)
    link_app = f'<a href="{escape(app,quote=True)}">{escape(app)}</a>' if args.app else escape(app)
    proof = f'<div class="screenshot"><img src="{escape(image_path,quote=True)}" alt="Genuine screenshot of assignment running on BITS lab"></div>' if image_path else '<div class="placeholder">REQUIRED: Insert genuine BITS Virtual Lab screenshot</div>'
    warning = '<div class="warning">INCOMPLETE DRAFT — NOT A VALID SUBMISSION. Add actual GitHub and Streamlit URLs and your genuine BITS Virtual Lab screenshot.</div>' if args.draft else ''
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<section class="cover"><p class="kicker">BITS PILANI · M.TECH AIML · MACHINE LEARNING</p>
<h1 class="cover-title">Assignment 2 — Submission Report</h1>
<p style="font-size:11pt">Handwritten Digit Classification &nbsp; | &nbsp; INK / Digit Inspector</p>
{warning}
<p class="detail"><strong>Student name:</strong> {escape(args.name)} &nbsp;&nbsp;&nbsp; <strong>BITS ID:</strong> {escape(args.bits_id)}</p>
<div class="instruction">Submission contents follow the faculty's mandatory order. All claims about live links and screenshot evidence must be personally verified by the student.</div>
<h2>1. GitHub Repository Link</h2><p class="detail">{link_github}</p>
<h2>2. Live Streamlit App Link</h2><p class="detail">{link_app}</p>
<h2>3. Screenshot: execution on BITS Virtual Lab</h2>
{proof}
<p style="font-size:8pt">This screenshot is to be captured from the student's real Virtual Lab session and is not generated by the project.</p>
</section>
<section><p class="kicker">4. Complete README.md content</p>
<h1 class="readme-title">README / Experiment Report</h1>
{readme_html}
</section></body></html>'''
    HTML(string=html, base_url=str(ROOT)).write_pdf(str(pdf_name))
    print(f'PDF created: {pdf_name} ({pdf_name.stat().st_size / 1024:.1f} KiB)')
    if args.draft:
        print('PREVIEW ONLY: this document is NOT ready for faculty submission.')
    else:
        print('Final file built; open, inspect links/screenshot/README, then verify the portal submission.')


if __name__ == '__main__':
    main()
