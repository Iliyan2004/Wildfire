"""Renders an HTML document to PDF with WeasyPrint.   ~/flood/venv/bin/python tools/render_pdf.py docs/X.html [docs/X.pdf]"""
import sys
import weasyprint
src = sys.argv[1]; dst = sys.argv[2] if len(sys.argv) > 2 else src[:-5] + ".pdf"
weasyprint.HTML(src).write_pdf(dst)
print("rendered", dst)
