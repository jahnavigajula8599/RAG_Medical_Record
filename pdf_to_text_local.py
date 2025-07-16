import re
from pathlib import Path
from io import BytesIO

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def pdf_to_text(pdf_path: str, txt_path: str | None = None) -> Path:
    """Extract text (and OCR images) from a single local PDF.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        txt_path: Optional path for the output .txt file. If omitted, the text
            file is created next to the PDF with the same stem.

    Returns:
        Path to the text file that was written.
    """

    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Default output = same folder, same name, .txt extension
    if txt_path is None:
        txt_path = pdf_path.with_suffix(".txt")
    txt_path = Path(txt_path).expanduser().resolve()

    doc = fitz.open(pdf_path)

    # Regex helpers
    paragraph_pattern = re.compile(r"\n\s*\n")  # empty line → paragraph break
    newline_pattern = re.compile(r"\n")           # single newline inside paragraph

    pages_text: list[str] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        page_number = page_num + 1  # human‑friendly numbering

        if text:
            # Break into paragraphs → clean interior newlines
            paragraphs = paragraph_pattern.split(text)
            cleaned = [newline_pattern.sub(" ", p).strip() for p in paragraphs]
            page_content = "\n\n".join(cleaned)
        else:
            # If no direct text, OCR any images on the page
            page_content = ""
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                img_ext = base["ext"].lower()

                if img_ext not in {
                    "jpg", "jpeg", "png", "bmp", "tiff", "tif",
                    "pbm", "pgm", "ppm", "webp", "gif",
                }:
                    print(f"Skipping unsupported image format {img_ext} on page {page_number}")
                    continue

                pil_img = Image.open(BytesIO(img_bytes))
                ocr_text = pytesseract.image_to_string(pil_img)
                paragraphs = paragraph_pattern.split(ocr_text)
                cleaned = [newline_pattern.sub(" ", p).strip() for p in paragraphs]
                page_content += "\n\n".join(cleaned)

        # Store page with a clear delimiter
        pages_text.append(
            f"[PAGE {page_number} START]\n{page_content}\n[PAGE {page_number} END]\n"
        )

    # Combine pages and write out
    text_content = "\n\n".join(pages_text)
    txt_path.write_text(text_content, encoding="utf-8")
    print(f"✅ Extracted text saved to: {txt_path}")
    return txt_path


if __name__ == "__main__":
    # Absolute path to your PDF (Windows‑friendly). Adjust if needed.
    pdf_path = r"C:/Users/jahna/Documents/Demo-RAG/Hyponatremia Synthetic Pos Complex.pdf"
    pdf_to_text(pdf_path)
