"""
Convertisseur PDF → Markdown pour les documents juridiques français.
Supporte les PDFs textuels (PyMuPDF4LLM) et les PDFs scannés (Docling OCR).

Lancer avec :
    python -m app.utils.pdf_converter
"""

import re
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def clean_and_format_french_law(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(
        r'(?m)^[ \t]*["\-\*\.\'\_ ]+[ \t]*(?=(?:Livre|Partie|Titre|Chapitre|'
        r'Section|Sous-section|Paragraphe|Article))',
        '', text,
    )
    text = re.sub(r'(?m)^[ \t]*(Livre\b.*)$', r'# \1', text)
    text = re.sub(r'(?m)^[ \t]*(Partie\b.*)$', r'## \1', text)
    text = re.sub(r'(?m)^[ \t]*(Titre\b.*)$', r'### \1', text)
    text = re.sub(
        r'(?m)^[ \t]*(Chapitre|Section|Sous-section|Paragraphe)\b(.*)$',
        r'#### \1\2', text,
    )
    text = re.sub(r'(?m)^[ \t]*(Article\b.*)$', r'##### \1', text)
    text = re.sub(r'([a-z,;])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def convert_pdfs_pymupdf4llm(pdf_folder: str, output_folder: str) -> None:
    import pymupdf4llm

    pdf_path = Path(pdf_folder)
    pdf_files = list(pdf_path.glob("*.pdf"))
    logger.info(f"{len(pdf_files)} PDFs textuels trouvés dans {pdf_folder}")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        try:
            md_text = pymupdf4llm.to_markdown(
                str(pdf_file),
                margins=0,
                show_images=False,
                force_markdown=False,
                header=False,
                footer=False,
                page_separators=False,
                ignore_images=True,
                write_images=False,
            )
            plain_text = re.sub(r'(?m)^#+\s*', '', md_text)
            plain_text = re.sub(r'[*_]{1,}', '', plain_text)
            cleaned_md = clean_and_format_french_law(plain_text)

            output_file = output_path / f"{pdf_file.stem}.md"
            output_file.write_text(cleaned_md, encoding="utf-8")
            logger.info(f"Converti (PyMuPDF) : {pdf_file.name}")
        except Exception as e:
            logger.error(f"Erreur lors de la conversion de {pdf_file.name} : {e}")


def convert_pdfs_docling(pdf_folder: str, output_folder: str) -> None:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
        AcceleratorOptions,
        AcceleratorDevice,
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["fr", "en"])
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.MPS
    )
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_picture_images = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    logger.info(f"{len(pdf_files)} PDFs scannés trouvés dans {pdf_folder}")

    for pdf_file in pdf_files:
        try:
            result = converter.convert(str(pdf_file))
            plain_text = result.document.export_to_text()
            cleaned_md = clean_and_format_french_law(plain_text)

            output_file = output_path / f"{pdf_file.stem}.md"
            output_file.write_text(cleaned_md, encoding="utf-8")
            logger.info(f"Converti (Docling OCR) : {pdf_file.name}")
        except Exception as e:
            logger.error(f"Erreur lors de la conversion de {pdf_file.name} : {e}")


if __name__ == "__main__":
    raw_dir = settings.DATA_RAW_DIR

    convert_pdfs_pymupdf4llm(
        str(raw_dir / "text" / "codes"),
        str(settings.DATA_PROCESSED_DIR / "codes"),
    )
    # convert_pdfs_docling(
    #     str(raw_dir / "scan"),
    #     str(settings.DATA_PROCESSED_DIR),
    # )

    logger.info("Conversion PDF terminée.")
