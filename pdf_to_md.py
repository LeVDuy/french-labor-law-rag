import re
import pymupdf4llm
import pathlib
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, AcceleratorOptions, AcceleratorDevice, TesseractOcrOptions


def clean_and_format_french_law(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(?m)^[ \t]*["\-\*\.\'\_]+[ \t]*(?=(?:Livre|Partie|Titre|Chapitre|Section|Sous-section|Paragraphe|Article))', '', text)
    text = re.sub(r'(?m)^[ \t]*(Livre\b.*)$', r'# \1', text)                                       
    text = re.sub(r'(?m)^[ \t]*(Partie\b.*)$', r'## \1', text)                                   
    text = re.sub(r'(?m)^[ \t]*(Titre\b.*)$', r'### \1', text)                                  
    text = re.sub(r'(?m)^[ \t]*(Chapitre|Section|Sous-section|Paragraphe)\b(.*)$', r'#### \1\2', text)
    text = re.sub(r'(?m)^[ \t]*(Article\b.*)$', r'##### \1', text)                     
    text = re.sub(r'([a-z,;])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# PDF TEXT (PyMuPDF4LLM)
def convert_pdfs_pymupdf4llm(pdf_folder: str, output_folder: str):
    pdf_path = pathlib.Path(pdf_folder)
    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"Trouvé {len(pdf_files)} fichiers PDF texte dans {pdf_folder}")
    output_path = pathlib.Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for pdf_file in pdf_files:
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_file), margins=0, show_images=False, force_markdown=False, header=False, footer=False, page_separators=False, ignore_images=True, write_images=False)
            plain_text = re.sub(r'(?m)^#+\s*', '', md_text)
            plain_text = re.sub(r'[*_]{1,}', '', plain_text)
            cleaned_md = clean_and_format_french_law(plain_text)
            output_file = output_path / f"{pdf_file.stem}.md"
            output_file.write_text(cleaned_md, encoding='utf-8')
            print(f"PyMuPDF4LLM: {pdf_file.name}")
        except Exception as e:
            print(f"Erreur PyMuPDF4LLM {pdf_file.name}: {e}")

# 2.PDF SCAN (Docling)
def convert_pdfs_docling(pdf_folder: str, output_folder: str):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = True 
    pipeline_options.ocr_options = EasyOcrOptions(lang=["fr", "en"])
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)
    pipeline_options.images_scale = 1.0 
    pipeline_options.generate_picture_images = False
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    output_path = pathlib.Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    print(f"Trouvé {len(pdf_files)} fichiers PDF scannés dans {pdf_folder}")
    
    for pdf_file in pdf_files:
        try:
            result = converter.convert(str(pdf_file))
            plain_text = result.document.export_to_text()
            cleaned_md = clean_and_format_french_law(plain_text)           
            output_file = output_path / f"{pdf_file.stem}.md"
            output_file.write_text(cleaned_md, encoding='utf-8')
            print(f"Docling: {pdf_file.name}")
        except Exception as e:
            print(f"Erreur Docling {pdf_file.name}: {e}")

# CONVERT PDF TO MARKDOWN
if __name__ == "__main__":
    convert_pdfs_pymupdf4llm("data/pdf_files/text/codes", "data/markdown_files/codes")
    # convert_pdfs_docling("data/pdf_files/scan", "data/markdown_files")
    
    print("\nDone!")