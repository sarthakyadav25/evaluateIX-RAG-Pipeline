from pypdf import PdfReader
import docx
import io

def extract_text_from_bytes(content: bytes, file_ext: str) -> str:
    """Extracts text from binary content based on file extension."""
    text_content = ""
    file_stream = io.BytesIO(content)

    if 'pdf' in file_ext:
        try:
            reader = PdfReader(file_stream)
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")
            
    elif 'doc' in file_ext:
        try:
            doc = docx.Document(file_stream)
            for para in doc.paragraphs:
                text_content += para.text + "\n"
        except Exception as e:
            raise ValueError(f"Error parsing DOCX: {str(e)}")
            
    elif 'txt' in file_ext or 'md' in file_ext:
        text_content = content.decode('utf-8')
        
    else:
        # Placeholder for Image OCR (requires pytesseract)
        raise ValueError(f"Unsupported or OCR-dependent file type: {file_ext}")

    return text_content