import PyPDF2
import sys

def extract_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"DEBUG: Total Pages = {len(reader.pages)}")
            full_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += f"--- Page {i+1} ---\n{page_text}\n"
                else:
                    full_text += f"--- Page {i+1} ---\n[No text extracted]\n"
            
            with open("extracted_pdf.txt", "w", encoding="utf-8") as out:
                out.write(full_text)
            print("DEBUG: Wrote to extracted_pdf.txt")
            return "DONE"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(extract_text(sys.argv[1]))
    else:
        print("Usage: python read_pdf.py <path_to_pdf>")
