import os
import PyPDF2

directory_path = 'reports'

total_pages = 0
for filename in os.listdir(directory_path):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(directory_path, filename)
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            num_pages = len(pdf.pages)
            total_pages += num_pages
            print(f"{filename}: {num_pages} pages")

print(f"\nTotal pages in all PDF reports: {total_pages}")
