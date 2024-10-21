import os
import requests
import fitz  
import pandas as pd
import json

csv_file_path = 'compiled_papers.csv'
pdf_folder = 'pdf_files'
json_file_path = 'arxiv_papers.json'
error_log_path = 'error_log.json'

os.makedirs(pdf_folder, exist_ok=True)

df = pd.read_csv(csv_file_path)
papers_data = []
error_log = []

def download_pdf(url, save_path):
    """Download a PDF file from the given URL and save it."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {save_path}")
            return True
        else:
            print(f"Failed to download: {url}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extract text from the given PDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text()  
        print(f"Text extracted from: {pdf_path}")
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def delete_pdf(pdf_path):
    """Delete the PDF file to save space after extracting the text."""
    try:
        os.remove(pdf_path)
        print(f"Deleted PDF: {pdf_path}")
    except Exception as e:
        print(f"Error deleting {pdf_path}: {e}")


for index, row in df.iterrows():
    print(index, "/92780") 
    arxiv_id = row['arxiv_abstract_url'].split('/')[-1] 
    pdf_url = row['arxiv_pdf_url']
    pdf_filename = os.path.join(pdf_folder, f"{arxiv_id}.pdf")
    
    if download_pdf(pdf_url, pdf_filename):
        text = extract_text_from_pdf(pdf_filename)
        
        if text is not None:
            delete_pdf(pdf_filename)
            
            paper_details = {
                'arxiv_id': arxiv_id,
                'title': row['title'],
                'authors': row['authors'],
                'summary': row['summary'],
                'published': row['published'],
                'text': text
            }
            papers_data.append(paper_details)
        else:
            error_log.append({
                'arxiv_id': arxiv_id,
                'pdf_url': pdf_url,
                'error': 'Text extraction failed'
            })
    else:
        error_log.append({
            'arxiv_id': arxiv_id,
            'pdf_url': pdf_url,
            'error': 'Download failed'
        })

with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(papers_data, json_file, ensure_ascii=False, indent=4)

with open(error_log_path, 'w', encoding='utf-8') as error_file:
    json.dump(error_log, error_file, ensure_ascii=False, indent=4)

print(f"All paper data has been saved to {json_file_path}.")
print(f"Error log has been saved to {error_log_path}.")