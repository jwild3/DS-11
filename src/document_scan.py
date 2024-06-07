from llama_index.readers.smart_pdf_loader import SmartPDFLoader


def scan_pdf():
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_url = "../data/pdf/PRD_DOC_BUW_669950-00001__SEN__AIN__V5.pdf"  # also allowed is a file path e.g. /home/downloads/xyz.pdf
    pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
    documents = pdf_loader.load_data(pdf_url)
    # todo enhance the document scan and enhance the meta data
    return documents
