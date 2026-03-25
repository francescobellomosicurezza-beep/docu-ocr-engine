from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF
import tempfile

app = FastAPI()

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Docu OCR Engine online"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()

    text = ""

    # salva temporaneamente il file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # apri PDF e leggi testo
    try:
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text()
    except:
        text = "Errore lettura PDF"

    return {
        "results": [
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "testo_estratto": text[:1000],  # primi 1000 caratteri
                "categoria": "Da determinare",
                "nome": "",
                "cognome": "",
                "corso": "",
                "tipo_percorso": "",
                "data_conclusione": "",
                "data_scadenza": "",
                "confidenza": "bassa"
            }
        ]
    }
