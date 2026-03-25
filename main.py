from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Docu OCR Engine online"}

@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        content = await file.read()

        results.append({
            "filename": file.filename,
            "categoria": "Attestato",
            "nome": "Mario",
            "cognome": "Rossi",
            "corso": "Primo Soccorso",
            "tipo_percorso": "base",
            "data_conclusione": "01/01/2022",
            "data_scadenza": "01/01/2025",
            "confidenza": "media"
        })

    return {"results": results}
