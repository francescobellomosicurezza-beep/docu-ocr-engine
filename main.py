from fastapi import FastAPI, UploadFile, File

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

    return {
        "results": [
            {
                "filename": file.filename,
                "categoria": "Attestato",
                "nome": "Mario",
                "cognome": "Rossi",
                "corso": "Primo Soccorso",
                "tipo_percorso": "base",
                "data_conclusione": "01/01/2022",
                "data_scadenza": "01/01/2025",
                "confidenza": "media"
            }
        ]
    }
