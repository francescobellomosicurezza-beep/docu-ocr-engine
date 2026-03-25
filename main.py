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
                "content_type": file.content_type,
                "size_bytes": len(content),
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
