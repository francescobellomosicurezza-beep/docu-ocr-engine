from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Tuple, Annotated
import fitz  # PyMuPDF
import tempfile
import zipfile
import io
import os
import re
from datetime import datetime
from google.cloud import vision

# =========================
# SCRIVE IL JSON SU FILE
# =========================

creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    with open("/tmp/gcp-key.json", "w", encoding="utf-8") as f:
        f.write(creds_json)

app = FastAPI()


# =========================
# CONFIG
# =========================

FOLDERS = {
    "attestati": "attestati",
    "nomine": "nomine",
    "visite_mediche": "visite_mediche",
    "verbali_dpi": "verbali_dpi",
    "documenti_aziendali": "documenti_aziendali",
    "altri_da_verificare": "altri_da_verificare",
}

# years = None -> nessuna scadenza automatica
COURSE_RULES = {
    "FORMAZIONE_GENERALE": {"years": None, "label": "nessuna_scadenza"},
    "FORMAZIONE_SPECIFICA": {"years": 5, "label": "data_scadenza"},
    "AGGIORNAMENTO_FORMAZIONE_LAVORATORI": {"years": 5, "label": "data_scadenza"},
    "PRIMO_SOCCORSO": {"years": 3, "label": "data_scadenza"},
    "PREPOSTO": {"years": 2, "label": "data_scadenza"},
    "PONTEGGI": {"years": 4, "label": "data_scadenza"},
    "RLS": {"years": 1, "label": "prossimo_aggiornamento"},
    "RSPP_DL": {"years": 5, "label": "data_scadenza"},
    "ANTINCENDIO": {"years": 5, "label": "data_scadenza"},
    "CARRELLISTA": {"years": 5, "label": "data_scadenza"},
    "PLE": {"years": 5, "label": "data_scadenza"},
    "LAVORI_IN_QUOTA": {"years": 5, "label": "data_scadenza"},
    "HACCP": {"years": 5, "label": "data_scadenza"},
    "DEFAULT": {"years": 5, "label": "data_scadenza"},
}

OCR_SOFT_LIMIT = int(os.getenv("OCR_SOFT_LIMIT", "800"))


# =========================
# HELPERS GENERALI
# =========================

def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text_for_matching(text: str) -> str:
    text = text or ""
    text = normalize_spaces(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def upper_no_accents(text: str) -> str:
    repl = {
        "à": "a", "á": "a", "è": "e", "é": "e", "ì": "i", "í": "i",
        "ò": "o", "ó": "o", "ù": "u", "ú": "u",
        "À": "A", "Á": "A", "È": "E", "É": "E", "Ì": "I", "Í": "I",
        "Ò": "O", "Ó": "O", "Ù": "U", "Ú": "U"
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.upper()


def is_text_sufficient(text: str) -> bool:
    clean = normalize_text_for_matching(text)

    if not clean:
        return False
    if len(clean) < 80:
        return False
    if len(clean.split()) < 15:
        return False

    return True


def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s\-.]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name


def parse_date(date_str: str) -> Optional[datetime]:
    date_str = date_str.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt
        except ValueError:
            continue
    return None


def format_date(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.strftime("%d/%m/%Y")


def add_years_safe(dt: datetime, years: int) -> datetime:
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        return dt.replace(month=2, day=28, year=dt.year + years)


def first_non_empty(*values: str) -> str:
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return ""


# =========================
# OCR GOOGLE VISION
# =========================

def get_vision_client():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise RuntimeError("Variabile GOOGLE_APPLICATION_CREDENTIALS mancante")

    if not os.path.exists(credentials_path):
        raise RuntimeError(f"File credenziali non trovato: {credentials_path}")

    return vision.ImageAnnotatorClient()


def ocr_image_bytes(image_bytes: bytes) -> str:
    client = get_vision_client()
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Errore Google Vision OCR: {response.error.message}")

    texts = response.text_annotations
    if texts:
        return normalize_spaces(texts[0].description)

    return ""


def pdf_to_page_images(content: bytes, dpi: int = 200) -> List[bytes]:
    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()

    return images


def ocr_pdf_pages(content: bytes) -> Tuple[str, int]:
    page_images = pdf_to_page_images(content)
    texts = []

    for img_bytes in page_images:
        page_text = ocr_image_bytes(img_bytes)
        if page_text:
            texts.append(page_text)

    return normalize_spaces("\n\n".join(texts)), len(page_images)


# =========================
# ESTRAZIONE TESTO FILE
# =========================

def extract_pdf_text(content: bytes) -> str:
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except Exception:
        text = ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return normalize_spaces(text)


def extract_text_from_file(filename: str, content: bytes, content_type: str) -> dict:
    ext = os.path.splitext(filename.lower())[1]

    result = {
        "text": "",
        "extraction_method": "",
        "ocr_used": False,
        "ocr_pages": 0,
        "ocr_soft_limit": OCR_SOFT_LIMIT,
        "ocr_alert": False,
    }

    is_pdf = content_type == "application/pdf" or ext == ".pdf"
    is_image = (
        content_type.startswith("image/")
        or ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]
    )

    if is_pdf:
        pdf_text = extract_pdf_text(content)
        result["text"] = pdf_text
        result["extraction_method"] = "pymupdf"

        if not is_text_sufficient(pdf_text):
            ocr_text, page_count = ocr_pdf_pages(content)
            result["text"] = ocr_text
            result["extraction_method"] = "google_vision_ocr_pdf_pages"
            result["ocr_used"] = True
            result["ocr_pages"] = page_count
            result["ocr_alert"] = page_count >= OCR_SOFT_LIMIT

        return result

    if is_image:
        ocr_text = ocr_image_bytes(content)
        result["text"] = ocr_text
        result["extraction_method"] = "google_vision_ocr_image"
        result["ocr_used"] = True
        result["ocr_pages"] = 1
        result["ocr_alert"] = 1 >= OCR_SOFT_LIMIT
        return result

    result["text"] = ""
    result["extraction_method"] = "unsupported_file_type"
    return result


# =========================
# CLASSIFICAZIONE DOCUMENTI
# =========================

def score_category(text: str, filename: str) -> Tuple[str, dict]:
    blob = f"{filename}\n{text}".lower()

    scores = {
        "attestati": 0,
        "nomine": 0,
        "visite_mediche": 0,
        "verbali_dpi": 0,
        "documenti_aziendali": 0,
    }

    # Attestati
    if "attestato" in blob:
        scores["attestati"] += 5
    if "corso" in blob:
        scores["attestati"] += 2
    if "formazione" in blob:
        scores["attestati"] += 2
    if "partecipazione" in blob:
        scores["attestati"] += 2
    if "verifica dell’apprendimento" in blob or "verifica dell'apprendimento" in blob:
        scores["attestati"] += 2
    if "haccp" in blob:
        scores["attestati"] += 3

    # Nomine
    if "nomina" in blob:
        scores["nomine"] += 5
    if "designazione" in blob:
        scores["nomine"] += 3
    if "incarico" in blob:
        scores["nomine"] += 2
    if "preposto" in blob:
        scores["nomine"] += 2

    # Visite mediche
    if "giudizio di idoneità" in blob or "giudizio di idoneita" in blob:
        scores["visite_mediche"] += 5
    if "medico competente" in blob:
        scores["visite_mediche"] += 3
    if "idoneo" in blob or "idonea" in blob:
        scores["visite_mediche"] += 2
    if "sorveglianza sanitaria" in blob:
        scores["visite_mediche"] += 2

    # DPI
    if re.search(r"\bdpi\b", blob):
        scores["verbali_dpi"] += 5
    if "dispositivi di protezione individuale" in blob:
        scores["verbali_dpi"] += 4
    if "consegna" in blob:
        scores["verbali_dpi"] += 2
    if "firma per ricevuta" in blob:
        scores["verbali_dpi"] += 2

    # Documenti aziendali
    if re.search(r"\bdvr\b", blob):
        scores["documenti_aziendali"] += 5
    if "valutazione dei rischi" in blob:
        scores["documenti_aziendali"] += 4
    if "organigramma" in blob:
        scores["documenti_aziendali"] += 3
    if "protocollo" in blob:
        scores["documenti_aziendali"] += 2

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    if best_score <= 2:
        return "altri_da_verificare", scores

    return best_category, scores


# =========================
# ESTRAZIONE DATI ATTESTATI
# =========================

def extract_title_block(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title_lines = []
    for line in lines[:20]:
        if len(line) > 3:
            title_lines.append(line)
        if len(title_lines) >= 6:
            break
    return " ".join(title_lines).lower()


def detect_course_family(text: str, filename: str) -> Tuple[str, str]:
    blob = f"{filename}\n{text}".lower()
    title_blob = extract_title_block(text)

    # Prima individuiamo i casi più specifici
    if "haccp" in blob or "igiene degli alimenti" in blob or "alimentarista" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "HACCP", "aggiornamento" if is_update else "base"

    if (
        "aggiornamento della formazione per lavoratori" in blob
        or "aggiornamento formazione lavoratori" in blob
        or "corso di aggiornamento della formazione per lavoratori" in blob
        or "corso di aggiornamento della formazione lavoratori" in blob
    ):
        return "AGGIORNAMENTO_FORMAZIONE_LAVORATORI", "aggiornamento"

    if "formazione specifica" in blob or "parte specifica" in blob or "rischio alto" in title_blob or "rischio medio" in title_blob or "rischio basso" in title_blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "FORMAZIONE_SPECIFICA", "aggiornamento" if is_update else "base"

    if "formazione generale" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "FORMAZIONE_GENERALE", "aggiornamento" if is_update else "base"

    if "primo soccorso" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "PRIMO_SOCCORSO", "aggiornamento" if is_update else "base"

    if "preposto" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "PREPOSTO", "aggiornamento" if is_update else "base"

    if "ponteggi" in blob or "montaggio smontaggio trasformazione ponteggi" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "PONTEGGI", "aggiornamento" if is_update else "base"

    if "rappresentante dei lavoratori per la sicurezza" in blob or re.search(r"\brls\b", blob):
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "RLS", "aggiornamento" if is_update else "base"

    if "datore di lavoro" in blob and ("prevenzione e protezione" in blob or re.search(r"\brspp\b", blob)):
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "RSPP_DL", "aggiornamento" if is_update else "base"

    if "antincendio" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "ANTINCENDIO", "aggiornamento" if is_update else "base"

    if "carrello" in blob or "carrellista" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "CARRELLISTA", "aggiornamento" if is_update else "base"

    if re.search(r"\bple\b", blob):
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "PLE", "aggiornamento" if is_update else "base"

    if "lavori in quota" in blob:
        is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
        return "LAVORI_IN_QUOTA", "aggiornamento" if is_update else "base"

    is_update = any(k in blob for k in ["aggiornamento", "refresh", "rinnovo", "retraining"])
    return "CORSO_NON_RICONOSCIUTO", "aggiornamento" if is_update else "base"


def extract_name_from_attestato(text: str) -> Tuple[str, str]:
    clean_text = normalize_spaces(text)
    lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

    # 1. dopo "conferito a"
    m = re.search(r"conferito a\s+([A-ZÀ-Ù' ]{4,})", clean_text, re.IGNORECASE)
    if m:
        raw = normalize_spaces(m.group(1))
        raw = re.split(r"\n|nato a|nata a|nato il|nata il|qualifica", raw, flags=re.IGNORECASE)[0].strip()
        parts = [p for p in raw.split(" ") if p]
        if len(parts) >= 2:
            return parts[0].title(), " ".join(parts[1:]).title()

    # 2. riga prima di nato/nata a
    for i, line in enumerate(lines):
        if re.search(r"\bnato a\b|\bnata a\b", line, re.IGNORECASE) and i > 0:
            prev = lines[i - 1].strip()
            if re.fullmatch(r"[A-ZÀ-Ù' ]{4,}", prev) or re.fullmatch(r"[A-ZÀ-Ùa-zà-ù' ]{4,}", prev):
                parts = [p for p in prev.split() if p]
                if len(parts) >= 2:
                    return parts[0].title(), " ".join(parts[1:]).title()

    return "", ""


def extract_birth_date(text: str) -> Optional[datetime]:
    m = re.search(r"(nato a|nata a).*?(\d{2}[/-]\d{2}[/-]\d{2,4})", text, re.IGNORECASE | re.DOTALL)
    if m:
        return parse_date(m.group(2))
    return None


def extract_dates(text: str) -> List[datetime]:
    raw_dates = re.findall(r"\b\d{2}[/-]\d{2}[/-]\d{2,4}\b", text)
    parsed = []
    for d in raw_dates:
        dt = parse_date(d)
        if dt:
            parsed.append(dt)
    return parsed


def extract_conclusion_date(text: str) -> Optional[datetime]:
    # 1. priorità assoluta: data di conclusione del corso
    patterns = [
        r"data di conclusione del corso\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{2,4})",
        r"conclusione del corso\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{2,4})",
        r"data conclusione corso\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{2,4})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            dt = parse_date(m.group(1))
            if dt:
                return dt

    # 2. periodo di svolgimento corso -> prende ultima data
    m = re.search(
        r"(periodo di svolgimento del corso|svolgimento del corso|periodo corso)\s*:?\s*(.+?)(data emissione|attestato emesso|programma del corso|il responsabile|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        block = m.group(2)
        dates = extract_dates(block)
        if dates:
            return max(dates)

    # 3. dal ... al ...
    m = re.search(
        r"dal\s+(\d{2}[/-]\d{2}[/-]\d{2,4}).{0,30}?al\s+(\d{2}[/-]\d{2}[/-]\d{2,4})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        dt = parse_date(m.group(2))
        if dt:
            return dt

    # 4. ultima data plausibile escludendo nascita, emissione attestato
    all_dates = extract_dates(text)
    if not all_dates:
        return None

    birth_date = extract_birth_date(text)

    emission_match = re.search(
        r"(data emissione|attestato emesso il)\s*:?[\s]*(\d{2}[/-]\d{2}[/-]\d{2,4})",
        text,
        re.IGNORECASE,
    )
    emission_date = parse_date(emission_match.group(2)) if emission_match else None

    valid = []
    for d in all_dates:
        if birth_date and d.date() == birth_date.date():
            continue
        if emission_date and d.date() == emission_date.date():
            continue
        valid.append(d)

    if valid:
        return max(valid)

    return None


def compute_scadenza(course_family: str, conclusion_date: Optional[datetime]) -> Tuple[str, str]:
    if not conclusion_date:
        return "", ""

    rule = COURSE_RULES.get(course_family, COURSE_RULES["DEFAULT"])
    years = rule["years"]
    label = rule["label"]

    if years is None:
        return label, ""

    scad = add_years_safe(conclusion_date, years)
    return label, format_date(scad)


def build_attestato_filename(cognome: str, nome: str, course_family: str, original_filename: str) -> str:
    base_name = f"{cognome}_{nome}_ATTESTATO_{course_family}".strip("_")
    if base_name == "ATTESTATO_":
        base_name = os.path.splitext(original_filename)[0]
    return safe_filename(base_name) + ".pdf"


def parse_attestato(text: str, filename: str) -> dict:
    nome, cognome = extract_name_from_attestato(text)
    course_family, tipo_percorso = detect_course_family(text, filename)
    conclusion_date = extract_conclusion_date(text)
    scad_label, scad_value = compute_scadenza(course_family, conclusion_date)

    confidenza = "bassa"
    if nome and cognome and course_family != "CORSO_NON_RICONOSCIUTO" and conclusion_date:
        confidenza = "alta"
    elif (nome or cognome) and (course_family != "CORSO_NON_RICONOSCIUTO" or conclusion_date):
        confidenza = "media"

    suggested_name = build_attestato_filename(
        cognome.upper() if cognome else "",
        nome.upper() if nome else "",
        course_family,
        filename,
    )

    return {
        "categoria": "Attestato",
        "nome": nome,
        "cognome": cognome,
        "corso": course_family,
        "tipo_percorso": tipo_percorso,
        "data_conclusione": format_date(conclusion_date),
        "data_scadenza": scad_value if scad_label == "data_scadenza" else "",
        "prossimo_aggiornamento": scad_value if scad_label == "prossimo_aggiornamento" else "",
        "scadenza_label": scad_label,
        "confidenza": confidenza,
        "suggested_filename": suggested_name,
    }


# =========================
# ANALISI DOCUMENTO
# =========================

def analyze_document(filename: str, content: bytes, content_type: str) -> dict:
    extraction = extract_text_from_file(filename, content, content_type)
    text = extraction["text"]
    category, scores = score_category(text, filename)

    result = {
        "filename": filename,
        "content_type": content_type,
        "size_bytes": len(content),
        "testo_estratto": text[:2000],
        "categoria": category,
        "cartella": FOLDERS.get(category, "altri_da_verificare"),
        "nome": "",
        "cognome": "",
        "corso": "",
        "tipo_percorso": "",
        "data_conclusione": "",
        "data_scadenza": "",
        "prossimo_aggiornamento": "",
        "scadenza_label": "",
        "confidenza": "bassa",
        "score_details": scores,
        "suggested_filename": safe_filename(filename),
        "extraction_method": extraction["extraction_method"],
        "ocr_used": extraction["ocr_used"],
        "ocr_pages": extraction["ocr_pages"],
        "ocr_soft_limit": extraction["ocr_soft_limit"],
        "ocr_alert": extraction["ocr_alert"],
    }

    if category == "attestati":
        attestato_data = parse_attestato(text, filename)
        result.update(attestato_data)

    return result


# =========================
# ZIP + REPORT
# =========================

def build_report_attestati(items: List[dict]) -> str:
    lines = []
    lines.append("REPORT ATTESTATI")
    lines.append("=" * 100)
    lines.append("")

    for item in items:
        label = item.get("scadenza_label", "data_scadenza")
        label_value = first_non_empty(item.get("data_scadenza", ""), item.get("prossimo_aggiornamento", ""))
        lines.append(
            " | ".join([
                item.get("suggested_filename", ""),
                item.get("cognome", ""),
                item.get("nome", ""),
                item.get("corso", ""),
                item.get("tipo_percorso", ""),
                item.get("data_conclusione", ""),
                label,
                label_value,
                item.get("confidenza", ""),
                item.get("extraction_method", ""),
            ])
        )

    lines.append("")
    return "\n".join(lines)


def build_zip(files_data: List[Tuple[UploadFile, bytes]], analyzed: List[dict]) -> io.BytesIO:
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        attestati_report_items = []

        for (upload_file, content), item in zip(files_data, analyzed):
            folder = item["cartella"]
            ext = os.path.splitext(upload_file.filename)[1] or ".bin"

            suggested = item.get("suggested_filename", safe_filename(upload_file.filename))
            if not suggested.lower().endswith(ext.lower()):
                if folder != "attestati":
                    suggested = os.path.splitext(suggested)[0] + ext

            zip_path = f"{folder}/{suggested}"
            zf.writestr(zip_path, content)

            if folder == "attestati":
                attestati_report_items.append(item)

        if attestati_report_items:
            report_text = build_report_attestati(attestati_report_items)
            zf.writestr("attestati/report_attestati.txt", report_text)

    zip_buffer.seek(0)
    return zip_buffer


# =========================
# ENDPOINTS
# =========================

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Docu OCR Engine online"
    }


@app.post("/analyze")
async def analyze(file: Annotated[UploadFile, File(...)]):
    content = await file.read()
    item = analyze_document(file.filename, content, file.content_type or "")
    return {"results": [item]}


@app.post("/organize-zip")
async def organize_zip(files: Annotated[list[UploadFile], File(...)]):
    if not files:
        raise HTTPException(status_code=400, detail="Nessun file caricato")

    files_data: List[Tuple[UploadFile, bytes]] = []
    analyzed: List[dict] = []

    for file in files:
        content = await file.read()
        files_data.append((file, content))
        analyzed.append(analyze_document(file.filename, content, file.content_type or ""))

    zip_buffer = build_zip(files_data, analyzed)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="archivio_documenti.zip"'},
    )


@app.get("/health")
def health():
    return JSONResponse({"status": "healthy"})
