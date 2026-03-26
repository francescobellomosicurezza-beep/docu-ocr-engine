from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from typing import List, Optional, Tuple, Annotated
import fitz  # PyMuPDF
import tempfile
import zipfile
import io
import os
import re
import unicodedata
from datetime import datetime
from google.cloud import vision


# =========================
# GCP CREDS
# =========================

creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    with open("/tmp/gcp-key.json", "w", encoding="utf-8") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp-key.json"

app = FastAPI(title="Docu OCR Engine", version="2.1.0")


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

NOISE_LINE_PATTERNS = [
    r"^accedi$",
    r"^condividi$",
    r"^trova$",
    r"^titolo \d+$",
    r"^sottotitolo",
    r"^sostituisci$",
    r"^paragrafo$",
    r"^stili$",
    r"^abilita modifica$",
    r"^seleziona$",
    r"^modifica$",
    r"^adobe$",
    r"^q cerca$",
    r"^trova testo o strumenti",
    r"^chiedi all",
    r"^password ?\d*$",
    r"^lenovo$",
    r"^\d{1,2}:\d{2}$",
    r"^\d{1,2}/\d{1,2}/\d{4}$",
    r"^f\d{1,2}$",
    r"^pag$",
    r"^ins$",
    r"^canc$",
    r"^fine$",
    r"^bloc$",
    r"^num$",
    r"^stamp$",
    r"^pausa$",
    r"^interr$",
    r"^scorr$",
    r"^rsist$",
    r"^000$",
    r"^000 punti",
    r"^impostazioni di visualizzazione",
    r"^focus$",
]

NAME_ANCHORS = [
    "conferito a",
    "rilasciato a",
    "si attesta che",
    "attesta che",
    "certifica che",
    "ha partecipato",
    "ha frequentato",
]

UPDATE_WORDS = [
    "aggiornamento",
    "refresh",
    "rinnovo",
    "retraining",
    "update",
    "periodico",
    "modulo integrativo",
]

GENERIC_WORKER_TRAINING_PATTERNS = [
    "aggiornamento della formazione per lavoratori",
    "aggiornamento formazione lavoratori",
    "corso di aggiornamento della formazione per lavoratori",
    "corso di aggiornamento della formazione lavoratori",
]

SPECIFIC_COURSE_KEYWORDS = {
    "PRIMO_SOCCORSO": [
        "primo soccorso",
        "addetto al primo soccorso",
        "addetti al primo soccorso",
        "incaricati al primo soccorso",
        "lavoratori incaricati al primo soccorso",
        "d.m. 388/03",
        "gruppo b-c",
        "gruppo b/c",
    ],
    "ANTINCENDIO": [
        "antincendio",
        "lotta antincendio",
        "gestione emergenze incendio",
        "addetto antincendio",
        "addetti antincendio",
        "incaricati antincendio",
    ],
    "PREPOSTO": [
        "preposto",
        "corso preposto",
    ],
    "PONTEGGI": [
        "ponteggi",
        "montaggio smontaggio trasformazione ponteggi",
        "pi.m.u.s",
        "pimus",
    ],
    "RLS": [
        "rappresentante dei lavoratori per la sicurezza",
        " rls ",
    ],
    "RSPP_DL": [
        "datore di lavoro rspp",
        "responsabile del servizio di prevenzione e protezione",
        "rspp datore di lavoro",
        "datore di lavoro che svolge i compiti del servizio di prevenzione e protezione",
    ],
    "CARRELLISTA": [
        "carrellista",
        "carrello elevatore",
        "carrelli elevatori",
        "muletto",
        "mulettista",
    ],
    "PLE": [
        " ple ",
        "piattaforma di lavoro elevabile",
        "piattaforme di lavoro elevabili",
    ],
    "LAVORI_IN_QUOTA": [
        "lavori in quota",
        "sistemi anticaduta",
    ],
    "HACCP": [
        "haccp",
        "igiene degli alimenti",
        "igiene alimentare",
        "alimentarista",
        "alimentaristi",
        "settore alimentare",
        "addetti del settore alimentare",
        "addetti alla manipolazione di alimenti",
        "manipolazione di alimenti",
        "alimenti deperibili",
        "regolamenti ce n. 852/2004",
        "regolamento ce 852/2004",
        "regolamento ce 853/2004",
        "852/04",
        "853/04",
        "tipologia a",
        "tipologia b",
        "modulo integrativo",
    ],
}

GENERAL_TRAINING_KEYWORDS = {
    "FORMAZIONE_GENERALE": [
        "formazione generale",
        "parte generale",
    ],
    "FORMAZIONE_SPECIFICA": [
        "formazione specifica",
        "parte specifica",
        "rischio basso",
        "rischio medio",
        "rischio alto",
    ],
}


# =========================
# HELPERS TESTO
# =========================

def normalize_spaces(text: str) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def normalize_text_for_matching(text: str) -> str:
    text = normalize_spaces(text)
    text = strip_accents(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_line_for_matching(text: str) -> str:
    text = strip_accents(text or "")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def upper_no_accents(text: str) -> str:
    return strip_accents(text or "").upper()


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


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# =========================
# HELPERS DATE
# =========================

def parse_date(date_str: str) -> Optional[datetime]:
    date_str = (date_str or "").strip()

    # fallback robusto per 1 o 2 cifre
    m = re.fullmatch(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})", date_str)
    if m:
        d = int(m.group(1))
        mth = int(m.group(2))
        y = int(m.group(3))
        if y < 100:
            y += 2000 if y < 70 else 1900
        try:
            return datetime(y, mth, d)
        except ValueError:
            return None

    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt
        except Exception:
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


def extract_dates(text: str) -> List[datetime]:
    raw_dates = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", text or "")
    parsed = []
    for d in raw_dates:
        dt = parse_date(d)
        if dt:
            parsed.append(dt)
    return parsed


# =========================
# HELPERS PERSONA
# =========================

def is_plausible_person_name_line(line: str) -> bool:
    raw = normalize_spaces(line)
    if not raw:
        return False

    line_norm = normalize_line_for_matching(raw)

    if len(raw) < 5 or len(raw) > 80:
        return False

    if re.search(r"\d{2,}", raw):
        return False
    if re.search(r"[<>{}\[\]|_=+/*\\]", raw):
        return False

    forbidden = [
        "attestato",
        "corso",
        "nato",
        "nata",
        "durata",
        "ore",
        "data",
        "responsabile",
        "progetto",
        "modalita",
        "e-learning",
        "elearning",
        "regolamenti",
        "tipologia",
        "ai sensi",
        "rilasciato",
        "conferito",
        "certifica",
        "partecipazione",
        "formazione",
        "modulo",
        "programma",
    ]
    if any(tok in line_norm for tok in forbidden):
        return False

    words = [w for w in re.split(r"\s+", raw) if w]
    if len(words) < 2 or len(words) > 5:
        return False

    letter_words = 0
    for w in words:
        clean = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'’\-]", "", w)
        if len(clean) >= 2:
            letter_words += 1
    if letter_words < 2:
        return False

    return True


def split_name_line(line: str) -> Tuple[str, str]:
    words = [w.strip(" ,.;:-") for w in line.split() if w.strip(" ,.;:-")]
    if len(words) < 2:
        return "", ""

    nome = words[0].title()
    cognome = " ".join(words[1:]).title()
    return nome, cognome


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
# PULIZIA OCR
# =========================

def remove_noise_lines(text: str) -> str:
    cleaned_lines = []

    for line in (text or "").splitlines():
        raw = normalize_spaces(line)
        if not raw:
            continue

        line_norm = normalize_line_for_matching(raw)

        is_noise = False
        for pat in NOISE_LINE_PATTERNS:
            if re.search(pat, line_norm, re.IGNORECASE):
                is_noise = True
                break
        if is_noise:
            continue

        if len(raw) <= 2:
            continue
        if re.fullmatch(r"[\W_]+", raw):
            continue
        if re.fullmatch(r"[A-Za-z]?\d{1,2}[A-Za-z]?", raw):
            continue

        letters = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", raw))
        digits = len(re.findall(r"\d", raw))
        if letters == 0 and digits <= 4:
            continue

        cleaned_lines.append(raw)

    return normalize_spaces("\n".join(cleaned_lines))


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
        "raw_text": "",
        "extraction_method": "",
        "ocr_used": False,
        "ocr_pages": 0,
        "ocr_soft_limit": OCR_SOFT_LIMIT,
        "ocr_alert": False,
        "extraction_error": "",
    }

    is_pdf = content_type == "application/pdf" or ext == ".pdf"
    is_image = (
        content_type.startswith("image/")
        or ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]
    )

    try:
        if is_pdf:
            pdf_text = extract_pdf_text(content)
            result["raw_text"] = pdf_text
            result["text"] = remove_noise_lines(pdf_text)
            result["extraction_method"] = "pymupdf"

            if not is_text_sufficient(result["text"]):
                ocr_text, page_count = ocr_pdf_pages(content)
                result["raw_text"] = ocr_text
                result["text"] = remove_noise_lines(ocr_text)
                result["extraction_method"] = "google_vision_ocr_pdf_pages"
                result["ocr_used"] = True
                result["ocr_pages"] = page_count
                result["ocr_alert"] = page_count >= OCR_SOFT_LIMIT

            return result

        if is_image:
            ocr_text = ocr_image_bytes(content)
            result["raw_text"] = ocr_text
            result["text"] = remove_noise_lines(ocr_text)
            result["extraction_method"] = "google_vision_ocr_image"
            result["ocr_used"] = True
            result["ocr_pages"] = 1
            result["ocr_alert"] = False
            return result

        result["text"] = ""
        result["raw_text"] = ""
        result["extraction_method"] = "unsupported_file_type"
        return result

    except Exception as e:
        result["text"] = ""
        result["raw_text"] = ""
        result["extraction_method"] = "extraction_failed"
        result["extraction_error"] = str(e)
        return result


# =========================
# CLASSIFICAZIONE DOCUMENTI
# =========================

def score_category(text: str, filename: str) -> Tuple[str, dict]:
    blob = normalize_text_for_matching(f"{filename}\n{text}")

    scores = {
        "attestati": 0,
        "nomine": 0,
        "visite_mediche": 0,
        "verbali_dpi": 0,
        "documenti_aziendali": 0,
    }

    if "attestato" in blob:
        scores["attestati"] += 5
    if "corso" in blob:
        scores["attestati"] += 2
    if "formazione" in blob:
        scores["attestati"] += 2
    if "partecipazione" in blob:
        scores["attestati"] += 2
    if "verifica dell'apprendimento" in blob or "verifica dell apprendimento" in blob:
        scores["attestati"] += 2
    if "data di svolgimento del corso" in blob or "data di conclusione del corso" in blob:
        scores["attestati"] += 2

    if "nomina" in blob:
        scores["nomine"] += 5
    if "designazione" in blob:
        scores["nomine"] += 3
    if "incarico" in blob:
        scores["nomine"] += 2
    if "preposto" in blob:
        scores["nomine"] += 2

    if "giudizio di idoneita" in blob or "giudizio di idoneità" in blob:
        scores["visite_mediche"] += 5
    if "medico competente" in blob:
        scores["visite_mediche"] += 3
    if "idoneo" in blob or "idonea" in blob:
        scores["visite_mediche"] += 2
    if "sorveglianza sanitaria" in blob:
        scores["visite_mediche"] += 2

    if re.search(r"\bdpi\b", blob):
        scores["verbali_dpi"] += 5
    if "dispositivi di protezione individuale" in blob:
        scores["verbali_dpi"] += 4
    if "consegna" in blob:
        scores["verbali_dpi"] += 2
    if "firma per ricevuta" in blob:
        scores["verbali_dpi"] += 2

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
# PARSER ATTESTATI
# =========================

def extract_title_block(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title_lines = []
    for line in lines[:20]:
        if len(line) > 3:
            title_lines.append(line)
        if len(title_lines) >= 8:
            break
    return normalize_text_for_matching(" ".join(title_lines))


def has_any_keyword(blob: str, keywords: List[str]) -> bool:
    blob_padded = f" {blob} "
    for kw in keywords:
        k = normalize_text_for_matching(kw)
        if k in blob or k in blob_padded:
            return True
    return False


def detect_course_family(text: str, filename: str) -> Tuple[str, str, List[str]]:
    blob = normalize_text_for_matching(f"{filename}\n{text}")
    title_blob = extract_title_block(text)
    debug = []

    is_update = any(w in blob for w in UPDATE_WORDS)

    # 1. corsi specifici: devono vincere SEMPRE sui generici
    for family in [
        "PRIMO_SOCCORSO",
        "ANTINCENDIO",
        "PREPOSTO",
        "PONTEGGI",
        "RLS",
        "RSPP_DL",
        "CARRELLISTA",
        "PLE",
        "LAVORI_IN_QUOTA",
        "HACCP",
    ]:
        if has_any_keyword(blob, SPECIFIC_COURSE_KEYWORDS[family]) or has_any_keyword(title_blob, SPECIFIC_COURSE_KEYWORDS[family]):
            debug.append(f"match corso specifico: {family}")
            return family, "aggiornamento" if is_update else "base", debug

    # 2. formazione lavoratori GENERICA
    # solo se NON ci sono indicatori di corso specifico
    if has_any_keyword(blob, GENERIC_WORKER_TRAINING_PATTERNS):
        debug.append("match corso: AGGIORNAMENTO_FORMAZIONE_LAVORATORI")
        return "AGGIORNAMENTO_FORMAZIONE_LAVORATORI", "aggiornamento", debug

    if has_any_keyword(blob, GENERAL_TRAINING_KEYWORDS["FORMAZIONE_SPECIFICA"]) or has_any_keyword(title_blob, GENERAL_TRAINING_KEYWORDS["FORMAZIONE_SPECIFICA"]):
        debug.append("match corso: FORMAZIONE_SPECIFICA")
        return "FORMAZIONE_SPECIFICA", "aggiornamento" if is_update else "base", debug

    if has_any_keyword(blob, GENERAL_TRAINING_KEYWORDS["FORMAZIONE_GENERALE"]):
        debug.append("match corso: FORMAZIONE_GENERALE")
        return "FORMAZIONE_GENERALE", "aggiornamento" if is_update else "base", debug

    debug.append("corso non riconosciuto")
    return "CORSO_NON_RICONOSCIUTO", "aggiornamento" if is_update else "base", debug


def extract_name_from_attestato(text: str) -> Tuple[str, str, List[str]]:
    debug = []
    clean_text = normalize_spaces(text)
    lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

    anchors_regex = "|".join(re.escape(a) for a in NAME_ANCHORS)
    m = re.search(
        rf"(?:{anchors_regex})\s*[:\-]?\s*([A-Za-zÀ-ÖØ-öø-ÿ'’\-\s]{{5,80}})",
        clean_text,
        re.IGNORECASE,
    )
    if m:
        raw = normalize_spaces(m.group(1))
        raw = re.split(
            r"\b(nato a|nata a|nato\/a a|nato il|nata il|data di nascita|qualifica|mansione|il corso|data di conclusione|data di svolgimento|attestato emesso|data emissione)\b",
            raw,
            flags=re.IGNORECASE,
        )[0].strip(" ,.;:-")
        if is_plausible_person_name_line(raw):
            nome, cognome = split_name_line(raw)
            if nome and cognome:
                debug.append("nome trovato con regex dopo anchor")
                return nome, cognome, debug

    for i, line in enumerate(lines):
        line_norm = normalize_line_for_matching(line)
        if any(anchor in line_norm for anchor in NAME_ANCHORS):
            for cand in lines[i + 1:i + 5]:
                if is_plausible_person_name_line(cand):
                    nome, cognome = split_name_line(cand)
                    if nome and cognome:
                        debug.append("nome trovato nelle righe successive ad anchor")
                        return nome, cognome, debug

    for i, line in enumerate(lines):
        if re.search(r"\bnato\b|\bnata\b|\bnato\/a\b", line, re.IGNORECASE):
            prev_candidates = list(reversed(lines[max(0, i - 3):i]))
            for prev in prev_candidates:
                if is_plausible_person_name_line(prev):
                    nome, cognome = split_name_line(prev)
                    if nome and cognome:
                        debug.append("nome trovato prima di riga con nato/nata")
                        return nome, cognome, debug

    for cand in lines[:35]:
        if is_plausible_person_name_line(cand):
            nome, cognome = split_name_line(cand)
            if nome and cognome:
                debug.append("nome trovato per fallback su riga plausibile")
                return nome, cognome, debug

    debug.append("nome non trovato")
    return "", "", debug


def extract_birth_date(text: str) -> Optional[datetime]:
    patterns = [
        r"(nato a|nata a|nato\/a a|nato in|nata in).*?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"(nato il|nata il|data di nascita).*?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return parse_date(m.group(2))
    return None


def find_date_near_labels(lines: List[str], labels: List[str]) -> Optional[datetime]:
    for i, line in enumerate(lines):
        line_norm = normalize_line_for_matching(line)
        if any(lbl in line_norm for lbl in labels):
            same_line_dates = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", line)
            for d in same_line_dates:
                dt = parse_date(d)
                if dt:
                    return dt

            for j in range(i + 1, min(i + 3, len(lines))):
                near_dates = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", lines[j])
                for d in near_dates:
                    dt = parse_date(d)
                    if dt:
                        return dt
    return None


def extract_conclusion_date(text: str) -> Tuple[Optional[datetime], List[str]]:
    debug = []
    lines = [l.strip() for l in normalize_spaces(text).splitlines() if l.strip()]

    label_groups = [
        [
            "data di conclusione del corso",
            "conclusione del corso",
            "data conclusione corso",
            "data conclusione",
        ],
        [
            "data di svolgimento del corso",
            "data di svolgimento",
            "data svolgimento corso",
            "svolgimento del corso",
        ],
        [
            "data fine corso",
            "fine corso",
            "terminato il",
            "concluso il",
        ],
    ]

    for labels in label_groups:
        dt = find_date_near_labels(lines, labels)
        if dt:
            debug.append(f"data conclusione trovata vicino a label: {labels[0]}")
            return dt, debug

    patterns = [
        r"data di conclusione del corso\s*:?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"conclusione del corso\s*:?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"data conclusione corso\s*:?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"data di svolgimento del corso\s*:?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"data di svolgimento\s*:?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"data fine corso\s*:?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            dt = parse_date(m.group(1))
            if dt:
                debug.append("data conclusione trovata con regex diretta")
                return dt, debug

    m = re.search(
        r"(periodo di svolgimento del corso|svolgimento del corso|periodo corso|dal|durata del corso)(.+?)(data emissione|attestato emesso|programma del corso|il responsabile|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        block = m.group(2)
        dates = extract_dates(block)
        if dates:
            debug.append("data conclusione ricavata da blocco periodo corso")
            return max(dates), debug

    m = re.search(
        r"dal\s+(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}).{0,80}?al\s+(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        dt = parse_date(m.group(2))
        if dt:
            debug.append("data conclusione ricavata da pattern dal/al")
            return dt, debug

    all_dates = extract_dates(text)
    if not all_dates:
        debug.append("nessuna data trovata")
        return None, debug

    birth_date = extract_birth_date(text)

    emission_match = re.search(
        r"(data emissione|attestato emesso il|rilasciato il)\s*:?[\s]*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
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
        if d.year < 1990 or d.year > datetime.now().year + 1:
            continue
        valid.append(d)

    if valid:
        debug.append("data conclusione scelta come ultima data plausibile filtrata")
        return max(valid), debug

    debug.append("nessuna data conclusione affidabile")
    return None, debug


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
    if base_name in {"", "ATTESTATO", "ATTESTATO_"}:
        base_name = os.path.splitext(original_filename)[0]
    return safe_filename(base_name) + ".pdf"


def compute_review_flag(
    nome: str,
    cognome: str,
    course_family: str,
    conclusion_date: Optional[datetime],
    confidenza: str,
) -> Tuple[bool, List[str]]:
    reasons = []

    if not nome or not cognome:
        reasons.append("nome_o_cognome_non_estratti")

    if course_family == "CORSO_NON_RICONOSCIUTO":
        reasons.append("corso_non_riconosciuto")

    if not conclusion_date:
        reasons.append("data_conclusione_non_trovata")

    if confidenza == "bassa":
        reasons.append("confidenza_bassa")

    return len(reasons) > 0, reasons


def parse_attestato(text: str, filename: str) -> dict:
    debug_notes = []

    nome, cognome, name_debug = extract_name_from_attestato(text)
    debug_notes.extend(name_debug)

    course_family, tipo_percorso, course_debug = detect_course_family(text, filename)
    debug_notes.extend(course_debug)

    conclusion_date, date_debug = extract_conclusion_date(text)
    debug_notes.extend(date_debug)

    scad_label, scad_value = compute_scadenza(course_family, conclusion_date)

    points = 0
    if nome and cognome:
        points += 3
    if course_family != "CORSO_NON_RICONOSCIUTO":
        points += 3
    if conclusion_date:
        points += 3
    if scad_label or scad_value or course_family == "FORMAZIONE_GENERALE":
        points += 1

    if points >= 8:
        confidenza = "alta"
    elif points >= 5:
        confidenza = "media"
    else:
        confidenza = "bassa"

    suggested_name = build_attestato_filename(
        cognome.upper() if cognome else "",
        nome.upper() if nome else "",
        course_family,
        filename,
    )

    needs_review, review_reasons = compute_review_flag(
        nome=nome,
        cognome=cognome,
        course_family=course_family,
        conclusion_date=conclusion_date,
        confidenza=confidenza,
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
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "parser_debug": unique_preserve_order(debug_notes),
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
        "testo_estratto": text[:3000],
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
        "needs_review": False,
        "review_reasons": [],
        "score_details": scores,
        "suggested_filename": safe_filename(filename),
        "extraction_method": extraction["extraction_method"],
        "ocr_used": extraction["ocr_used"],
        "ocr_pages": extraction["ocr_pages"],
        "ocr_soft_limit": extraction["ocr_soft_limit"],
        "ocr_alert": extraction["ocr_alert"],
        "extraction_error": extraction.get("extraction_error", ""),
        "parser_debug": [],
    }

    if extraction["extraction_method"] == "extraction_failed":
        result["categoria"] = "altri_da_verificare"
        result["cartella"] = FOLDERS["altri_da_verificare"]
        result["needs_review"] = True
        result["review_reasons"] = ["estrazione_testo_fallita"]
        result["confidenza"] = "bassa"
        result["parser_debug"] = ["estrazione testo fallita"]
        return result

    if category == "attestati":
        attestato_data = parse_attestato(text, filename)
        result.update(attestato_data)

    return result


# =========================
# ZIP + REPORT
# =========================

def first_non_empty(*values: str) -> str:
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return ""


def build_report_attestati(items: List[dict]) -> str:
    lines = []
    lines.append("REPORT ATTESTATI")
    lines.append("=" * 120)
    lines.append("")

    for item in items:
        label = item.get("scadenza_label", "data_scadenza")
        label_value = first_non_empty(
            item.get("data_scadenza", ""),
            item.get("prossimo_aggiornamento", "")
        )
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
                str(item.get("needs_review", False)),
                ",".join(item.get("review_reasons", [])),
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
    return {"status": "ok", "message": "Docu OCR Engine online"}


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return """
    <html>
      <head>
        <title>Upload Attestati</title>
      </head>
      <body style="font-family:Arial;padding:40px;">
        <h2>Carica attestato singolo</h2>
        <form action="/analyze" enctype="multipart/form-data" method="post">
          <input name="file" type="file" />
          <button type="submit">Analizza</button>
        </form>
        <hr>
        <h2>Carica più file</h2>
        <form action="/organize-zip" enctype="multipart/form-data" method="post">
          <input name="files" type="file" multiple />
          <button type="submit">Organizza ZIP</button>
        </form>
      </body>
    </html>
    """


@app.post("/analyze")
async def analyze(file: Annotated[UploadFile, File(...)]):
    try:
        content = await file.read()
        item = analyze_document(file.filename, content, file.content_type or "")
        return {"results": [item]}
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "results": [{
                    "filename": file.filename if file else "",
                    "content_type": file.content_type if file else "",
                    "size_bytes": len(content) if "content" in locals() else 0,
                    "testo_estratto": "",
                    "categoria": "altri_da_verificare",
                    "cartella": "altri_da_verificare",
                    "nome": "",
                    "cognome": "",
                    "corso": "",
                    "tipo_percorso": "",
                    "data_conclusione": "",
                    "data_scadenza": "",
                    "prossimo_aggiornamento": "",
                    "scadenza_label": "",
                    "confidenza": "bassa",
                    "needs_review": True,
                    "review_reasons": ["errore_interno_backend"],
                    "score_details": {},
                    "suggested_filename": safe_filename(file.filename) if file else "",
                    "extraction_method": "fatal_error",
                    "ocr_used": False,
                    "ocr_pages": 0,
                    "ocr_soft_limit": OCR_SOFT_LIMIT,
                    "ocr_alert": False,
                    "extraction_error": str(e),
                    "parser_debug": ["eccezione gestita in endpoint /analyze"],
                }]
            }
        )


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
