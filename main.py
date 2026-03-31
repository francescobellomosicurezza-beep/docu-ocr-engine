from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from typing import List, Optional, Tuple, Dict, Any, Annotated
import fitz  # PyMuPDF
import tempfile
import zipfile
import io
import os
import re
import unicodedata
import json
from datetime import datetime
from google.cloud import vision


# =========================================================
# GCP CREDS
# =========================================================

creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    with open("/tmp/gcp-key.json", "w", encoding="utf-8") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp-key.json"


# =========================================================
# APP
# =========================================================

app = FastAPI(title="Docu OCR Engine", version="6.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CONFIG
# =========================================================

FOLDERS = {
    "attestati": "attestati",
    "nomine": "nomine",
    "visite_mediche": "visite_mediche",
    "verbali_dpi": "verbali_dpi",
    "documenti_aziendali": "documenti_aziendali",
    "altri_da_verificare": "altri_da_verificare",
}

CATEGORY_LABELS = {
    "attestati": "Attestati",
    "nomine": "Nomine",
    "visite_mediche": "Visite Mediche",
    "verbali_dpi": "Verbali DPI",
    "documenti_aziendali": "Documenti Aziendali",
    "altri_da_verificare": "Da verificare",
}

CATEGORY_LABEL_TO_KEY = {
    "attestati": "attestati",
    "attestato": "attestati",
    "nomine": "nomine",
    "nomina": "nomine",
    "visite mediche": "visite_mediche",
    "visita medica": "visite_mediche",
    "verbali dpi": "verbali_dpi",
    "verbale dpi": "verbali_dpi",
    "documenti aziendali": "documenti_aziendali",
    "documento aziendale": "documenti_aziendali",
    "da verificare": "altri_da_verificare",
    "altri_da_verificare": "altri_da_verificare",
    "altri da verificare": "altri_da_verificare",
}

COURSE_RULES = {
    "FORMAZIONE_LAVORATORI": {"years": 5, "label": "data_scadenza"},
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
    "CORSO_NON_RICONOSCIUTO": {"years": 5, "label": "data_scadenza"},
    "DEFAULT": {"years": 5, "label": "data_scadenza"},
}

OCR_SOFT_LIMIT = int(os.getenv("OCR_SOFT_LIMIT", "800"))


# =========================================================
# COSTANTI TESTO / PARSING
# =========================================================

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

INVALID_NAME_TOKENS = {
    "il", "la", "lo", "i", "gli", "le",
    "sig", "sig.", "sigra", "sig.ra", "sig.na",
    "signor", "signora", "sign", "dr", "avv", "ing",
    "dott", "dott.", "dottssa", "dott.ssa",
    "nato", "nata", "nato/a", "nata/a",
    "attestato", "corso", "data", "luogo", "nr", "n",
    "conferito", "rilasciato", "certifica", "attesta",
    "ai", "sensi", "della", "del", "dei",
    "qualifica", "mansione", "settore", "riferimento",
    "codice", "ateco", "verifica", "apprendimento",
    "test", "esito", "superamento",
    "azienda", "srl", "spa", "snc", "sas"
}

UPDATE_WORDS = [
    "aggiornamento",
    "refresh",
    "rinnovo",
    "retraining",
    "update",
    "periodico",
    "modulo integrativo",
    "agg.to",
]

GENERIC_WORKER_TRAINING_PATTERNS = [
    "aggiornamento della formazione per lavoratori",
    "aggiornamento formazione lavoratori",
    "corso di aggiornamento della formazione per lavoratori",
    "corso di aggiornamento della formazione lavoratori",
    "formazione dei lavoratori",
    "formazione lavoratori",
]

SPECIFIC_COURSE_KEYWORDS = {
    "PRIMO_SOCCORSO": [
        "primo soccorso",
        "addetto al primo soccorso",
        "addetti al primo soccorso",
        "incaricati al primo soccorso",
        "lavoratori incaricati al primo soccorso",
        "corso di primo soccorso",
        "aggiornamento primo soccorso",
        "addetto primo soccorso",
        "d.m. 388/03",
        "dm 388/03",
        "388/03",
        "gruppo b-c",
        "gruppo b/c",
        "gruppo a",
    ],
    "ANTINCENDIO": [
        "antincendio",
        "lotta antincendio",
        "gestione emergenze incendio",
        "addetto antincendio",
        "addetti antincendio",
        "addetto alla prevenzione incendi",
        "addetti alla prevenzione incendi",
        "incaricati antincendio",
        "corso antincendio",
        "aggiornamento antincendio",
        "addetti alla lotta antincendio",
        "gestione delle emergenze",
        "rischio basso",
        "rischio medio",
        "rischio alto",
    ],
    "PREPOSTO": [
        "preposto",
        "corso preposto",
        "aggiornamento preposto",
        "formazione particolare aggiuntiva per il preposto",
        "formazione aggiuntiva per il preposto",
    ],
    "PONTEGGI": [
        "ponteggi",
        "montaggio smontaggio trasformazione ponteggi",
        "montaggio e smontaggio ponteggi",
        "trasformazione ponteggi",
        "pi.m.u.s",
        "pimus",
        "addetti ai ponteggi",
    ],
    "RLS": [
        "rappresentante dei lavoratori per la sicurezza",
        " rls ",
        "r.l.s",
        "corso rls",
        "aggiornamento rls",
    ],
    "RSPP_DL": [
        "datore di lavoro rspp",
        "responsabile del servizio di prevenzione e protezione",
        "rspp datore di lavoro",
        "datore di lavoro che svolge i compiti del servizio di prevenzione e protezione",
        "corso rspp datore di lavoro",
        "aggiornamento rspp datore di lavoro",
    ],
    "CARRELLISTA": [
        "carrellista",
        "carrello elevatore",
        "carrelli elevatori",
        "muletto",
        "mulettista",
        "addetto alla conduzione di carrelli elevatori",
        "abilitazione carrelli elevatori",
        "uso del carrello elevatore",
        "conduzione di carrelli elevatori",
    ],
    "PLE": [
        "piattaforma di lavoro elevabile",
        "piattaforme di lavoro elevabili",
        "ple",
        "uso di ple",
        "utilizzo di ple",
        "addetti all'uso di ple",
        "addetti all’uso di ple",
        "lavoratori addetti all'uso di ple",
        "lavoratori addetti all’uso di ple",
        "ple con stabilizzatori",
        "ple senza stabilizzatori",
        "ple con o senza stabilizzatori",
        "con o senza stabilizzatori",
        "corso ple",
        "aggiornamento ple",
    ],
    "LAVORI_IN_QUOTA": [
        "lavori in quota",
        "sistemi anticaduta",
        "dpi anticaduta",
        "uso dei dpi anticaduta",
        "corso lavori in quota",
        "aggiornamento lavori in quota",
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

NOMINA_ROLE_KEYWORDS = {
    "PRIMO_SOCCORSO": ["primo soccorso", "addetto primo soccorso", "addetto al primo soccorso"],
    "ANTINCENDIO": ["antincendio", "addetto antincendio", "addetto alla prevenzione incendi"],
    "PREPOSTO": ["preposto"],
    "RSPP": ["rspp", "responsabile del servizio di prevenzione e protezione"],
    "RLS": ["rls", "rappresentante dei lavoratori per la sicurezza"],
}

VISITA_ESITI = [
    "idoneo con prescrizioni",
    "idonea con prescrizioni",
    "temporaneamente non idoneo",
    "temporaneamente non idonea",
    "non idoneo",
    "non idonea",
    "idoneo",
    "idonea",
]

ATTESTATO_POSITIVE_SIGNALS = [
    "attestato",
    "attestato di frequenza",
    "attestato di formazione",
    "attestato di partecipazione",
    "si attesta che",
    "certifica che",
    "ha frequentato",
    "ha partecipato",
    "verifica dell'apprendimento",
    "verifica apprendimento",
    "rilasciato",
    "conferito",
    "programma del corso",
]

ATTESTATO_NEGATIVE_SIGNALS = [
    "nomina",
    "designazione",
    "verbale di consegna",
    "dispositivi di protezione individuale",
    "giudizio di idoneita",
    "giudizio di idoneità",
    "medico competente",
]

DATE_STRONG_LABELS = [
    "data di conclusione del corso",
    "conclusione del corso",
    "data conclusione corso",
    "data conclusione",
    "data di svolgimento del corso",
    "data di svolgimento",
    "data svolgimento corso",
    "svolgimento del corso",
    "svolto in data",
    "concluso il",
    "terminato il",
    "data fine corso",
    "fine corso",
]

DATE_WEAK_PERIOD_LABELS = [
    "periodo di svolgimento del corso",
    "giorni",
    "dal",
    "al",
]

DATE_NEGATIVE_CONTEXTS = [
    "nato il",
    "nata il",
    "data di nascita",
    "accreditat",
    "regione",
    "d.d.",
    "d.d. n",
    "attestato emesso",
    "data emissione",
    "rilasciato il",
    "n. iscrizione",
]
MONTHS_IT = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}

# =========================================================
# HELPERS BASE
# =========================================================

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


def safe_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w\s\-.]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name or "documento"


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def first_non_empty(*values: str) -> str:
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return ""


def has_any_keyword(blob: str, keywords: List[str]) -> bool:
    for kw in keywords:
        kw_norm = normalize_text_for_matching(kw)
        if kw_norm in blob:
            return True
    return False


def count_keywords(blob: str, keywords: List[str]) -> int:
    found = 0
    for kw in keywords:
        kw_norm = normalize_text_for_matching(kw)
        if kw_norm in blob:
            found += 1
    return found


def is_text_sufficient(text: str) -> bool:
    clean = normalize_text_for_matching(text)
    if not clean:
        return False
    if len(clean) < 80:
        return False
    if len(clean.split()) < 15:
        return False
    return True


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


# =========================================================
# HELPERS DATE
# =========================================================

def parse_date(date_str: str) -> Optional[datetime]:
    date_str = normalize_spaces(date_str or "")
    if not date_str:
        return None

    # formato numerico: 01/10/2021
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

    # formato testuale italiano: 01 Ottobre 2021
    text_norm = normalize_text_for_matching(date_str)
    m2 = re.fullmatch(
        r"(\d{1,2})\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+(\d{4})",
        text_norm,
    )
    if m2:
        d = int(m2.group(1))
        month_name = m2.group(2)
        y = int(m2.group(3))
        mth = MONTHS_IT.get(month_name)
        if mth:
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
    out = []

    raw_numeric_dates = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", text or "")
    raw_text_dates = re.findall(
        r"\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}\b",
        normalize_text_for_matching(text or ""),
        flags=re.IGNORECASE,
    )

    for d in raw_numeric_dates:
        dt = parse_date(d)
        if dt:
            out.append(dt)

    for d in raw_text_dates:
        dt = parse_date(d)
        if dt:
            out.append(dt)

    return out


def extract_birth_date(text: str) -> Optional[datetime]:
    patterns = [
        r"(nato a|nata a|nato\/a a|nato in|nata in).*?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"(nato il|nata il|data di nascita).*?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"(nato a|nata a|nato\/a a|nato in|nata in).*?(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
        r"(nato il|nata il|data di nascita).*?(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, normalize_text_for_matching(text), re.IGNORECASE | re.DOTALL)
        if m:
            return parse_date(m.group(2))
    return None


# =========================================================
# HELPERS NOME PERSONA
# =========================================================

def clean_person_line(line: str) -> str:
    raw = normalize_spaces(line)
    raw = re.sub(r"\b(Il|La|Lo)\s+(Sig\.?|Sig\.ra|Sig\.na|Signor|Signora)\b", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\b(Sig\.?|Sig\.ra|Sig\.na|Signor|Signora)\b", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\b(Dott\.ssa|Dott.ssa|Dott\.|Dott|Dr\.|Dr)\b", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip(" ,.;:-")
    return raw


def is_valid_name_token(token: str) -> bool:
    t = normalize_line_for_matching(token).strip(".")
    if not t:
        return False
    if t in INVALID_NAME_TOKENS:
        return False
    if len(t) < 2:
        return False
    if re.search(r"\d", t):
        return False
    return True


def looks_like_company_or_org(text: str) -> bool:
    s = normalize_line_for_matching(text)

    blacklist_tokens = [
        "srl", "s.r.l", "spa", "s.p.a", "snc", "sas", "s.a.s", "s.n.c",
        "societa", "società", "azienda", "impresa", "ente", "istituto",
        "associazione", "cooperativa", "consorzio", "fondazione",
        "academy", "training", "consulting",
        "centro studi", "studio", "servizi", "service",
        "a.s.q.&s", "asq&s", "asq", "safety", "quality",
        "via ", "viale ", "piazza ", "comune", "provincia", "c.f.", "p.iva"
    ]

    if any(tok in s for tok in blacklist_tokens):
        return True

    if "&" in text:
        return True

    if re.search(r"\b(srl|spa|snc|sas)\b", s):
        return True

    return False


def looks_like_role_or_label(text: str) -> bool:
    s = normalize_line_for_matching(text)

    bad_prefixes = [
        "qualifica",
        "mansione",
        "settore di riferimento",
        "codice ateco",
        "il corso",
        "data di",
        "durata",
        "ore",
        "modulo",
        "programma",
        "attestato",
        "corso",
        "rischio",
        "luogo",
        "docente",
        "il responsabile",
        "periodo di svolgimento",
    ]

    if any(s.startswith(p) for p in bad_prefixes):
        return True

    bad_contains = [
        "operaio",
        "impiegato",
        "preposto",
        "magazziniere",
        "autista",
        "responsabile",
        "amministratore",
        "verifica",
        "apprendimento",
        "test",
        "superamento",
        "esito",
        "durata",
        "ore",
        "codice ateco",
        "settore di riferimento",
    ]

    if any(tok in s for tok in bad_contains):
        return True

    return False


def is_plausible_person_name_line(line: str) -> bool:
    raw = clean_person_line(line)
    if not raw:
        return False

    # uppercase tipo "SAMIR BELFIORE"
    if raw.isupper():
        words = [w for w in raw.split() if w]
        if 2 <= len(words) <= 4 and not re.search(r"\d", raw):
            return True

    line_norm = normalize_line_for_matching(raw)

    if len(raw) < 4 or len(raw) > 80:
        return False

    if re.search(r"\d{2,}", raw):
        return False

    if re.search(r"[<>{}\[\]|_=+/*\\]", raw):
        return False

    forbidden = [
        "attestato", "corso", "durata", "ore",
        "data", "responsabile", "progetto", "modalita",
        "e-learning", "regolamenti", "tipologia",
        "ai sensi", "rilasciato", "conferito", "certifica",
        "formazione", "modulo", "programma", "giudizio",
        "idoneita", "nomina", "designazione", "verbale",
        "dpi", "azienda", "srl", "spa", "via ", "codice ateco",
        "settore di riferimento", "periodo di svolgimento"
    ]
    if any(tok in line_norm for tok in forbidden):
        return False

    words = [w for w in re.split(r"\s+", raw) if w]
    if len(words) < 2 or len(words) > 5:
        return False

    valid_words = 0
    for w in words:
        clean = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'’\-]", "", w)
        if is_valid_name_token(clean):
            valid_words += 1

    return valid_words >= 2


def split_name_line(line: str) -> Tuple[str, str]:
    line = clean_person_line(line)
    words = [w.strip(" ,.;:-") for w in line.split() if w.strip(" ,.;:-")]
    words = [w for w in words if is_valid_name_token(w)]

    if len(words) < 2:
        return "", ""

    # euristica: ultimo token = cognome, precedenti = nome
    if len(words) == 2:
        nome = words[0].title()
        cognome = words[1].title()
    else:
        nome = " ".join(words[:-1]).title()
        cognome = words[-1].title()

    if normalize_line_for_matching(nome) in INVALID_NAME_TOKENS:
        return "", ""
    if normalize_line_for_matching(cognome) in INVALID_NAME_TOKENS:
        return "", ""

    return nome, cognome


def validate_person_candidate(line: str) -> Tuple[bool, str]:
    raw = normalize_spaces(line)
    if not raw:
        return False, "vuoto"

    # uppercase tipo NOMINATIVO
    if raw.isupper():
        words = raw.split()
        if 2 <= len(words) <= 4 and not re.search(r"\d", raw):
            nome, cognome = split_name_line(raw)
            if nome and cognome:
                return True, "ok_uppercase"

    if looks_like_company_or_org(raw):
        return False, "sembra_azienda"

    if looks_like_role_or_label(raw):
        return False, "sembra_ruolo_o_label"

    if not is_plausible_person_name_line(raw):
        return False, "non_plausibile"

    nome, cognome = split_name_line(raw)
    if not nome or not cognome:
        return False, "split_non_valido"

    if len(nome) <= 1 or len(cognome) <= 1:
        return False, "troppo corto"

    full = f"{nome} {cognome}".strip()
    if looks_like_company_or_org(full):
        return False, "split_sembra_azienda"

    return True, "ok"


def score_name_candidate(line: str, idx: int) -> Tuple[int, List[str]]:
    reasons = []
    score = 0
    raw = clean_person_line(line)

    if raw.isupper():
        score += 4
        reasons.append("uppercase")

    words = raw.split()
    if 2 <= len(words) <= 4:
        score += 2
        reasons.append("word_count_ok")

    if idx < 12:
        score += 2
        reasons.append("top_of_document")

    if not looks_like_company_or_org(raw):
        score += 2
        reasons.append("not_company")

    if not looks_like_role_or_label(raw):
        score += 2
        reasons.append("not_role_label")

    if re.search(r"\d", raw):
        score -= 4
        reasons.append("contains_digits")

    return score, reasons


def extract_name_after_anchor(clean_text: str) -> Tuple[str, str, str]:
    anchor_pattern = "|".join(re.escape(a) for a in NAME_ANCHORS)

    # 1. stesso rigo: "conferito a SAMIR BELFIORE"
    m = re.search(
        rf"(?:{anchor_pattern})\s*[:\-]?\s*([A-Za-zÀ-ÖØ-öø-ÿ'’\-\s&\.]{{5,140}})",
        clean_text,
        re.IGNORECASE,
    )
    if m:
        raw = normalize_spaces(m.group(1))
        raw = re.split(
            r"\b(nato a|nata a|nato\/a a|nato il|nata il|data di nascita|qualifica|mansione|il corso|data di conclusione|data di svolgimento|attestato emesso|data emissione|giudizio|idoneita|idoneità|con la seguente qualifica|settore di riferimento|codice ateco|periodo di svolgimento)\b",
            raw,
            flags=re.IGNORECASE,
        )[0].strip(" ,.;:-")

        ok, _ = validate_person_candidate(raw)
        if ok:
            nome, cognome = split_name_line(raw)
            if nome and cognome:
                return nome, cognome, "anchor_regex_same_line"

    # 2. righi sotto anchor
    lines = [l.strip() for l in normalize_spaces(clean_text).splitlines() if l.strip()]
    for i, line in enumerate(lines):
        line_norm = normalize_line_for_matching(line)
        if any(anchor in line_norm for anchor in NAME_ANCHORS):
            for cand in lines[i + 1:i + 5]:
                ok, _ = validate_person_candidate(cand)
                if ok:
                    nome, cognome = split_name_line(cand)
                    if nome and cognome:
                        return nome, cognome, "anchor_lines_below"

    return "", "", ""


def extract_name_ai_style(text: str) -> Tuple[str, str, List[str]]:
    debug = []
    lines = [l.strip() for l in normalize_spaces(text).splitlines() if l.strip()]
    candidates: List[Dict[str, Any]] = []

    for idx, line in enumerate(lines[:30]):
        raw = clean_person_line(line)
        if len(raw) < 4 or len(raw) > 80:
            continue

        ok, reason = validate_person_candidate(raw)
        if not ok:
            debug.append(f"candidato scartato ai_style [{idx}]: {reason} -> {raw}")
            continue

        nome, cognome = split_name_line(raw)
        if not nome or not cognome:
            debug.append(f"candidato scartato ai_style [{idx}]: split_vuoto -> {raw}")
            continue

        score, reasons = score_name_candidate(raw, idx)
        candidates.append({
            "line": raw,
            "idx": idx,
            "score": score,
            "reasons": reasons,
            "nome": nome,
            "cognome": cognome,
        })

    if not candidates:
        return "", "", debug

    candidates.sort(key=lambda x: (x["score"], -x["idx"]), reverse=True)
    best = candidates[0]
    debug.append(
        f"best_candidate_ai_style: {best['line']} score={best['score']} reasons={','.join(best['reasons'])}"
    )
    return best["nome"], best["cognome"], debug


def extract_name_generic(text: str) -> Tuple[str, str, List[str]]:
    debug: List[str] = []
    clean_text = normalize_spaces(text)
    lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

    # 1. anchor forte
    nome, cognome, src = extract_name_after_anchor(clean_text)
    if nome and cognome:
        debug.append(f"nome trovato con priorità alta: {src}")
        return nome, cognome, debug

    # 2. prima di "nato/nata"
    for i, line in enumerate(lines):
        if re.search(r"\bnato\b|\bnata\b|\bnato\/a\b", line, re.IGNORECASE):
            prev_candidates = list(reversed(lines[max(0, i - 3):i]))
            for prev in prev_candidates:
                ok, reason = validate_person_candidate(prev)
                if ok:
                    nome, cognome = split_name_line(prev)
                    if nome and cognome:
                        debug.append("nome trovato prima di riga con nato/nata")
                        return nome, cognome, debug
                else:
                    debug.append(f"candidato scartato prima di nato/nata: {reason}")

    # 3. fallback layout/ai-style
    nome, cognome, dbg = extract_name_ai_style(clean_text)
    debug.extend(dbg)
    if nome and cognome:
        debug.append("nome trovato con fallback ai_style")
        return nome, cognome, debug

    debug.append("nome non trovato")
    return "", "", debug


# =========================================================
# OCR GOOGLE VISION
# =========================================================

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


def ocr_pdf_pages_separately(content: bytes) -> List[str]:
    page_images = pdf_to_page_images(content)
    texts = []

    for img_bytes in page_images:
        try:
            page_text = ocr_image_bytes(img_bytes)
            texts.append(remove_noise_lines(normalize_spaces(page_text)))
        except Exception:
            texts.append("")

    return texts


# =========================================================
# ESTRAZIONE TESTO FILE
# =========================================================

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


def extract_pdf_text_by_page(content: bytes) -> List[str]:
    pages_text = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        for page in doc:
            txt = page.get_text() or ""
            pages_text.append(remove_noise_lines(normalize_spaces(txt)))
        doc.close()
    except Exception:
        pages_text = []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return pages_text


def extract_text_from_file(filename: str, content: bytes, content_type: str) -> Dict[str, Any]:
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
        or ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic"]
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


# =========================================================
# ZONE TESTO
# =========================================================

def split_text_zones(text: str) -> Dict[str, str]:
    lines = [l.strip() for l in normalize_spaces(text).splitlines() if l.strip()]

    title_lines = lines[:15]
    identity_lines = lines[:12]
    body_lines = lines[15:] if len(lines) > 15 else []

    anchor_idx = -1
    for i, line in enumerate(lines[:50]):
        line_norm = normalize_line_for_matching(line)
        if any(anchor in line_norm for anchor in NAME_ANCHORS):
            anchor_idx = i
            break

    if anchor_idx >= 0:
        identity_lines = lines[max(0, anchor_idx - 2): min(len(lines), anchor_idx + 5)]

    return {
        "title_zone": normalize_spaces("\n".join(title_lines)),
        "identity_zone": normalize_spaces("\n".join(identity_lines)),
        "body_zone": normalize_spaces("\n".join(body_lines)),
        "full_text": normalize_spaces(text),
    }


# =========================================================
# CLASSIFICAZIONE DOCUMENTI
# =========================================================

def detect_nomina_strong(text: str, filename: str = "") -> bool:
    blob = normalize_text_for_matching(f"{filename}\n{text}")

    strong_nomina_phrases = [
        "lettera di nomina",
        "nomina",
        "designazione",
        "viene nominato",
        "viene designato",
        "si nomina",
        "si designa",
        "incarico di",
        "nomina ad addetto",
        "designazione ad addetto",
    ]

    role_phrases = [
        "primo soccorso",
        "antincendio",
        "preposto",
        "rspp",
        "r.l.s",
        "rls",
        "rappresentante dei lavoratori per la sicurezza",
        "responsabile del servizio di prevenzione e protezione",
    ]

    neg_attestato_phrases = [
        "attestato di frequenza",
        "attestato di formazione",
        "attestato di partecipazione",
        "verifica dell'apprendimento",
        "ha frequentato",
        "ha partecipato",
        "programma del corso",
        "programma corso",
        "corso di aggiornamento",
        "data di svolgimento del corso",
        "data svolgimento del corso",
    ]

    strong_hits = sum(1 for p in strong_nomina_phrases if p in blob)
    role_hits = sum(1 for p in role_phrases if p in blob)
    attestato_hits = sum(1 for p in neg_attestato_phrases if p in blob)

    if strong_hits >= 1 and role_hits >= 1 and attestato_hits == 0:
        return True

    if strong_hits >= 2 and role_hits >= 1 and attestato_hits == 0:
        return True

    return False


def score_category(text: str, filename: str) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    blob = normalize_text_for_matching(f"{filename}\n{text}")
    zones = split_text_zones(text)
    title_blob = normalize_text_for_matching(f"{filename}\n{zones.get('title_zone', '')}")
    identity_blob = normalize_text_for_matching(zones.get("identity_zone", ""))

    # BLOCCO PRIORITARIO NOMINE
    if detect_nomina_strong(text, filename):
        scores = {
            "attestati": 0,
            "nomine": 12,
            "visite_mediche": 0,
            "verbali_dpi": 0,
            "documenti_aziendali": 0,
        }
        meta = {
            "best_category": "nomine",
            "best_score": 12,
            "second_category": "attestati",
            "second_score": 0,
            "delta": 12,
            "classification_debug": {
                "forced": "nomina_strong_detection"
            },
        }
        return "nomine", scores, meta

    scores = {
        "attestati": 0,
        "nomine": 0,
        "visite_mediche": 0,
        "verbali_dpi": 0,
        "documenti_aziendali": 0,
    }

    debug = {
        "positive_hits": [],
        "negative_hits": [],
    }

    # =====================================================
    # ATTESTATI
    # =====================================================
    if "attestato" in title_blob:
        scores["attestati"] += 8
        debug["positive_hits"].append("attestati:+8 titolo contiene attestato")
    elif "attestato" in blob:
        scores["attestati"] += 3
        debug["positive_hits"].append("attestati:+3 testo contiene attestato")

    attestato_title_hits = count_keywords(title_blob, ATTESTATO_POSITIVE_SIGNALS)
    attestato_full_hits = count_keywords(blob, ATTESTATO_POSITIVE_SIGNALS)

    if attestato_title_hits >= 1:
        scores["attestati"] += attestato_title_hits * 3
        debug["positive_hits"].append(f"attestati:+{attestato_title_hits * 3} segnali attestato nel titolo")

    if attestato_full_hits >= 2:
        scores["attestati"] += 2
        debug["positive_hits"].append("attestati:+2 conferma dal corpo")

    if "conferito a" in blob or "rilasciato a" in blob:
        scores["attestati"] += 2
        debug["positive_hits"].append("attestati:+2 struttura tipica attestato")
        
    if "programma corso" in title_blob or "programma del corso" in title_blob:
        scores["attestati"] += 6
        debug["positive_hits"].append("attestati:+6 titolo da programma corso allegato")
    
    # =====================================================
    # NOMINE
    # =====================================================
    if "nomina" in title_blob:
        scores["nomine"] += 8
        debug["positive_hits"].append("nomine:+8 titolo contiene nomina")
    elif "nomina" in blob:
        scores["nomine"] += 4
        debug["positive_hits"].append("nomine:+4 testo contiene nomina")

    if "designazione" in title_blob:
        scores["nomine"] += 6
        debug["positive_hits"].append("nomine:+6 designazione nel titolo")
    elif "designazione" in blob:
        scores["nomine"] += 3

    if "lettera di nomina" in blob:
        scores["nomine"] += 5

    if "viene nominato" in blob or "viene designato" in blob or "si nomina" in blob:
        scores["nomine"] += 4
        debug["positive_hits"].append("nomine:+4 formula tipica nomina")

    nomina_structure_present = any(
    x in blob for x in [
        "nomina",
        "designazione",
        "lettera di nomina",
        "viene nominato",
        "viene designato",
        "si nomina",
        "si designa",
        "incarico di",
        "nomina ad addetto",
        "designazione ad addetto",
    ]
)

for _, kws in NOMINA_ROLE_KEYWORDS.items():
    if has_any_keyword(title_blob, kws) and nomina_structure_present:
        scores["nomine"] += 3
    elif has_any_keyword(identity_blob, kws) and nomina_structure_present:
        scores["nomine"] += 2
    elif has_any_keyword(blob, kws) and nomina_structure_present:
        scores["nomine"] += 1

    # =====================================================
    # VISITE MEDICHE
    # =====================================================
    if "giudizio di idoneita" in title_blob or "giudizio di idoneità" in title_blob:
        scores["visite_mediche"] += 8
        debug["positive_hits"].append("visite:+8 giudizio nel titolo")
    elif "giudizio di idoneita" in blob or "giudizio di idoneità" in blob:
        scores["visite_mediche"] += 5
        debug["positive_hits"].append("visite:+5 giudizio nel testo")

    if "medico competente" in blob:
        scores["visite_mediche"] += 3
    if "sorveglianza sanitaria" in blob:
        scores["visite_mediche"] += 3
    if "prescrizioni" in blob:
        scores["visite_mediche"] += 2
    if "idoneo" in blob or "idonea" in blob:
        scores["visite_mediche"] += 2

    # =====================================================
    # VERBALI DPI
    # =====================================================
    dpi_score = 0

    if "verbale di consegna" in title_blob:
        dpi_score += 8
        debug["positive_hits"].append("dpi:+8 verbale di consegna nel titolo")
    elif "verbale di consegna" in blob:
        dpi_score += 4
        debug["positive_hits"].append("dpi:+4 verbale di consegna nel testo")

    if "consegna dpi" in title_blob:
        dpi_score += 7
        debug["positive_hits"].append("dpi:+7 consegna dpi nel titolo")
    elif "consegna dpi" in blob:
        dpi_score += 4

    if "firma per ricevuta" in blob:
        dpi_score += 4
    if "il lavoratore dichiara di aver ricevuto" in blob:
        dpi_score += 5
    if "dispositivi di protezione individuale" in title_blob:
        dpi_score += 5
    if re.search(r"\bdpi\b", title_blob):
        dpi_score += 2

    scores["verbali_dpi"] = dpi_score

    # =====================================================
    # DOCUMENTI AZIENDALI
    # =====================================================
    doc_score = 0

    if re.search(r"\bdvr\b", title_blob):
        doc_score += 9
        debug["positive_hits"].append("doc_az:+9 DVR nel titolo")
    elif re.search(r"\bdvr\b", blob):
        doc_score += 4

    if "valutazione dei rischi" in title_blob:
        doc_score += 9
        debug["positive_hits"].append("doc_az:+9 valutazione rischi nel titolo")

    if re.search(r"\bpos\b", title_blob):
        doc_score += 8
    elif re.search(r"\bpos\b", blob):
        doc_score += 3

    if re.search(r"\bpsc\b", title_blob):
        doc_score += 8
    elif re.search(r"\bpsc\b", blob):
        doc_score += 3

    if "organigramma" in title_blob:
        doc_score += 7
    if "procedura" in title_blob:
        doc_score += 6
    if "protocollo" in title_blob:
        doc_score += 5

    scores["documenti_aziendali"] = doc_score

    # =====================================================
    # PENALITÀ INCROCIATE
    # =====================================================
    for neg in ATTESTATO_NEGATIVE_SIGNALS:
        if neg in blob:
            scores["attestati"] -= 2
            debug["negative_hits"].append(f"attestati:-2 presenza '{neg}'")

    # Se compaiono parole forti da nomina, abbassa molto attestati
    if "nomina" in blob or "designazione" in blob or "viene nominato" in blob or "viene designato" in blob:
        scores["attestati"] -= 6
        debug["negative_hits"].append("attestati:-6 segnali forti da nomina")

    if "attestato" in title_blob:
        scores["verbali_dpi"] = max(0, scores["verbali_dpi"] - 4)
        scores["documenti_aziendali"] = max(0, scores["documenti_aziendali"] - 5)
        debug["negative_hits"].append("dpi:-4 titolo da attestato")
        debug["negative_hits"].append("doc_az:-5 titolo da attestato")

    if any(x in title_blob for x in ["dvr", "valutazione dei rischi", "pos", "psc"]):
        scores["attestati"] = max(0, scores["attestati"] - 5)
        debug["negative_hits"].append("attestati:-5 titolo da documento aziendale")

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_category, best_score = ordered[0]
    second_category, second_score = ordered[1]
    delta = best_score - second_score

    meta = {
        "best_category": best_category,
        "best_score": best_score,
        "second_category": second_category,
        "second_score": second_score,
        "delta": delta,
        "classification_debug": debug,
    }

    if best_score < 4:
        return "altri_da_verificare", scores, meta

    if delta <= 1:
        meta["classification_warning"] = "classificazione_ambigua"
        return "altri_da_verificare", scores, meta

    return best_category, scores, meta



# =========================================================
# DETECTION CORSO ATTESTATI
# =========================================================
def score_course_family_by_zone(zones: Dict[str, str], filename: str) -> Tuple[str, str, str, Dict[str, int], List[str]]:
    title_blob = normalize_text_for_matching(f"{filename}\n{zones.get('title_zone', '')}")
    identity_blob = normalize_text_for_matching(zones.get("identity_zone", ""))
    body_blob = normalize_text_for_matching(zones.get("body_zone", ""))
    full_blob = normalize_text_for_matching(zones.get("full_text", ""))

    debug = []
    scores: Dict[str, int] = {
        "FORMAZIONE_GENERALE": 0,
        "FORMAZIONE_SPECIFICA": 0,
        "AGGIORNAMENTO_FORMAZIONE_LAVORATORI": 0,
        "PRIMO_SOCCORSO": 0,
        "ANTINCENDIO": 0,
        "PREPOSTO": 0,
        "RLS": 0,
        "RSPP_DL": 0,
        "HACCP": 0,
        "PONTEGGI": 0,
        "CARRELLISTA": 0,
        "PLE": 0,
        "LAVORI_IN_QUOTA": 0,
        "CORSO_NON_RICONOSCIUTO": 0,
    }

    if detect_nomina_strong(zones.get("full_text", ""), filename):
        debug.append("blocco parser corsi: testo riconosciuto come nomina")
        return "CORSO_NON_RICONOSCIUTO", "base", "", scores, debug

    is_update = any(w in title_blob for w in UPDATE_WORDS) or (
        "aggiornamento" in full_blob and "formazione lavoratori" in full_blob
    )

        specific_families = [
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
    ]

    for family in specific_families:
        kws = SPECIFIC_COURSE_KEYWORDS[family]

        title_hits = count_keywords(title_blob, kws)
        identity_hits = count_keywords(identity_blob, kws)
        body_hits_raw = count_keywords(body_blob, kws)

        body_hits = 1 if body_hits_raw >= 3 else 0

        scores[family] += title_hits * 8
        scores[family] += identity_hits * 2
        scores[family] += body_hits

        if title_hits:
            debug.append(f"{family}: +{title_hits * 8} match titolo")
        if identity_hits:
            debug.append(f"{family}: +{identity_hits * 2} match zona nominativo")
        if body_hits:
            debug.append(f"{family}: +{body_hits} match corpo")

    # =====================================================
    # BOOST ESPLICITI SU TITOLO (corsi molto riconoscibili)
    # =====================================================

    # PLE
    if (
        "ple" in title_blob
        and (
            "uso di ple" in title_blob
            or "utilizzo di ple" in title_blob
            or "con o senza stabilizzatori" in title_blob
            or "addetti all'uso di ple" in title_blob
            or "addetti all’uso di ple" in title_blob
        )
    ):
        scores["PLE"] += 10
        debug.append("PLE: +10 boost titolo esplicito uso di PLE con/senza stabilizzatori")

    # CARRELLISTA
    if (
        "carrello elevatore" in title_blob
        or "carrelli elevatori" in title_blob
        or "muletto" in title_blob
        or "mulettista" in title_blob
        or "conduzione di carrelli elevatori" in title_blob
    ):
        scores["CARRELLISTA"] += 10
        debug.append("CARRELLISTA: +10 boost titolo esplicito")

    # ANTINCENDIO
    if "antincendio" in title_blob:
        scores["ANTINCENDIO"] += 8
        debug.append("ANTINCENDIO: +8 boost titolo esplicito")

        if "aggiornamento" in title_blob:
            scores["ANTINCENDIO"] += 4
            debug.append("ANTINCENDIO: +4 aggiornamento nel titolo")

    # PRIMO SOCCORSO
    if "primo soccorso" in title_blob:
        scores["PRIMO_SOCCORSO"] += 8
        debug.append("PRIMO_SOCCORSO: +8 boost titolo esplicito")

        if "aggiornamento" in title_blob:
            scores["PRIMO_SOCCORSO"] += 4
            debug.append("PRIMO_SOCCORSO: +4 aggiornamento nel titolo")

    # PREPOSTO
    if "preposto" in title_blob:
        scores["PREPOSTO"] += 8
        debug.append("PREPOSTO: +8 boost titolo esplicito")

        if "aggiornamento" in title_blob:
            scores["PREPOSTO"] += 4
            debug.append("PREPOSTO: +4 aggiornamento nel titolo")

    # RLS
    if "rls" in title_blob or "rappresentante dei lavoratori per la sicurezza" in title_blob:
        scores["RLS"] += 8
        debug.append("RLS: +8 boost titolo esplicito")

        if "aggiornamento" in title_blob:
            scores["RLS"] += 4
            debug.append("RLS: +4 aggiornamento nel titolo")

    # RSPP DATORE DI LAVORO
    if (
        "rspp datore di lavoro" in title_blob
        or "datore di lavoro rspp" in title_blob
        or "datore di lavoro che svolge i compiti del servizio di prevenzione e protezione" in title_blob
    ):
        scores["RSPP_DL"] += 10
        debug.append("RSPP_DL: +10 boost titolo esplicito")

    # PONTEGGI
    if (
        "ponteggi" in title_blob
        or "montaggio smontaggio trasformazione ponteggi" in title_blob
        or "pimus" in title_blob
        or "pi.m.u.s" in title_blob
    ):
        scores["PONTEGGI"] += 10
        debug.append("PONTEGGI: +10 boost titolo esplicito")

    # LAVORI IN QUOTA
    if "lavori in quota" in title_blob or "anticaduta" in title_blob:
        scores["LAVORI_IN_QUOTA"] += 10
        debug.append("LAVORI_IN_QUOTA: +10 boost titolo esplicito")

    # HACCP
    if (
        "haccp" in title_blob
        or "igiene alimentare" in title_blob
        or "igiene degli alimenti" in title_blob
        or "alimentarista" in title_blob
        or "alimentaristi" in title_blob
    ):
        scores["HACCP"] += 10
        debug.append("HACCP: +10 boost titolo esplicito")

    # Boost specifico antincendio già utile sui tuoi casi reali
    if "aggiornamento" in title_blob and "antincendio" in title_blob:
        scores["ANTINCENDIO"] += 10
        debug.append("ANTINCENDIO: +10 boost aggiornamento+antincendio nel titolo")

    fg_title = count_keywords(title_blob, GENERAL_TRAINING_KEYWORDS["FORMAZIONE_GENERALE"])
    fs_title = count_keywords(title_blob, GENERAL_TRAINING_KEYWORDS["FORMAZIONE_SPECIFICA"])
    fl_title = count_keywords(title_blob, GENERIC_WORKER_TRAINING_PATTERNS)

    fg_full = count_keywords(full_blob, GENERAL_TRAINING_KEYWORDS["FORMAZIONE_GENERALE"])
    fs_full = count_keywords(full_blob, GENERAL_TRAINING_KEYWORDS["FORMAZIONE_SPECIFICA"])
    fl_full = count_keywords(full_blob, GENERIC_WORKER_TRAINING_PATTERNS)

    scores["FORMAZIONE_GENERALE"] += fg_title * 8
    scores["FORMAZIONE_SPECIFICA"] += fs_title * 8
    scores["AGGIORNAMENTO_FORMAZIONE_LAVORATORI"] += fl_title * 8

    scores["FORMAZIONE_GENERALE"] += fg_full * 1
    scores["FORMAZIONE_SPECIFICA"] += fs_full * 1
    scores["AGGIORNAMENTO_FORMAZIONE_LAVORATORI"] += fl_full * 1

    if fg_title:
        debug.append(f"FORMAZIONE_GENERALE: +{fg_title * 8} match titolo")
    if fs_title:
        debug.append(f"FORMAZIONE_SPECIFICA: +{fs_title * 8} match titolo")
    if fl_title:
        debug.append(f"AGGIORNAMENTO_FORMAZIONE_LAVORATORI: +{fl_title * 8} match titolo")

    if "formazione generale" in title_blob:
        scores["FORMAZIONE_GENERALE"] += 4
        debug.append("FORMAZIONE_GENERALE: +4 boost titolo esplicito")

    if "formazione specifica" in title_blob:
        scores["FORMAZIONE_SPECIFICA"] += 4
        debug.append("FORMAZIONE_SPECIFICA: +4 boost titolo esplicito")

    if "rischio alto" in title_blob:
        scores["FORMAZIONE_SPECIFICA"] += 4
        debug.append("FORMAZIONE_SPECIFICA: +4 rischio alto nel titolo")
    elif "rischio medio" in title_blob:
        scores["FORMAZIONE_SPECIFICA"] += 3
        debug.append("FORMAZIONE_SPECIFICA: +3 rischio medio nel titolo")
    elif "rischio basso" in title_blob:
        scores["FORMAZIONE_SPECIFICA"] += 3
        debug.append("FORMAZIONE_SPECIFICA: +3 rischio basso nel titolo")

    has_worker_training_title = (
        fl_title > 0
        or fg_title > 0
        or fs_title > 0
        or "formazione lavoratori" in title_blob
        or "formazione dei lavoratori" in title_blob
    )

    fl_score = (
        scores["FORMAZIONE_GENERALE"] +
        scores["FORMAZIONE_SPECIFICA"] +
        scores["AGGIORNAMENTO_FORMAZIONE_LAVORATORI"]
    )

    best_specific_family = max(specific_families, key=lambda k: scores[k])
    best_specific_score = scores[best_specific_family]

    debug.append(f"best_specific_family={best_specific_family}")
    debug.append(f"best_specific_score={best_specific_score}")
    debug.append(f"fl_score={fl_score}")

    if best_specific_score >= 10 and best_specific_score >= fl_score + 2:
        debug.append(
            f"famiglia specifica dominante forzata: "
            f"{best_specific_family} ({best_specific_score} vs fl_score={fl_score})"
        )
        return best_specific_family, "aggiornamento" if is_update else "base", "", scores, debug

    if has_worker_training_title and fl_score >= 8 and fl_score >= best_specific_score:
        modulo = ""

        if (
            scores["AGGIORNAMENTO_FORMAZIONE_LAVORATORI"] >= max(
                scores["FORMAZIONE_GENERALE"],
                scores["FORMAZIONE_SPECIFICA"]
            )
            and is_update
        ):
            modulo = "aggiornamento"
            tipo = "aggiornamento"
        elif scores["FORMAZIONE_GENERALE"] >= scores["FORMAZIONE_SPECIFICA"]:
            modulo = "generale"
            tipo = "base"
        else:
            if "rischio alto" in title_blob:
                modulo = "specifica_rischio_alto"
            elif "rischio medio" in title_blob:
                modulo = "specifica_rischio_medio"
            elif "rischio basso" in title_blob:
                modulo = "specifica_rischio_basso"
            else:
                modulo = "specifica"
            tipo = "base"

        debug.append(
            f"famiglia unificata FORMAZIONE_LAVORATORI "
            f"(fl_score={fl_score}, best_specific={best_specific_family}:{best_specific_score}, modulo={modulo}, tipo={tipo})"
        )
        return "FORMAZIONE_LAVORATORI", tipo, modulo, scores, debug

    if scores["RLS"] >= 8 and scores["RLS"] >= fl_score + 2:
        debug.append("RLS forzato come aggiornamento")
        return "RLS", "aggiornamento", "", scores, debug

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_family, best_score = ordered[0]
    second_family, second_score = ordered[1]

    debug.append(f"ordered_best={best_family}:{best_score}")
    debug.append(f"ordered_second={second_family}:{second_score}")

    if best_score < 8:
        debug.append("famiglia corso incerta: best_score troppo basso")
        return "CORSO_NON_RICONOSCIUTO", "aggiornamento" if is_update else "base", "", scores, debug

    if (best_score - second_score) <= 2:
        debug.append(
            f"famiglia corso incerta: delta troppo basso ({best_family}:{best_score} vs {second_family}:{second_score})"
        )
        return "CORSO_NON_RICONOSCIUTO", "aggiornamento" if is_update else "base", "", scores, debug

    debug.append(f"famiglia scelta: {best_family} ({best_score} vs {second_score})")
    return best_family, "aggiornamento" if is_update else "base", "", scores, debug



# =========================================================
# DATE PESATE
# =========================================================
def build_date_candidates(text: str) -> List[Dict[str, Any]]:
    lines = [l.strip() for l in normalize_spaces(text).splitlines() if l.strip()]
    candidates: List[Dict[str, Any]] = []

    for i, line in enumerate(lines):
        found_numeric = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", line)
        found_textual = re.findall(
            r"\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}\b",
            normalize_text_for_matching(line),
            flags=re.IGNORECASE,
        )

        found = list(found_numeric) + list(found_textual)

        if not found:
            continue

        context_window = " ".join(lines[max(0, i - 2): min(len(lines), i + 3)])
        context_norm = normalize_text_for_matching(context_window)
        line_norm = normalize_text_for_matching(line)

        for raw_date in found:
            dt = parse_date(raw_date)
            if not dt:
                continue

            score = 0
            reasons = []

            if any(lbl in context_norm for lbl in DATE_STRONG_LABELS):
                score += 15
                reasons.append("strong_label")

            if any(lbl in context_norm for lbl in DATE_WEAK_PERIOD_LABELS):
                score += 5
                reasons.append("period_label")

            if any(neg in context_norm for neg in DATE_NEGATIVE_CONTEXTS):
                score -= 20
                reasons.append("negative_context")

            # bonus forte su formule tipiche del corso
            if "svolto in data" in line_norm:
                score += 18
                reasons.append("svolto_in_data_line")

            if "perugia" in line_norm and dt.year >= 2000:
                score += 8
                reasons.append("place_and_date_line")

            if 1990 <= dt.year <= datetime.now().year + 1:
                score += 1
                reasons.append("year_plausible")
            else:
                score -= 10
                reasons.append("year_implausible")

            candidates.append({
                "raw": raw_date,
                "date": dt,
                "line_index": i,
                "line": line[:200],
                "context": context_window[:400],
                "score": score,
                "reasons": reasons,
            })

    return candidates


def extract_conclusion_date(text: str) -> Tuple[Optional[datetime], List[str], str, List[Dict[str, Any]]]:
    debug = []
    candidates = build_date_candidates(text)

    if not candidates:
        debug.append("nessuna data trovata")
        return None, debug, "none", []

    birth_date = extract_birth_date(text)
    filtered = []

    for c in candidates:
        dt = c["date"]
        if birth_date and dt.date() == birth_date.date():
            c["score"] -= 25
            c["reasons"].append("birth_date_penalty")
        filtered.append(c)

    ordered_all = sorted(filtered, key=lambda x: (x["score"], x["date"]), reverse=True)

    strong_candidates = [
        c for c in ordered_all
        if "strong_label" in c["reasons"]
        and "negative_context" not in c["reasons"]
        and c["score"] >= 10
    ]
    if strong_candidates:
        best = strong_candidates[0]
        debug.append(f"data conclusione scelta da strong_label: {best['raw']}")
        return best["date"], debug, "strong_score", ordered_all

    period_block_match = re.search(
        r"(giorni|periodo di svolgimento del corso|svolgimento del corso|dal)(.+?)(data emissione|attestato emesso|programma del corso|il responsabile|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if period_block_match:
        block = period_block_match.group(0)
        period_dates = extract_dates(block)

        valid_period_dates = []
        for dt in period_dates:
            if 1990 <= dt.year <= datetime.now().year + 1:
                valid_period_dates.append(dt)

        if valid_period_dates:
            dt = max(valid_period_dates)
            debug.append("data conclusione scelta come ultima data valida di blocco periodo")
            return dt, debug, "period_block", ordered_all

    positive = [
        c for c in ordered_all
        if c["score"] >= 1 and "negative_context" not in c["reasons"]
    ]
    if positive:
        best = positive[0]
        debug.append(f"data conclusione scelta con fallback positivo: {best['raw']}")
        return best["date"], debug, "weak_positive", ordered_all

    debug.append("nessuna data conclusione affidabile")
    return None, debug, "none", ordered_all


# =========================================================
# HELPERS FILENAME / SCADENZA
# =========================================================

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


def build_nomina_filename(cognome: str, nome: str, ruolo: str, original_filename: str) -> str:
    base_name = f"{cognome}_{nome}_NOMINA_{ruolo}".strip("_")
    if base_name in {"", "NOMINA"}:
        base_name = os.path.splitext(original_filename)[0]
    return safe_filename(base_name) + ".pdf"


def build_dpi_filename(cognome: str, nome: str, original_filename: str) -> str:
    base_name = f"{cognome}_{nome}_VERBALE_DPI".strip("_")
    if base_name in {"", "VERBALE_DPI"}:
        base_name = os.path.splitext(original_filename)[0]
    return safe_filename(base_name) + ".pdf"


def build_visita_filename(cognome: str, nome: str, original_filename: str) -> str:
    base_name = f"{cognome}_{nome}_VISITA_MEDICA".strip("_")
    if base_name in {"", "VISITA_MEDICA"}:
        base_name = os.path.splitext(original_filename)[0]
    return safe_filename(base_name) + ".pdf"


# =========================================================
# PARSER ATTESTATI
# =========================================================

def compute_attestato_confidenza(
    nome: str,
    cognome: str,
    course_family: str,
    conclusion_date: Optional[datetime],
    date_source: str,
    course_scores: Dict[str, int],
) -> str:
    points = 0

    if nome and cognome:
        points += 3
    if course_family != "CORSO_NON_RICONOSCIUTO":
        points += 3
    if conclusion_date:
        points += 2

    if date_source == "strong_score":
        points += 2
    elif date_source == "period_block":
        points += 2

    ordered = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)
    if len(ordered) >= 2 and (ordered[0][1] - ordered[1][1]) >= 3:
        points += 1

    if points >= 8:
        return "alta"
    if points >= 5:
        return "media"
    return "bassa"


def compute_review_flag_attestato(
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


def parse_attestato(text: str, filename: str) -> Dict[str, Any]:
    debug_notes = []
    zones = split_text_zones(text)

    nome, cognome, name_debug = extract_name_generic(text)
    debug_notes.extend(name_debug)

    family, tipo_percorso, modulo_fl, course_scores, course_debug = score_course_family_by_zone(zones, filename)
    debug_notes.extend(course_debug)

    conclusion_date, date_debug, date_source, date_candidates = extract_conclusion_date(text)
    debug_notes.extend(date_debug)

    scad_label, scad_value = compute_scadenza(family, conclusion_date)
    confidenza = compute_attestato_confidenza(nome, cognome, family, conclusion_date, date_source, course_scores)

    suggested_name = build_attestato_filename(
        cognome.upper() if cognome else "",
        nome.upper() if nome else "",
        family,
        filename,
    )

    needs_review, review_reasons = compute_review_flag_attestato(
        nome=nome,
        cognome=cognome,
        course_family=family,
        conclusion_date=conclusion_date,
        confidenza=confidenza,
    )

    top_dates = []
    for cand in date_candidates[:5]:
        top_dates.append({
            "raw": cand["raw"],
            "score": cand["score"],
            "reasons": cand["reasons"],
            "line": cand["line"],
        })

    ordered_course_scores = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "categoria": "Attestato",
        "nome": nome,
        "cognome": cognome,
        "corso": family,
        "famiglia_corso": family,
        "tipo_percorso": tipo_percorso,
        "modulo_formazione_lavoratori": modulo_fl,
        "data_conclusione": format_date(conclusion_date),
        "data_scadenza": scad_value if scad_label == "data_scadenza" else "",
        "prossimo_aggiornamento": scad_value if scad_label == "prossimo_aggiornamento" else "",
        "scadenza_label": scad_label,
        "confidenza": confidenza,
        "suggested_filename": suggested_name,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "parser_debug": unique_preserve_order(debug_notes),
        "course_score_details": dict(ordered_course_scores),
        "date_candidates_top": top_dates,
        "date_source": date_source,
    }


# =========================================================
# PARSER NOMINE
# =========================================================

def detect_nomina_role(text: str) -> str:
    blob = normalize_text_for_matching(text)
    for role, kws in NOMINA_ROLE_KEYWORDS.items():
        if has_any_keyword(blob, kws):
            return role
    return "NOMINA_NON_RICONOSCIUTA"


def extract_document_date(text: str) -> Tuple[Optional[datetime], str]:
    candidates = build_date_candidates(text)
    ordered = sorted(candidates, key=lambda x: (x["score"], x["date"]), reverse=True)
    if ordered:
        best = ordered[0]
        if best["score"] >= 1:
            return best["date"], "scored"
    return None, "none"


def parse_nomina(text: str, filename: str) -> Dict[str, Any]:
    debug = []

    nome, cognome, name_debug = extract_name_generic(text)
    debug.extend(name_debug)

    ruolo = detect_nomina_role(text)
    if ruolo != "NOMINA_NON_RICONOSCIUTA":
        debug.append(f"ruolo nomina riconosciuto: {ruolo}")
    else:
        debug.append("ruolo nomina non riconosciuto")

    doc_date, date_src = extract_document_date(text)
    if doc_date:
        debug.append(f"data documento trovata: {date_src}")

    confidenza = "bassa"
    if nome and cognome and ruolo != "NOMINA_NON_RICONOSCIUTA":
        confidenza = "alta"
    elif ruolo != "NOMINA_NON_RICONOSCIUTA" or doc_date:
        confidenza = "media"

    needs_review = False
    review_reasons = []

    if not nome or not cognome:
        needs_review = True
        review_reasons.append("nome_o_cognome_non_estratti")
    if ruolo == "NOMINA_NON_RICONOSCIUTA":
        needs_review = True
        review_reasons.append("ruolo_nomina_non_riconosciuto")

    return {
        "categoria": "Nomina",
        "nome": nome,
        "cognome": cognome,
        "corso": "",
        "famiglia_corso": "",
        "tipo_percorso": "",
        "modulo_formazione_lavoratori": "",
        "data_conclusione": format_date(doc_date),
        "data_scadenza": "",
        "prossimo_aggiornamento": "",
        "scadenza_label": "",
        "ruolo": ruolo,
        "confidenza": confidenza,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "suggested_filename": build_nomina_filename(
            cognome.upper() if cognome else "",
            nome.upper() if nome else "",
            ruolo,
            filename,
        ),
        "parser_debug": unique_preserve_order(debug),
    }


# =========================================================
# PARSER VERBALI DPI
# =========================================================

def extract_dpi_reference(text: str) -> str:
    lines = [l.strip() for l in normalize_spaces(text).splitlines() if l.strip()]
    candidates = []
    for line in lines:
        norm = normalize_line_for_matching(line)
        if "dpi" in norm or "dispositivi di protezione individuale" in norm:
            candidates.append(line)
    if candidates:
        return candidates[0][:120]
    return ""


def parse_verbale_dpi(text: str, filename: str) -> Dict[str, Any]:
    debug = []

    nome, cognome, name_debug = extract_name_generic(text)
    debug.extend(name_debug)

    doc_date, date_src = extract_document_date(text)
    if doc_date:
        debug.append(f"data documento trovata: {date_src}")

    dpi_ref = extract_dpi_reference(text)
    if dpi_ref:
        debug.append("riferimento dpi trovato")

    confidenza = "bassa"
    if nome and cognome and dpi_ref:
        confidenza = "alta"
    elif dpi_ref or doc_date:
        confidenza = "media"

    needs_review = False
    review_reasons = []
    if not dpi_ref:
        needs_review = True
        review_reasons.append("riferimento_dpi_non_trovato")
    if not nome or not cognome:
        needs_review = True
        review_reasons.append("nome_o_cognome_non_estratti")

    return {
        "categoria": "Verbale DPI",
        "nome": nome,
        "cognome": cognome,
        "corso": "",
        "famiglia_corso": "",
        "tipo_percorso": "",
        "modulo_formazione_lavoratori": "",
        "data_conclusione": format_date(doc_date),
        "data_scadenza": "",
        "prossimo_aggiornamento": "",
        "scadenza_label": "",
        "riferimento_dpi": dpi_ref,
        "confidenza": confidenza,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "suggested_filename": build_dpi_filename(
            cognome.upper() if cognome else "",
            nome.upper() if nome else "",
            filename,
        ),
        "parser_debug": unique_preserve_order(debug),
    }


# =========================================================
# PARSER VISITE MEDICHE
# =========================================================

def extract_visit_esito(text: str) -> str:
    blob = normalize_text_for_matching(text)
    ordered = sorted(VISITA_ESITI, key=lambda x: len(x), reverse=True)
    for e in ordered:
        if normalize_text_for_matching(e) in blob:
            return e.upper()
    return ""


def parse_visita_medica(text: str, filename: str) -> Dict[str, Any]:
    debug = []

    nome, cognome, name_debug = extract_name_generic(text)
    debug.extend(name_debug)

    doc_date, date_src = extract_document_date(text)
    if doc_date:
        debug.append(f"data visita trovata: {date_src}")

    esito = extract_visit_esito(text)
    if esito:
        debug.append(f"esito trovato: {esito}")

    confidenza = "bassa"
    if nome and cognome and esito:
        confidenza = "alta"
    elif esito or doc_date:
        confidenza = "media"

    needs_review = False
    review_reasons = []
    if not nome or not cognome:
        needs_review = True
        review_reasons.append("nome_o_cognome_non_estratti")
    if not esito:
        needs_review = True
        review_reasons.append("esito_non_trovato")

    return {
        "categoria": "Visita Medica",
        "nome": nome,
        "cognome": cognome,
        "corso": "",
        "famiglia_corso": "",
        "tipo_percorso": "",
        "modulo_formazione_lavoratori": "",
        "data_conclusione": format_date(doc_date),
        "data_scadenza": "",
        "prossimo_aggiornamento": "",
        "scadenza_label": "",
        "esito": esito,
        "confidenza": confidenza,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "suggested_filename": build_visita_filename(
            cognome.upper() if cognome else "",
            nome.upper() if nome else "",
            filename,
        ),
        "parser_debug": unique_preserve_order(debug),
    }


# =========================================================
# PARSER GENERICO
# =========================================================

def parse_documento_generico(text: str, filename: str, categoria_label: str) -> Dict[str, Any]:
    confidenza = "media" if categoria_label != "Da verificare" else "bassa"
    needs_review = categoria_label == "Da verificare"
    review_reasons = ["classificazione_incerta"] if needs_review else []

    ext = os.path.splitext(filename)[1] or ".pdf"

    return {
        "categoria": categoria_label,
        "nome": "",
        "cognome": "",
        "corso": "",
        "famiglia_corso": "",
        "tipo_percorso": "",
        "modulo_formazione_lavoratori": "",
        "data_conclusione": "",
        "data_scadenza": "",
        "prossimo_aggiornamento": "",
        "scadenza_label": "",
        "confidenza": confidenza,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "suggested_filename": safe_filename(os.path.splitext(filename)[0]) + ext,
        "parser_debug": ["parser generico"],
    }


# =========================================================
# PDF MISTI / COMPOSTI
# =========================================================

def detect_mixed_pdf_categories(filename: str, content: bytes, content_type: str) -> Dict[str, Any]:
    ext = os.path.splitext(filename.lower())[1]
    is_pdf = content_type == "application/pdf" or ext == ".pdf"

    result = {
        "is_mixed": False,
        "page_categories": [],
        "dominant_category": None,
        "distinct_categories": [],
        "debug": [],
    }

    if not is_pdf:
        return result

    pages_text = extract_pdf_text_by_page(content)

    usable_pages = [p for p in pages_text if p and len(p.strip()) >= 20]
    if not usable_pages:
        pages_text = ocr_pdf_pages_separately(content)

    if not pages_text:
        return result

    page_categories = []
    debug_rows = []
    first_page_category = None

    for idx, page_text in enumerate(pages_text, start=1):
        if not page_text or len(page_text.strip()) < 20:
            page_categories.append("vuota")
            debug_rows.append(f"p{idx}=vuota")
            continue

        page_blob = normalize_text_for_matching(page_text)
        cat, scores_tmp, meta_tmp = score_category(page_text, f"{filename}#page_{idx}")

        if idx == 1:
            first_page_category = cat

        debug_rows.append(
            f"p{idx}_raw_category={cat} | first_page_category={first_page_category} | "
            f"has_programma={'programma corso' in page_blob or 'programma del corso' in page_blob}"
        )

        # pagina programma allegata a un attestato
        if idx > 1 and first_page_category == "attestati":
            if (
                "programma corso" in page_blob
                or "programma del corso" in page_blob
                or "programma corso di aggiornamento" in page_blob
            ):
                debug_rows.append(f"p{idx}=attestati_programma_corso")
                page_categories.append("attestati")
                continue

        page_categories.append(cat)
        debug_rows.append(f"p{idx}={cat}")

    meaningful = [c for c in page_categories if c not in {"vuota", "altri_da_verificare"}]
    distinct = sorted(list(set(meaningful)))

    counts: Dict[str, int] = {}
    for c in meaningful:
        counts[c] = counts.get(c, 0) + 1

    dominant = None
    if counts:
        dominant = max(counts, key=counts.get)

    result["page_categories"] = page_categories
    result["dominant_category"] = dominant
    result["distinct_categories"] = distinct
    result["debug"] = debug_rows

    if len(distinct) >= 2:
        ordered_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_count = ordered_counts[0][1]
        second_count = ordered_counts[1][1] if len(ordered_counts) > 1 else 0

        if second_count >= 2 or top_count == second_count:
            result["is_mixed"] = True

    return result

# =========================================================
# ANALISI DOCUMENTO
# =========================================================

def analyze_document(filename: str, content: bytes, content_type: str) -> Dict[str, Any]:
    extraction = extract_text_from_file(filename, content, content_type)
    text = extraction["text"]

    category, scores, category_meta = score_category(text, filename)
    mixed_info = detect_mixed_pdf_categories(filename, content, content_type)

    result = {
        "filename": filename,
        "content_type": content_type,
        "size_bytes": len(content),
        "testo_estratto": text[:3000],
        "categoria": category,
        "categoria_label": CATEGORY_LABELS.get(category, "Da verificare"),
        "cartella": FOLDERS.get(category, "altri_da_verificare"),
        "nome": "",
        "cognome": "",
        "corso": "",
        "famiglia_corso": "",
        "tipo_percorso": "",
        "modulo_formazione_lavoratori": "",
        "data_conclusione": "",
        "data_scadenza": "",
        "prossimo_aggiornamento": "",
        "scadenza_label": "",
        "confidenza": "bassa",
        "needs_review": False,
        "review_reasons": [],
        "score_details": scores,
        "score_meta": category_meta,
        "suggested_filename": safe_filename(filename),
        "extraction_method": extraction["extraction_method"],
        "ocr_used": extraction["ocr_used"],
        "ocr_pages": extraction["ocr_pages"],
        "ocr_soft_limit": extraction["ocr_soft_limit"],
        "ocr_alert": extraction["ocr_alert"],
        "extraction_error": extraction.get("extraction_error", ""),
        "page_categories": mixed_info.get("page_categories", []),
        "mixed_document": mixed_info.get("is_mixed", False),
        "parser_debug": [],
    }

    if extraction["extraction_method"] == "extraction_failed":
        result["categoria"] = "altri_da_verificare"
        result["categoria_label"] = "Da verificare"
        result["cartella"] = FOLDERS["altri_da_verificare"]
        result["needs_review"] = True
        result["review_reasons"] = ["estrazione_testo_fallita"]
        result["confidenza"] = "bassa"
        result["parser_debug"] = ["estrazione testo fallita"]
        return result

    if mixed_info.get("is_mixed"):
        result["categoria"] = "altri_da_verificare"
        result["categoria_label"] = "Da verificare"
        result["cartella"] = FOLDERS["altri_da_verificare"]
        result["needs_review"] = True
        result["confidenza"] = "bassa"
        if "documento_composto_multi_categoria" not in result["review_reasons"]:
            result["review_reasons"].append("documento_composto_multi_categoria")
        result["parser_debug"] = mixed_info.get("debug", [])
        return result

    if category == "attestati":
        parsed = parse_attestato(text, filename)
        result.update(parsed)
    elif category == "nomine":
        parsed = parse_nomina(text, filename)
        result.update(parsed)
    elif category == "verbali_dpi":
        parsed = parse_verbale_dpi(text, filename)
        result.update(parsed)
    elif category == "visite_mediche":
        parsed = parse_visita_medica(text, filename)
        result.update(parsed)
    elif category == "documenti_aziendali":
        parsed = parse_documento_generico(text, filename, "Documento Aziendale")
        result.update(parsed)
    else:
        parsed = parse_documento_generico(text, filename, "Da verificare")
        result.update(parsed)

    result["cartella"] = FOLDERS.get(category, "altri_da_verificare")

    if category_meta.get("delta", 99) <= 1:
        result["needs_review"] = True
        if "classificazione_ambigua" not in result["review_reasons"]:
            result["review_reasons"].append("classificazione_ambigua")

    return result


# =========================================================
# OVERRIDE CATEGORIA DA FRONTEND
# =========================================================

def normalize_category_override(value: str) -> Optional[str]:
    if not value:
        return None
    key = normalize_text_for_matching(value)
    return CATEGORY_LABEL_TO_KEY.get(key)


def apply_category_override(item: Dict[str, Any], forced_category: Optional[str]) -> Dict[str, Any]:
    if not forced_category:
        return item

    filename = item.get("filename", "")
    ext = os.path.splitext(filename)[1] or ".pdf"

    item["categoria"] = forced_category
    item["categoria_label"] = CATEGORY_LABELS.get(forced_category, "Da verificare")
    item["cartella"] = FOLDERS.get(forced_category, "altri_da_verificare")

    if forced_category == "documenti_aziendali":
        item["categoria"] = "documenti_aziendali"
        item["categoria_label"] = "Documenti Aziendali"
        item["corso"] = ""
        item["famiglia_corso"] = ""
        item["tipo_percorso"] = ""
        item["modulo_formazione_lavoratori"] = ""
        item["data_scadenza"] = ""
        item["prossimo_aggiornamento"] = ""
        item["scadenza_label"] = ""
        item["suggested_filename"] = safe_filename(os.path.splitext(filename)[0]) + ext

    elif forced_category == "altri_da_verificare":
        item["categoria"] = "altri_da_verificare"
        item["categoria_label"] = "Da verificare"
        item["needs_review"] = True
        if "override_manuale_frontend" not in item["review_reasons"]:
            item["review_reasons"].append("override_manuale_frontend")
        item["suggested_filename"] = safe_filename(os.path.splitext(filename)[0]) + ext

    return item


# =========================================================
# ZIP + REPORT
# =========================================================

def build_report_attestati(items: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("REPORT ATTESTATI")
    lines.append("=" * 180)
    lines.append("")

    header = " | ".join([
        "FILE",
        "COGNOME",
        "NOME",
        "CORSO",
        "TIPO_PERCORSO",
        "MODULO_FL",
        "DATA_CONCLUSIONE",
        "LABEL_SCADENZA",
        "VALORE_SCADENZA",
        "CONFIDENZA",
        "NEEDS_REVIEW",
        "REVIEW_REASONS",
        "EXTRACTION_METHOD",
    ])
    lines.append(header)
    lines.append("-" * 180)

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
                item.get("modulo_formazione_lavoratori", ""),
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


def build_zip(files_data: List[Tuple[UploadFile, bytes]], analyzed: List[Dict[str, Any]]) -> io.BytesIO:
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


# =========================================================
# HELPERS ENDPOINT
# =========================================================

async def analyze_upload(file: UploadFile) -> Dict[str, Any]:
    content = await file.read()
    return analyze_document(file.filename, content, file.content_type or "")


def parse_overrides_json(overrides_json: Optional[str]) -> Dict[str, str]:
    if not overrides_json:
        return {}

    try:
        raw = json.loads(overrides_json)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    normalized = {}
    for filename, category_value in raw.items():
        if not isinstance(filename, str):
            continue
        forced = normalize_category_override(str(category_value))
        if forced:
            normalized[filename] = forced

    return normalized


# =========================================================
# ENDPOINTS
# =========================================================

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {
        "status": "ok",
        "message": "Docu OCR Engine online",
        "version": "6.1.0"
    }


@app.get("/health")
def health():
    return JSONResponse({"status": "healthy"})


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return """
    <html>
      <head>
        <title>Upload Documenti</title>
      </head>
      <body style="font-family:Arial;padding:40px;">
        <h2>Analizza documento singolo</h2>
        <form action="/analyze" enctype="multipart/form-data" method="post">
          <input name="file" type="file" />
          <button type="submit">Analizza</button>
        </form>

        <hr>

        <h2>Analizza più file (JSON)</h2>
        <form action="/analyze-batch" enctype="multipart/form-data" method="post">
          <input name="files" type="file" multiple />
          <button type="submit">Analizza batch</button>
        </form>

        <hr>

        <h2>Scarica ZIP (richiede conferma esplicita)</h2>
        <form action="/organize-zip" enctype="multipart/form-data" method="post">
          <input name="files" type="file" multiple />
          <input type="hidden" name="confirm_download" value="true" />
          <button type="submit">Scarica ZIP</button>
        </form>
      </body>
    </html>
    """


@app.post("/analyze")
async def analyze(file: Annotated[UploadFile, File(...)]):
    try:
        item = await analyze_upload(file)
        return {"results": [item]}
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "results": [{
                    "filename": file.filename if file else "",
                    "content_type": file.content_type if file else "",
                    "size_bytes": 0,
                    "testo_estratto": "",
                    "categoria": "altri_da_verificare",
                    "categoria_label": "Da verificare",
                    "cartella": "altri_da_verificare",
                    "nome": "",
                    "cognome": "",
                    "corso": "",
                    "famiglia_corso": "",
                    "tipo_percorso": "",
                    "modulo_formazione_lavoratori": "",
                    "data_conclusione": "",
                    "data_scadenza": "",
                    "prossimo_aggiornamento": "",
                    "scadenza_label": "",
                    "confidenza": "bassa",
                    "needs_review": True,
                    "review_reasons": ["errore_interno_backend"],
                    "score_details": {},
                    "score_meta": {},
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


@app.post("/analyze-batch")
async def analyze_batch(files: Annotated[List[UploadFile], File(...)]):
    if not files:
        raise HTTPException(status_code=400, detail="Nessun file caricato")

    results = []
    for file in files:
        try:
            item = await analyze_upload(file)
            results.append(item)
        except Exception as e:
            results.append({
                "filename": file.filename if file else "",
                "content_type": file.content_type if file else "",
                "size_bytes": 0,
                "testo_estratto": "",
                "categoria": "altri_da_verificare",
                "categoria_label": "Da verificare",
                "cartella": "altri_da_verificare",
                "nome": "",
                "cognome": "",
                "corso": "",
                "famiglia_corso": "",
                "tipo_percorso": "",
                "modulo_formazione_lavoratori": "",
                "data_conclusione": "",
                "data_scadenza": "",
                "prossimo_aggiornamento": "",
                "scadenza_label": "",
                "confidenza": "bassa",
                "needs_review": True,
                "review_reasons": ["errore_interno_backend"],
                "score_details": {},
                "score_meta": {},
                "suggested_filename": safe_filename(file.filename) if file else "",
                "extraction_method": "fatal_error",
                "ocr_used": False,
                "ocr_pages": 0,
                "ocr_soft_limit": OCR_SOFT_LIMIT,
                "ocr_alert": False,
                "extraction_error": str(e),
                "parser_debug": ["eccezione gestita in endpoint /analyze-batch"],
            })

    return {
        "results": results,
        "count": len(results),
    }


@app.post("/organize-zip")
async def organize_zip(
    files: Annotated[List[UploadFile], File(...)],
    confirm_download: Annotated[Optional[str], Form()] = None,
    overrides_json: Annotated[Optional[str], Form()] = None,
):
    if str(confirm_download).strip().lower() != "true":
        raise HTTPException(
            status_code=400,
            detail="confirm_download=true richiesto per generare lo ZIP"
        )

    if not files:
        raise HTTPException(status_code=400, detail="Nessun file caricato")

    overrides_map = parse_overrides_json(overrides_json)
 
    files_data: List[Tuple[UploadFile, bytes]] = []
    analyzed: List[Dict[str, Any]] = []

    for file in files:
        content = await file.read()
        item = analyze_document(file.filename, content, file.content_type or "")

        forced_category = overrides_map.get(file.filename)
        item = apply_category_override(item, forced_category)

        files_data.append((file, content))
        analyzed.append(item)

    zip_buffer = build_zip(files_data, analyzed)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="archivio_documenti.zip"'},
    )
