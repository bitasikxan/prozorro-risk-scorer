"""
api.py
======
FastAPI-сервіс для скорингу ризиків тендерів Prozorro.

Endpoints:
    GET  /                       — статичний index.html
    GET  /health                 — статус сервісу
    GET  /score/{tender_id}      — скоринг тендера за OCID або UUID

Запуск:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Залежності:
    pip install fastapi uvicorn httpx
"""

import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from feature_extractor import FeatureExtractor
from feature_phrases    import explain_factors
from model_registry     import ModelRegistry


# ══════════════════════════════════════════════════════════════
# НАЛАШТУВАННЯ
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("api")

# Prozorro сайтовий API — приймає OCID напряму, без UUID-резолву
PROZORRO_API     = "https://prozorro.gov.ua/api/tenders/{tender_id}/details"
PROZORRO_TIMEOUT = 15.0

ARTIFACTS_ROOT = "artifacts"
STATS_PATH     = "stats.pkl"
INDEX_HTML     = "index.html"


# ══════════════════════════════════════════════════════════════
# ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ОДИН РАЗ ПРИ СТАРТІ
# ══════════════════════════════════════════════════════════════

log.info("Завантаження екстрактора та реєстру моделей …")
EXTRACTOR = FeatureExtractor(stats_path=STATS_PATH)
REGISTRY  = ModelRegistry(artifacts_root=ARTIFACTS_ROOT)
log.info("Готово до прийому запитів")


# ══════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Prozorro Risk Scorer",
    description="ML-сервіс прогнозування корупційних ризиків у тендерах",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# КЛІЄНТ PROZORRO
# ══════════════════════════════════════════════════════════════

async def fetch_tender(tender_id: str) -> dict:
    """
    Тягне тендер з prozorro.gov.ua/api/tenders/{id}/details
    Підтримує:
        - OCID:  UA-2026-05-21-013532-a
        - UUID:  90470ef8845f44a49eaad88c51c9868d
        - URL:   https://prozorro.gov.ua/tender/UA-...
    """
    tender_id = tender_id.strip()

    # Витягти OCID/UUID з URL якщо передали посилання
    if "prozorro.gov.ua/tender/" in tender_id:
        tender_id = tender_id.split("/tender/")[-1].strip("/")

    url = PROZORRO_API.format(tender_id=tender_id)
    log.info(f"Запит: {url}")

    async with httpx.AsyncClient(timeout=PROZORRO_TIMEOUT) as client:
        r = await client.get(url)
        if r.status_code == 404:
            raise HTTPException(404, f"Тендер {tender_id!r} не знайдено")
        r.raise_for_status()
        return r.json()


# ══════════════════════════════════════════════════════════════
# НОРМАЛІЗАЦІЯ JSON → очікуваний формат екстрактора
# ══════════════════════════════════════════════════════════════

def normalize_tender(raw: dict) -> dict:
    """
    prozorro.gov.ua/details відрізняється від офіційного API.
    Приводимо до формату якого очікує FeatureExtractor.
    """
    # procurementMethod — відсутній, є procurementMethodType
    pmt = (raw.get("procurementMethodType") or "").lower()
    if "negotiation" in pmt or "defense" in pmt or "limited" in pmt:
        raw.setdefault("procurementMethod", "limited")
    elif "selective" in pmt:
        raw.setdefault("procurementMethod", "selective")
    else:
        raw.setdefault("procurementMethod", "open")

    # CPV з generalClassifier якщо items порожній
    if not raw.get("items"):
        gc = raw.get("generalClassifier") or {}
        cpv_id = gc.get("id") or gc.get("description") or ""
        # "45420000-7" → items з classification
        if cpv_id:
            raw["items"] = [{"classification": {"id": cpv_id}}]

    # buyers → procuringEntity (якщо buyers порожній)
    if not raw.get("buyers"):
        pe = raw.get("procuringEntity") or {}
        if pe:
            raw["buyers"] = [pe]

    # documents: publicDocuments — словник {id: [doc, ...]}
    if not raw.get("documents") and raw.get("publicDocuments"):
        docs = []
        for doc_list in raw["publicDocuments"].values():
            docs.extend(doc_list if isinstance(doc_list, list) else [doc_list])
        raw["documents"] = docs

    # enquiries: у цьому API поле відсутнє, використовуємо config
    if "enquiries" not in raw:
        cfg = raw.get("config") or {}
        # Якщо hasEnquiries=true і є enquiryPeriod — вважаємо що запитання можливі
        raw["enquiries"] = []  # реальних даних нема, завжди порожньо

    # Дата публікації
    if not raw.get("datePublished"):
        tp = raw.get("tenderPeriod") or {}
        raw["datePublished"] = tp.get("startDate") or raw.get("noticePublicationDate")

    return raw


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "stages":   list(REGISTRY.bundles.keys()),
        "n_buyers": len(EXTRACTOR.stats.get("buyers", {})),
    }


@app.get("/")
async def root():
    p = Path(INDEX_HTML)
    if p.exists():
        return FileResponse(p)
    return JSONResponse({"detail": "index.html ще не створено"}, status_code=404)


@app.get("/score/{tender_id:path}")
async def score(tender_id: str, top_k: int = 3):
    """
    Скорить тендер. tender_id може бути OCID, UUID або повним URL prozorro.gov.ua.
    """
    # 1) тягнемо JSON
    try:
        raw = await fetch_tender(tender_id)
    except httpx.HTTPError as e:
        log.exception("Prozorro API error")
        raise HTTPException(502, f"Помилка зв'язку з Prozorro: {e}")

    if not raw:
        raise HTTPException(404, "Порожня відповідь від Prozorro")

    # 2) нормалізуємо під формат екстрактора
    tender = normalize_tender(raw)

    # це - ранній вихід, якщо тип тендеру - reporting
    # (тендер було проведено десь-якось, на сайті лише звіт про цей тендер)
    pmt = tender.get("procurementMethodType", "").lower()
    status = tender.get("status") or "active.enquiries"
    NON_COMPETITIVE = ("reporting", "negotiation", "limited", "defense", "simple.defense")
    if any(x in pmt for x in NON_COMPETITIVE):
        if "reporting" in pmt:
            warning_text = ("Звітна закупівля (reporting) — скоринг не застосовується. "
                            "Процедура не має конкурсного відбору.")
        else:
            warning_text = ("Закупівля без конкурсу (переговорна/обмежена/оборонна) — "
                            "скоринг конкурентного ризику не застосовується.")
        pe = tender.get("procuringEntity") or {}
        ident = (pe.get("identifier") or {})
        value = tender.get("value") or {}
        gc = tender.get("generalClassifier") or {}
        return {
            "tender": {
                "id": tender.get("tenderID") or tender.get("id"),
                "uuid": tender.get("id"),
                "title": gc.get("description") or tender.get("description") or "—",
                "status": status,
                "procurement": tender.get("procurementMethodType"),
                "category": tender.get("mainProcurementCategory"),
                "value": value.get("amount"),
                "currency": value.get("currency"),
                "buyer_name": pe.get("name"),
                "buyer_id": ident.get("id"),
                "date_published": tender.get("datePublished")
                                  or (tender.get("tenderPeriod") or {}).get("startDate"),
            },
            "stage": None,
            "risk_prob": None,
            "risk_score": None,
            "risk_level": None,
            "is_flagged": False,
            "threshold": None,
            "shap_factors": [],
            "warning": warning_text,
        }

    # 3) витягуємо ознаки + скоримо
    try:
        features = EXTRACTOR.extract(tender)
        result = REGISTRY.score_all(features, status=status, top_k=top_k)
    except Exception as e:
        log.exception("Scoring error")
        raise HTTPException(500, f"Помилка скорингу: {e}")

    # збагачуємо SHAP-фактори фразами для кожної стадії
    for stage_result in result["timeline"]:
        stage_result["shap_factors"] = explain_factors(stage_result["shap_factors"])

    # для зворотньої сумісності з фронтом — поля поточної (останньої) стадії в корінь
    current = result["timeline"][-1] if result["timeline"] else {}

    # 5) метадані тендера для фронту
    pe    = tender.get("procuringEntity") or {}
    ident = (pe.get("identifier") or {})
    value = tender.get("value") or {}
    gc    = tender.get("generalClassifier") or {}

    return {
        "tender": {
            "id":             tender.get("tenderID") or tender.get("id"),
            "uuid":           tender.get("id"),
            "title":          gc.get("description") or tender.get("description") or "—",
            "status":         status,
            "procurement":    tender.get("procurementMethodType"),
            "category":       tender.get("mainProcurementCategory"),
            "value":          value.get("amount"),
            "currency":       value.get("currency"),
            "buyer_name":     pe.get("name"),
            "buyer_id":       ident.get("id"),
            "date_published": tender.get("datePublished")
                              or (tender.get("tenderPeriod") or {}).get("startDate"),
        },
        **result,
        **current,   # stage, risk_prob, risk_score, risk_level, is_flagged, threshold, shap_factors
        "timeline":      result["timeline"],
        "current_stage": result["current_stage"],
    }