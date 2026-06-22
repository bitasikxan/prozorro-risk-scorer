"""
feature_extractor.py
====================
Парсер Prozorro JSON у вектор ознак для трьох моделей (M1/M2/M3).

Використання:
    from feature_extractor import FeatureExtractor

    fe = FeatureExtractor(stats_path="stats.pkl")

    # Автоматично за статусом тендера:
    features = fe.extract(tender_data)

    # Або явно:
    features = fe.extract_m2(tender_data)

Повертає dict {feature_name: value}, готовий до pd.DataFrame.

Залежність:
    stats.pkl — згладжені історичні агрегати по замовниках/CPV.
    Формат:
        {
          "buyers": {buyer_id: {"violation_rate": float,
                                "total_tenders": int,
                                "median_log_value": float}},
          "cpv":    {cpv_2digit: {"median_log_value": float,
                                  "median_sw_days":   float}},
          "global": {"median_log_value": float,
                     "median_sw_days":   float,
                     "violation_rate":   float}
        }
"""

from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════
# КОНСТАНТИ
# ══════════════════════════════════════════════════════════════

# Високоризикові CPV (синхронно з build_tender_features.py)
HIGH_RISK_CPV = {9, 45, 71, 72, 73, 79}

# Маппінг статус тендера → стадія моделі
STAGE_BY_STATUS = {
    "active.enquiries":                       "m1",
    "active.tendering":                       "m2",
    "active.pre-qualification":               "m2",
    "active.pre-qualification.stand-still":   "m2",
    "active.auction":                         "m2",
    "active.qualification":                   "m3",
    "active.awarded":                         "m3",
    "complete":                               "m3",
    # рідкісні / завершені без переможця
    "cancelled":                              "m3",
    "unsuccessful":                           "m3",
}


# ══════════════════════════════════════════════════════════════
# КЛАС
# ══════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Витягує ознаки з Prozorro JSON для трьох моделей через композицію."""

    # Згладжування Лапласа для невідомого замовника: ALPHA / BETA
    _SMOOTH_ALPHA = 1.0
    _SMOOTH_BETA  = 10.0

    def __init__(self, stats_path: str = "stats.pkl"):
        p = Path(stats_path)
        if p.exists():
            with open(p, "rb") as f:
                self.stats = pickle.load(f)
        else:
            print(f"[FeatureExtractor] УВАГА: {stats_path} не знайдено. "
                  "Використовую fallback — рейтинги замовників будуть неточні.")
            self.stats = {
                "buyers": {},
                "cpv":    {},
                "global": {
                    "median_log_value": 12.0,   # ≈ e^12 = 160K UAH
                    "median_sw_days":   7.0,
                    "violation_rate":   self._SMOOTH_ALPHA / self._SMOOTH_BETA,
                },
            }

    # ── визначення стадії ──────────────────────────────────────

    @staticmethod
    def detect_stage(tender: dict) -> str:
        """Повертає 'm1' | 'm2' | 'm3' за статусом тендера."""
        status = tender.get("status") or "active.enquiries"
        return STAGE_BY_STATUS.get(status, "m1")

    # ── публічний інтерфейс ────────────────────────────────────

    def extract(self, tender: dict, stage: Optional[str] = None) -> dict:
        """Автоматично обирає M1/M2/M3 за статусом."""
        stage = stage or self.detect_stage(tender)
        if stage == "m1":
            return self.extract_m1(tender)
        if stage == "m2":
            return self.extract_m2(tender)
        return self.extract_m3(tender)

    # ── M1: ознаки моменту публікації ──────────────────────────

    def extract_m1(self, tender: dict) -> dict:
        feats: dict = {}

        # Процедурні
        feats["proc_method_enc"]    = self._encode_method(tender.get("procurementMethod"))
        feats["non_price_criteria"] = int(
            (tender.get("awardCriteria") or "lowestCost") != "lowestCost"
        )

        category = tender.get("mainProcurementCategory") or "goods"
        feats["is_works"]    = int(category == "works")
        feats["is_services"] = int(category == "services")

        # CPV
        cpv_2 = self._extract_cpv_2digit(tender)
        feats["is_high_risk_cpv"] = int(cpv_2 in HIGH_RISK_CPV)

        # Фінансові
        amount = self._safe_amount(tender.get("value"))
        feats["log_tender_value"] = math.log1p(amount)

        # Часові вікна
        published = self._get_publication_date(tender)
        end_date  = self._parse_date(
            (tender.get("tenderPeriod") or {}).get("endDate")
        )
        if published and end_date:
            sw = (end_date - published).days
            feats["submission_window_days"] = float(sw)
            feats["has_submission_window"]  = 1
        else:
            feats["submission_window_days"] = -1.0
            feats["has_submission_window"]  = 0

        # Кількісні
        feats["number_of_items"]     = float(len(tender.get("items") or []))
        docs = tender.get("documents")
        if not docs:
            pdocs = tender.get("publicDocuments") or {}
            if isinstance(pdocs, dict):
                # рахуємо документи у всіх ключах
                n_docs = sum(len(v) if isinstance(v, list) else 1
                             for v in pdocs.values())
            else:
                n_docs = len(pdocs)
        else:
            n_docs = len(docs)
        feats["number_of_documents"] = float(n_docs)

        # Прапори
        feats["is_buyer_masked"] = self._is_buyer_masked(tender)

        if published:
            feats["is_weekend"]  = int(published.weekday() >= 5)
            feats["is_q4"]       = int(published.month >= 10)
            feats["is_december"] = int(published.month == 12)
        else:
            feats["is_weekend"] = feats["is_q4"] = feats["is_december"] = 0

        # ── історичні lookup-и ─────────────────────────
        buyer_id = self._extract_buyer_id(tender)
        bstats   = self.stats["buyers"].get(buyer_id)

        if bstats:
            feats["buyer_violation_rate"] = float(bstats["violation_rate"])
            feats["buyer_total_tenders"]  = float(bstats["total_tenders"])
        else:
            # Згладжування Лапласа для невідомого замовника
            feats["buyer_violation_rate"] = self._SMOOTH_ALPHA / self._SMOOTH_BETA
            feats["buyer_total_tenders"]  = 0.0

        # ── відносні ───────────────────────────────────
        cpv_stats     = self.stats["cpv"].get(cpv_2, {})
        cpv_med_value = cpv_stats.get(
            "median_log_value", self.stats["global"]["median_log_value"]
        )
        cpv_med_sw    = cpv_stats.get(
            "median_sw_days",   self.stats["global"]["median_sw_days"]
        )

        feats["value_vs_cpv_median"]  = feats["log_tender_value"] - cpv_med_value
        feats["window_vs_cpv_median"] = feats["submission_window_days"] - cpv_med_sw

        buyer_med_value = (bstats or {}).get("median_log_value", cpv_med_value)
        feats["value_vs_buyer_median"] = feats["log_tender_value"] - buyer_med_value

        return feats

    # ── M2: + інтерес учасників на етапі тендерування ─────────

    def extract_m2(self, tender: dict) -> dict:
        feats = self.extract_m1(tender)
        enquiries = tender.get("enquiries") or []
        feats["has_enquiries"] = int(len(enquiries) > 0)
        return feats

    # ── M3: + постаукційні ознаки ──────────────────────────────

    def extract_m3(self, tender: dict) -> dict:
        feats = self.extract_m2(tender)

        bids   = tender.get("bids")   or []
        awards = tender.get("awards") or []

        # Учасники
        active_bids = [b for b in bids if b.get("status") in (None, "active", "valid")]
        n_bidders   = len(active_bids) if active_bids else len(bids)
        feats["number_of_tenderers"] = float(n_bidders)
        feats["is_single_bidder"]    = int(n_bidders == 1)
        feats["is_competitive"]      = int(n_bidders >= 2)

        # Зниження ціни — формула як у build_tender_features_m3.py:
        #   price_change_pct = (avg_award - tender_value) / tender_value * 100
        # Тобто: від'ємне значення = ціна впала (нормально), 0 = підозріло.
        initial = self._safe_amount(tender.get("value"))
        active_awards = [a for a in awards if a.get("status") == "active"]

        if initial > 0 and active_awards:
            awarded = [self._safe_amount(a.get("value")) for a in active_awards]
            awarded = [x for x in awarded if x > 0]
            if awarded:
                avg_award = sum(awarded) / len(awarded)
                price_change_pct = (avg_award - initial) / initial * 100.0
            else:
                price_change_pct = 0.0
        else:
            price_change_pct = 0.0

        feats["price_change_pct"] = price_change_pct
        # near_zero_discount = підозріло мала знижка (0% до 1% зниження)
        feats["near_zero_discount"] = int(-1.0 <= price_change_pct <= 0.0)
        # price_increase = ціна зросла (аномалія: переможець дорожче за стартову)
        feats["price_increase"]     = int(price_change_pct > 0)
        # discount_pct_avg = (1 - mean(awarded)/initial) * 100, формула з build_m3
        # фактично -price_change_pct, але в навчальному датасеті це окрема CSV-колонка
        feats["discount_pct_avg"]   = -price_change_pct

        # Концентрація нагород — індекс Герфіндаля Σ(share²)
        # 1 учасник = 1.0 (моноконцентрація), рівні частки = 1/N
        if len(active_awards) >= 1:
            vals = [self._safe_amount(a.get("value")) for a in active_awards]
            total = sum(vals)
            if total > 0:
                shares = [v / total for v in vals]
                feats["award_concentration"] = sum(s * s for s in shares)
            else:
                feats["award_concentration"] = 1.0 if len(active_awards) == 1 else 0.0
        else:
            feats["award_concentration"] = 0.0

        # Прапори
        feats["has_cancelled_awards"]    = int(any(a.get("status") == "cancelled"
                                                  for a in awards))
        feats["has_unsuccessful_awards"] = int(any(a.get("status") == "unsuccessful"
                                                  for a in awards))
        feats["has_multiple_awards"]     = int(len(awards) > 1)

        return feats

    # ══════════════════════════════════════════════════════════
    # ДОПОМІЖНІ
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _encode_method(method: Optional[str]) -> int:
        """
        Підтримує як procurementMethod (open/limited/selective)
        так і procurementMethodType (aboveThreshold/negotiation/...).
        """
        m = (method or "").lower()
        if any(x in m for x in ("negotiation", "limited", "defense")):
            return 2
        if "selective" in m:
            return 1
        return 0  # open / abovethreshold / belowthreshold / reporting / ...

    @staticmethod
    def _safe_amount(value_obj) -> float:
        """Безпечно витягує amount з {"amount": ..., "currency": "UAH"}."""
        if not isinstance(value_obj, dict):
            return 0.0
        try:
            return float(value_obj.get("amount") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_cpv_2digit(tender: dict) -> int:
        """
        Бере CPV з кількох можливих місць:
            1. items[].classification.id           ('45420000-7')
            2. generalClassifier.id                ('45420000-7')
            3. generalClassifier.description       ('ДК 021:2015: 45420000-7 — ...')
        """
        import re

        # 1. items
        for item in (tender.get("items") or []):
            cls = (item.get("classification") or {}).get("id", "")
            code = str(cls).split("-")[0]
            if code and code[:2].isdigit():
                return int(code[:2])

        # 2-3. generalClassifier — пробуємо id, потім description
        gc = tender.get("generalClassifier") or {}
        for field in ("id", "description"):
            text = gc.get(field) or ""
            # шукаємо перший CPV-код формату 12345678-9
            m = re.search(r"(\d{8})-\d", text)
            if m:
                return int(m.group(1)[:2])
            # або просто перші 8 цифр (на випадок 'description' без дефіса)
            m = re.search(r"(\d{8})", text)
            if m:
                return int(m.group(1)[:2])

        return 0

    @staticmethod
    def _extract_buyer_id(tender: dict) -> str:
        """Спробує buyers[0], потім procuringEntity. Повертає 8-значний код."""
        for b in (tender.get("buyers") or []):
            ident = (b.get("identifier") or {}).get("id")
            if ident:
                return str(ident).strip().zfill(8)
        pe = tender.get("procuringEntity") or {}
        ident = (pe.get("identifier") or {}).get("id")
        if ident:
            return str(ident).strip().zfill(8)
        return ""

    @staticmethod
    def _is_buyer_masked(tender: dict) -> int:
        """Маскування замовника: спецсимволи в назві або defense-процедура."""
        pe   = tender.get("procuringEntity") or {}
        name = (pe.get("name") or "").lower()
        if "*" in name or "ххх" in name:
            return 1
        pmt = (tender.get("procurementMethodType") or "").lower()
        if pmt.startswith("simple.defense") or "defense" in pmt:
            return 1
        return 0

    @staticmethod
    def _parse_date(s: Optional[str]) -> Optional[datetime]:
        """ISO 8601 → naive datetime (UTC-naive, відкидаємо tz)."""
        if not s:
            return None
        try:
            return datetime.fromisoformat(
                s.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except (ValueError, AttributeError, TypeError):
            return None

    @classmethod
    def _get_publication_date(cls, tender: dict) -> Optional[datetime]:
        """Шукає дату публікації в кількох можливих полях."""
        candidates = [
            (tender.get("tenderPeriod") or {}).get("startDate"),
            tender.get("datePublished"),
            tender.get("dateCreated"),
            tender.get("dateModified"),
        ]
        for s in candidates:
            d = cls._parse_date(s)
            if d:
                return d
        return None


# ══════════════════════════════════════════════════════════════
# ШВИДКА ПЕРЕВІРКА
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Використання: python feature_extractor.py <tender.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    tender = payload.get("data", payload)
    fe = FeatureExtractor()
    stage = fe.detect_stage(tender)
    feats = fe.extract(tender)

    print(f"Стадія: {stage}")
    print(f"Ознак:  {len(feats)}")
    print()
    for k, v in feats.items():
        print(f"  {k:<32}  {v}")