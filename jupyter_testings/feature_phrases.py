"""
feature_phrases.py
==================
Перетворює SHAP-фактори у людиночитаємі українські фрази.

Вхід — фактор з model_registry._top_shap_factors():
    {"feature": "buyer_violation_rate", "value": 0.34,
     "shap": 0.21, "direction": "збільшує"}

Вихід — той самий dict + поле "phrase":
    {..., "phrase": "У замовника висока частка попередніх порушень (34.0%)"}

Поле "direction" зберігається — фронтенд сам вирішить колір/іконку.
"""

import math


# ══════════════════════════════════════════════════════════════
# ФОРМАТУВАННЯ
# ══════════════════════════════════════════════════════════════

def _fmt_money(log_amount: float) -> str:
    """log1p(UAH) → людський рядок."""
    amount = math.expm1(log_amount)
    if amount >= 1e9:
        return f"{amount/1e9:.1f} млрд грн"
    if amount >= 1e6:
        return f"{amount/1e6:.1f} млн грн"
    if amount >= 1e3:
        return f"{amount/1e3:.0f} тис. грн"
    return f"{amount:.0f} грн"


def _fmt_days(days: float) -> str:
    d = int(round(days))
    if d == 1:
        return "1 день"
    if 2 <= d <= 4:
        return f"{d} дні"
    return f"{d} днів"


PROC_METHOD = {
    0: "Відкрита процедура",
    1: "Селективна процедура",
    2: "Обмежена/переговорна процедура",
}


# ══════════════════════════════════════════════════════════════
# СЛОВНИК → ФРАЗА
# ══════════════════════════════════════════════════════════════

def phrase_for(feature: str, value: float) -> str:
    """Українська фраза що описує ознаку та її значення."""

    # ── процедурні ───────────────────────────────────────────
    if feature == "proc_method_enc":
        return PROC_METHOD.get(int(value), "Невідома процедура")

    if feature == "non_price_criteria":
        return ("Критерій відбору не за найнижчою ціною" if value > 0.5
                else "Стандартний критерій (найнижча ціна)")

    if feature == "is_works":
        return "Категорія: будівельні роботи" if value > 0.5 else "Не будівельні роботи"

    if feature == "is_services":
        return "Категорія: послуги" if value > 0.5 else "Не послуги"

    if feature == "is_high_risk_cpv":
        return ("Високоризикова галузь CPV" if value > 0.5
                else "Галузь CPV із низьким ризиком")

    # ── фінансові ────────────────────────────────────────────
    if feature == "log_tender_value":
        return f"Сума тендера: {_fmt_money(value)}"

    if feature == "value_vs_cpv_median":
        if value > 0.5:  return "Сума суттєво вища за медіану CPV"
        if value < -0.5: return "Сума суттєво нижча за медіану CPV"
        return "Сума близька до медіани CPV"

    if feature == "value_vs_buyer_median":
        if value > 0.5:  return "Сума суттєво вища за типову для замовника"
        if value < -0.5: return "Сума суттєво нижча за типову для замовника"
        return "Сума типова для замовника"

    # ── часові ───────────────────────────────────────────────
    if feature == "submission_window_days":
        if value < 0:    return "Період подання не визначено"
        if value < 7:    return f"Замало часу на підготовку пропозицій ({_fmt_days(value)})"
        if value > 21:   return f"Тривалий період подання ({_fmt_days(value)})"
        return f"Період подання: {_fmt_days(value)}"

    if feature == "has_submission_window":
        return "Період подання визначено" if value > 0.5 else "Період подання НЕ визначено"

    if feature == "window_vs_cpv_median":
        if value < -3:   return f"Період подання коротший за медіану CPV на {_fmt_days(abs(value))}"
        if value > 3:    return f"Період подання довший за медіану CPV на {_fmt_days(value)}"
        return "Період подання близький до медіани CPV"

    # ── календарні ───────────────────────────────────────────
    if feature == "is_weekend":
        return "Опубліковано у вихідний день" if value > 0.5 else "Опубліковано у будній день"

    if feature == "is_q4":
        return ("Опубліковано у IV кварталі (освоєння бюджету)" if value > 0.5
                else "Опубліковано не у IV кварталі")

    if feature == "is_december":
        return ("Опубліковано у грудні (кінець року)" if value > 0.5
                else "Опубліковано не у грудні")

    # ── структурні ───────────────────────────────────────────
    if feature == "number_of_items":
        return f"Лотів у тендері: {int(round(value))}"

    if feature == "number_of_documents":
        return f"Завантажено документів: {int(round(value))}"

    if feature == "is_buyer_masked":
        return ("Замовника замасковано (defense або *)" if value > 0.5
                else "Замовник відкритий")

    # ── історія замовника ────────────────────────────────────
    if feature == "buyer_violation_rate":
        pct = value * 100
        if pct >= 20:
            return f"У замовника висока частка попередніх порушень ({pct:.1f}%)"
        if pct >= 10:
            return f"У замовника помірна частка попередніх порушень ({pct:.1f}%)"
        return f"У замовника низька частка попередніх порушень ({pct:.1f}%)"

    if feature == "buyer_total_tenders":
        return f"Усього тендерів замовника в історії: {int(round(value))}"

    # ── M2: інтерес учасників ────────────────────────────────
    if feature == "has_enquiries":
        return ("Учасники задавали запитання" if value > 0.5
                else "Запитань від учасників не було")

    # ── M3: учасники ─────────────────────────────────────────
    if feature == "is_single_bidder":
        return "Лише один учасник" if value > 0.5 else "Учасників більше одного"

    if feature == "is_competitive":
        return ("Конкурентний тендер (≥2 учасники)" if value > 0.5
                else "Неконкурентний тендер")

    if feature == "number_of_tenderers":
        return f"Учасників: {int(round(value))}"

    # ── M3: ціни ─────────────────────────────────────────────
    if feature == "price_change_pct":
        if abs(value) < 0.1:
            return "Ціна не змінилась"
        if value > 0:
            return f"Ціна нагороди вища за стартову на {value:.1f}%"
        return f"Ціна знизилась на {abs(value):.1f}%"

    if feature == "near_zero_discount":
        return ("Підозріло мала знижка (близько 0%)" if value > 0.5
                else "Знижка більш ніж на 1%")

    if feature == "price_increase":
        return ("Ціна нагороди ВИЩА за стартову (аномалія)" if value > 0.5
                else "Ціна нагороди не вища за стартову")

    if feature == "discount_pct_avg":
        return f"Середня знижка: {value:.1f}%"

    if feature == "award_concentration":
        if value >= 0.9:
            return f"Висока концентрація нагород ({value:.2f})"
        if value >= 0.5:
            return f"Помірна концентрація нагород ({value:.2f})"
        return f"Низька концентрація нагород ({value:.2f})"

    # ── M3: статуси нагород ──────────────────────────────────
    if feature == "has_cancelled_awards":
        return ("Були скасовані рішення про нагородження" if value > 0.5
                else "Скасованих нагороджень не було")

    if feature == "has_unsuccessful_awards":
        return ("Були невдалі нагородження" if value > 0.5
                else "Невдалих нагороджень не було")

    if feature == "has_multiple_awards":
        return ("Кілька переможців (багатолотовий)" if value > 0.5
                else "Один переможець")

    # ── fallback ─────────────────────────────────────────────
    return f"{feature} = {value:.3g}"


# ══════════════════════════════════════════════════════════════
# ПУБЛІЧНІ ФУНКЦІЇ
# ══════════════════════════════════════════════════════════════

def explain_factor(factor: dict) -> dict:
    """Збагачує SHAP-фактор полем 'phrase'. Інші поля зберігаються."""
    return {**factor, "phrase": phrase_for(factor["feature"], factor["value"])}


def explain_factors(factors: list) -> list:
    """Список SHAP-факторів → список збагачених словників."""
    return [explain_factor(f) for f in factors]


# ══════════════════════════════════════════════════════════════
# Швидкий тест
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        {"feature": "buyer_violation_rate",  "value": 0.34, "shap": 0.21, "direction": "збільшує"},
        {"feature": "proc_method_enc",       "value": 2.0,  "shap": 0.15, "direction": "збільшує"},
        {"feature": "submission_window_days","value": 3.0,  "shap": 0.10, "direction": "збільшує"},
        {"feature": "is_single_bidder",      "value": 1.0,  "shap": 0.30, "direction": "збільшує"},
        {"feature": "near_zero_discount",    "value": 1.0,  "shap": 0.18, "direction": "збільшує"},
        {"feature": "price_change_pct",      "value": -3.4, "shap":-0.05, "direction": "зменшує"},
        {"feature": "log_tender_value",      "value": 14.5, "shap": 0.04, "direction": "збільшує"},
        {"feature": "buyer_total_tenders",   "value": 142,  "shap":-0.02, "direction": "зменшує"},
        {"feature": "award_concentration",   "value": 1.0,  "shap": 0.07, "direction": "збільшує"},
    ]
    for f in tests:
        e = explain_factor(f)
        arrow = "↑" if e["direction"] == "збільшує" else "↓"
        print(f"  {arrow} {e['phrase']}")