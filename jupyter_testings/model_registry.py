"""
model_registry.py
=================
Реєстр трьох моделей (M1/M2/M3) з єдиним інтерфейсом скорингу.
"""

import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from feature_extractor import STAGE_BY_STATUS


# пороги ризику для ВСІХ трьох моделей.
RISK_THRESHOLDS = {
    "Низький":    (0.00, 0.05),
    "Помірний":   (0.05, 0.15),
    "Високий":    (0.15, 0.30),
    "Критичний":  (0.30, 1.01),
}

M1_FILES = {
    "model":      "tender_model_lgb.pkl",
    "calibrator": "tender_calibrator.pkl",
    "explainer":  "tender_shap_explainer.pkl",
    "medians":    "tender_feature_medians.pkl",
    "meta":       "tender_model_meta.json",
}
STD_FILES = {
    "model":      "model.pkl",
    "calibrator": "calibrator.pkl",
    "explainer":  "explainer.pkl",
    "medians":    "medians.pkl",
    "meta":       "meta.json",
}


def risk_level(prob: float, thresholds: dict = RISK_THRESHOLDS) -> str:
    """Назва рівня ризику для ймовірності."""
    for name, (lo, hi) in thresholds.items():
        if lo <= prob < hi:
            return name
    return "Критичний"   # на випадок prob == 1.0


def _load_bundle(directory: Path, files: dict) -> dict:
    """Зчитує model + calibrator + explainer + medians + meta з диска."""
    model      = joblib.load(directory / files["model"])
    calibrator = joblib.load(directory / files["calibrator"])
    medians    = joblib.load(directory / files["medians"])

    explainer_path = directory / files["explainer"]
    explainer = joblib.load(explainer_path) if explainer_path.exists() else None

    with open(directory / files["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "model":      model,
        "calibrator": calibrator,
        "explainer":  explainer,
        "medians":    medians if isinstance(medians, dict) else medians.to_dict(),
        "meta":       meta,
        "features":   meta["feature_cols"],
        "threshold":  float(meta.get("best_threshold", 0.15)),
    }


class ModelRegistry:
    """Тримає три моделі. Обирає за статусом тендера, скорить, повертає SHAP."""

    def __init__(self, artifacts_root: str = "artifacts"):
        root = Path(artifacts_root)
        print(f"[Registry] Завантаження моделей з {root}/")

        self.bundles = {
            "m1": _load_bundle(root,        M1_FILES),
            "m2": _load_bundle(root / "m2", STD_FILES),
            "m3": _load_bundle(root / "m3", STD_FILES),
        }

        for stage, b in self.bundles.items():
            shap_mark = "✓" if b["explainer"] is not None else "✗"
            print(f"  {stage.upper()}: {len(b['features'])} ознак | "
                  f"поріг {b['threshold']:.3f} | SHAP {shap_mark}")

    @staticmethod
    def stage_for_status(status: str) -> str:
        return STAGE_BY_STATUS.get(status, "m1")

    def score(self, features: dict, status: str, top_k: int = 3) -> dict:
        """
        Скорить тендер відповідною моделлю.

        Повертає:
            {
              "stage":         "m1" | "m2" | "m3",
              "risk_prob":     0..1,
              "risk_score":    0..100,
              "risk_level":    "Низький" | "Помірний" | "Високий" | "Критичний",
              "is_flagged":    bool   (prob >= threshold моделі),
              "threshold":     поріг моделі,
              "shap_factors":  [{feature, value, shap, direction}, ...]
            }
        """
        stage = self.stage_for_status(status)
        bundle = self.bundles[stage]

        X = self._prepare_input(features, bundle)

        raw  = bundle["model"].predict_proba(X)[:, 1]
        prob = float(bundle["calibrator"].predict(raw)[0])
        prob = max(0.0, min(1.0, prob))   # clip про всяк

        return {
            "stage":        stage,
            "risk_prob":    prob,
            "risk_score":   round(prob * 100, 1),
            "risk_level":   risk_level(prob),
            "is_flagged":   prob >= bundle["threshold"],
            "threshold":    bundle["threshold"],
            "shap_factors": self._top_shap_factors(X, bundle, top_k),
        }

    def score_all(self, features: dict, status: str, top_k: int = 3) -> dict:
        """
        Скорить тендер всіма моделями що доступні для його стадії.
        Тендер на M3 → повертає [M1, M2, M3].
        Тендер на M2 → повертає [M1, M2].
        Тендер на M1 → повертає [M1].
        """
        current_stage = self.stage_for_status(status)
        stage_order = ["m1", "m2", "m3"]
        current_idx = stage_order.index(current_stage)

        timeline = []
        for stage in stage_order[:current_idx + 1]:
            bundle = self.bundles.get(stage)
            if bundle is None:
                continue
            X = self._prepare_input(features, bundle)
            raw = bundle["model"].predict_proba(X)[:, 1]
            prob = float(bundle["calibrator"].predict(raw)[0])
            prob = max(0.0, min(1.0, prob))
            timeline.append({
                "stage": stage,
                "risk_prob": prob,
                "risk_score": round(prob * 100, 1),
                "risk_level": risk_level(prob),
                "is_flagged": prob >= bundle["threshold"],
                "threshold": bundle["threshold"],
                "shap_factors": self._top_shap_factors(X, bundle, top_k),
            })

        return {"current_stage": current_stage, "timeline": timeline}
    # ── приватне ────────────────────────────────────────────────

    @staticmethod
    def _prepare_input(features: dict, bundle: dict) -> pd.DataFrame:
        """Будує одно-рядковий DataFrame у точному порядку feature_cols.
        Відсутні ознаки заповнює медіаною з train."""
        cols    = bundle["features"]
        medians = bundle["medians"]
        row     = {c: features.get(c, medians.get(c, 0.0)) for c in cols}
        return pd.DataFrame([row], columns=cols)

    @staticmethod
    def _top_shap_factors(X: pd.DataFrame, bundle: dict, top_k: int) -> list:
        """Топ-K ознак з найбільшим |SHAP|, де значення відрізняється від медіани."""
        explainer = bundle["explainer"]
        if explainer is None:
            return []

        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv).flatten()

        medians = bundle["medians"]

        # Сортуємо всі ознаки за |SHAP| і беремо топ-K серед інформативних
        all_idx = np.argsort(np.abs(sv))[::-1]

        result = []
        for i in all_idx:
            feature = X.columns[i]
            value = float(X.iloc[0, i])
            median = medians.get(feature)
            # Пропускаємо якщо значення = медіані (ознака не відрізняє цей тендер)
            if median is not None and abs(value - median) < 0.01:
                continue
            result.append({
                "feature": feature,
                "value": value,
                "shap": float(sv[i]),
                "direction": "збільшує" if sv[i] > 0 else "зменшує",
            })
            if len(result) >= top_k:
                break

        # Fallback: якщо всі ознаки = медіані — повертаємо топ без фільтру
        if not result:
            for i in all_idx[:top_k]:
                result.append({
                    "feature": X.columns[i],
                    "value": float(X.iloc[0, i]),
                    "shap": float(sv[i]),
                    "direction": "збільшує" if sv[i] > 0 else "зменшує",
                })

        return result

if __name__ == "__main__":
    reg = ModelRegistry()

    print("\nМапа статус → стадія:")
    for s in ("active.enquiries", "active.tendering", "active.auction",
             "active.qualification", "active.awarded", "complete"):
        print(f"  {s:<24} → {reg.stage_for_status(s)}")