"""
build_stats.py
==============
Генерує stats.pkl — feature store з історичних агрегатів,
яких немає в JSON одного тендера.

Структура stats.pkl:
    {
      "buyers": {
        buyer_id: {
          "violation_rate":   float,   # згладжена частка порушень
          "total_tenders":    int,
          "median_log_value": float,
        },
        ...
      },
      "cpv": {
        cpv_2digit: {
          "median_log_value": float,
          "median_sw_days":   float,
        },
        ...
      },
      "global": {
        "median_log_value": float,
        "median_sw_days":   float,
        "violation_rate":   float,    # base rate як fallback
      },
    }

Згладжування Лапласа (синхронно з build_tender_features.py):
    violation_rate = (n_violations + 1) / (n_tenders + 10)

Запуск:
    python build_stats.py

Вхід:
    features_tenders_full.parquet   # M1-датасет (достатньо, бо потрібні
                                    # лише ознаки моменту публікації +
                                    # buyer_id + dasu_label)
Вихід:
    stats.pkl
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════
# НАЛАШТУВАННЯ
# ══════════════════════════════════════════════════════════════

FEATURES_PATH = "features_tenders_full.parquet"
OUTPUT_PATH   = "stats.pkl"

SMOOTH_ALPHA = 1
SMOOTH_BETA  = 10

# Мінімум тендерів щоб buyer потрапив у lookup
# (замовники з 1-2 тендерами → fallback на global)
MIN_BUYER_TENDERS = 1


# ══════════════════════════════════════════════════════════════
# ОСНОВНА ЛОГІКА
# ══════════════════════════════════════════════════════════════

def build_buyer_stats(df: pd.DataFrame) -> dict:
    """Агрегати по замовниках: violation_rate + median_log_value."""
    print("\n[1/3] Агрегати по замовниках ...")
    t0 = time.time()

    grp = df.groupby("buyer_id", observed=True).agg(
        n_tenders        = ("tender_id",        "count"),
        n_violations     = ("dasu_label",       "sum"),
        median_log_value = ("log_tender_value", "median"),
    ).reset_index()

    grp = grp[grp["n_tenders"] >= MIN_BUYER_TENDERS]

    grp["violation_rate"] = (
        (grp["n_violations"] + SMOOTH_ALPHA) /
        (grp["n_tenders"]    + SMOOTH_BETA)
    )

    buyers = {
        row.buyer_id: {
            "violation_rate":   float(row.violation_rate),
            "total_tenders":    int(row.n_tenders),
            "median_log_value": float(row.median_log_value),
        }
        for row in grp.itertuples(index=False)
    }

    elapsed = time.time() - t0
    print(f"      Замовників:                   {len(buyers):,}")
    print(f"      violation_rate min/med/max:   "
          f"{grp['violation_rate'].min():.4f} / "
          f"{grp['violation_rate'].median():.4f} / "
          f"{grp['violation_rate'].max():.4f}")
    print(f"      Час:                          {elapsed:.1f}с")
    return buyers


def build_cpv_stats(df: pd.DataFrame) -> dict:
    """Агрегати по CPV: median_log_value + median_sw_days."""
    print("\n[2/3] Агрегати по CPV ...")
    t0 = time.time()

    # submission_window: -1 означає "немає даних" → відфільтрувати
    df_sw = df[df["has_submission_window"] == 1]

    by_cpv_value = df.groupby("main_cpv_2_digit", observed=True)["log_tender_value"].median()
    by_cpv_sw    = df_sw.groupby("main_cpv_2_digit", observed=True)["submission_window_days"].median()

    cpv = {}
    for cpv_code in by_cpv_value.index:
        cpv[int(cpv_code)] = {
            "median_log_value": float(by_cpv_value.loc[cpv_code]),
            "median_sw_days":   float(by_cpv_sw.get(cpv_code, 7.0)),
        }

    elapsed = time.time() - t0
    print(f"      Унікальних CPV (2-знач.):     {len(cpv)}")
    print(f"      Час:                          {elapsed:.1f}с")
    return cpv


def build_global_stats(df: pd.DataFrame) -> dict:
    """Глобальні fallback-значення."""
    print("\n[3/3] Глобальні fallback ...")

    df_sw = df[df["has_submission_window"] == 1]

    global_stats = {
        "median_log_value": float(df["log_tender_value"].median()),
        "median_sw_days":   float(df_sw["submission_window_days"].median())
                            if len(df_sw) else 7.0,
        "violation_rate":   float(
            (df["dasu_label"].sum() + SMOOTH_ALPHA) /
            (len(df)                + SMOOTH_BETA)
        ),
    }
    for k, v in global_stats.items():
        print(f"      {k:<22} {v:.4f}")
    return global_stats


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  Генерація stats.pkl (feature store)")
    print("═" * 60)

    if not Path(FEATURES_PATH).exists():
        raise FileNotFoundError(
            f"Не знайдено {FEATURES_PATH}. "
            "Спочатку запусти build_tender_features.py"
        )

    print(f"\nЗавантаження {FEATURES_PATH} ...")
    t0 = time.time()
    df = pd.read_parquet(
        FEATURES_PATH,
        columns=[
            "tender_id", "buyer_id", "dasu_label",
            "main_cpv_2_digit", "log_tender_value",
            "submission_window_days", "has_submission_window",
        ],
    )
    print(f"      Рядків: {len(df):,}  |  {time.time() - t0:.1f}с")

    # Очистка: викидаємо рядки без buyer_id
    df = df[df["buyer_id"].notna() & df["buyer_id"].ne("nan")]

    stats = {
        "buyers": build_buyer_stats(df),
        "cpv":    build_cpv_stats(df),
        "global": build_global_stats(df),
    }

    print(f"\nЗбереження → {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = Path(OUTPUT_PATH).stat().st_size / 1024**2
    print(f"      Розмір файлу: {size_mb:.1f} МБ")
    print(f"\n✅ Готово")


if __name__ == "__main__":
    main()