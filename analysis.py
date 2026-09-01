"""AirPassengers 시계열 분석 및 시각화 생성 스크립트.

실행하면 원본 CSV를 내려받고, 정제 데이터·그래프·요약 통계를 생성한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"
RAW_PATH = DATA_DIR / "air_passengers_raw.csv"
CLEAN_PATH = DATA_DIR / "air_passengers_clean.csv"
SUMMARY_PATH = DATA_DIR / "analysis_summary.json"
DATA_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/"
    "Rdatasets/master/csv/datasets/AirPassengers.csv"
)


def setup_style() -> None:
    """한글 폰트가 있으면 사용하고, 없으면 영문 레이블로 안전하게 출력한다."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        urlretrieve(DATA_URL, RAW_PATH)

    raw = pd.read_csv(RAW_PATH)
    required = {"time", "value"}
    if not required.issubset(raw.columns):
        raise ValueError(f"필수 컬럼이 없습니다: {required - set(raw.columns)}")

    year = np.floor(raw["time"]).astype(int)
    month = np.rint((raw["time"] - year) * 12).astype(int) + 1
    month = np.clip(month, 1, 12)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                {"year": year, "month": month, "day": np.ones(len(raw), dtype=int)}
            ),
            "passengers_thousands": pd.to_numeric(raw["value"], errors="coerce"),
        }
    ).sort_values("date")

    if df["date"].duplicated().any():
        raise ValueError("중복 날짜가 발견되었습니다.")

    # 월 단위 전체 인덱스로 재색인하여 누락된 월 자체도 결측치로 확인한다.
    full_months = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    df = df.set_index("date").reindex(full_months).rename_axis("date").reset_index()
    missing_before = int(df["passengers_thousands"].isna().sum())
    if missing_before:
        df["passengers_thousands"] = df["passengers_thousands"].interpolate(
            method="linear", limit_direction="both"
        )

    df["ma_12"] = df["passengers_thousands"].rolling(12).mean()
    df["mom_pct"] = df["passengers_thousands"].pct_change() * 100
    df["yoy_pct"] = df["passengers_thousands"].pct_change(12) * 100
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # 이상치는 원시 수준이 아니라 전월 대비 변화율의 IQR 기준으로 탐지한다.
    valid_changes = df["mom_pct"].dropna()
    q1, q3 = valid_changes.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["is_change_outlier"] = (df["mom_pct"] < lower) | (df["mom_pct"] > upper)
    df.to_csv(CLEAN_PATH, index=False, date_format="%Y-%m-%d")
    df.attrs.update(
        missing_before=missing_before,
        outlier_lower=float(lower),
        outlier_upper=float(upper),
    )
    return df


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(df: pd.DataFrame) -> dict[str, float | int | str]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    series = df.set_index("date")["passengers_thousands"]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(df["date"], df["passengers_thousands"], color="#4472C4", lw=1.5, label="Monthly")
    ax.plot(df["date"], df["ma_12"], color="#C00000", lw=2.5, label="12-month moving average")
    ax.set(title="International Airline Passengers and 12-Month Moving Average", xlabel="Date", ylabel="Passengers (thousands)")
    ax.legend()
    save_figure(fig, "01_trend_moving_average.png")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = np.where(df["yoy_pct"] >= 0, "#70AD47", "#C00000")
    ax.bar(df["date"], df["yoy_pct"], width=25, color=colors)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set(title="Year-over-Year Passenger Growth", xlabel="Date", ylabel="YoY change (%)")
    save_figure(fig, "02_yoy_growth.png")

    monthly = df.groupby("month")["passengers_thousands"].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly.index, monthly["mean"], color="#5B9BD5", label="Mean")
    ax.vlines(monthly.index, monthly["min"], monthly["max"], color="#264478", lw=2, label="Min–max")
    ax.set_xticks(range(1, 13))
    ax.set(title="Average Passengers by Calendar Month", xlabel="Month", ylabel="Passengers (thousands)")
    ax.legend()
    save_figure(fig, "03_monthly_seasonality.png")

    decomposition = seasonal_decompose(series, model="multiplicative", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    components = [
        (series, "Observed"),
        (decomposition.trend, "Trend"),
        (decomposition.seasonal, "Seasonal factor"),
        (decomposition.resid, "Residual"),
    ]
    for ax, (component, label) in zip(axes, components):
        ax.plot(component.index, component.values, color="#4472C4", lw=1.3)
        ax.set_ylabel(label)
    axes[0].set_title("Multiplicative Time-Series Decomposition")
    axes[-1].set_xlabel("Date")
    save_figure(fig, "04_decomposition.png")

    # 마지막 12개월을 숨긴 뒤, 12개월 전 같은 달을 예측값으로 쓰는 계절 나이브 검증.
    train = series.iloc[:-12]
    test = series.iloc[-12:]
    forecast = train.iloc[-12:].copy()
    forecast.index = test.index
    errors = test - forecast
    mae = float(errors.abs().mean())
    mape = float((errors.abs() / test).mean() * 100)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index[-36:], train.iloc[-36:], label="Training data", color="#7F7F7F")
    ax.plot(test.index, test, label="Actual (1960)", color="#4472C4", lw=2.3, marker="o")
    ax.plot(forecast.index, forecast, label="Seasonal-naive forecast", color="#ED7D31", lw=2, ls="--", marker="o")
    ax.set(title=f"12-Month Seasonal-Naive Backtest (MAE={mae:.1f}K, MAPE={mape:.1f}%)", xlabel="Date", ylabel="Passengers (thousands)")
    ax.legend()
    save_figure(fig, "05_seasonal_naive_forecast.png")

    peak_row = df.loc[df["passengers_thousands"].idxmax()]
    trough_row = df.loc[df["passengers_thousands"].idxmin()]
    first_12ma = float(df["ma_12"].dropna().iloc[0])
    last_12ma = float(df["ma_12"].dropna().iloc[-1])
    first_value = float(series.iloc[0])
    last_value = float(series.iloc[-1])
    best_yoy = df.loc[df["yoy_pct"].idxmax()]
    weakest_yoy = df.loc[df["yoy_pct"].idxmin()]
    july_mean = float(monthly.loc[7, "mean"])
    november_mean = float(monthly.loc[11, "mean"])

    summary: dict[str, float | int | str] = {
        "row_count": int(len(df)),
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "missing_before_processing": int(df.attrs["missing_before"]),
        "change_outlier_count": int(df["is_change_outlier"].sum()),
        "outlier_lower_pct": round(float(df.attrs["outlier_lower"]), 2),
        "outlier_upper_pct": round(float(df.attrs["outlier_upper"]), 2),
        "first_value_thousands": first_value,
        "last_value_thousands": last_value,
        "total_growth_pct": round((last_value / first_value - 1) * 100, 2),
        "cagr_pct": round(((last_value / first_value) ** (12 / (len(df) - 1)) - 1) * 100, 2),
        "first_12ma_thousands": round(first_12ma, 2),
        "last_12ma_thousands": round(last_12ma, 2),
        "moving_average_growth_pct": round((last_12ma / first_12ma - 1) * 100, 2),
        "peak_date": peak_row["date"].strftime("%Y-%m-%d"),
        "peak_value_thousands": float(peak_row["passengers_thousands"]),
        "trough_date": trough_row["date"].strftime("%Y-%m-%d"),
        "trough_value_thousands": float(trough_row["passengers_thousands"]),
        "best_yoy_date": best_yoy["date"].strftime("%Y-%m-%d"),
        "best_yoy_pct": round(float(best_yoy["yoy_pct"]), 2),
        "weakest_yoy_date": weakest_yoy["date"].strftime("%Y-%m-%d"),
        "weakest_yoy_pct": round(float(weakest_yoy["yoy_pct"]), 2),
        "july_mean_thousands": round(july_mean, 2),
        "november_mean_thousands": round(november_mean, 2),
        "july_vs_november_pct": round((july_mean / november_mean - 1) * 100, 2),
        "forecast_mae_thousands": round(mae, 2),
        "forecast_mape_pct": round(mape, 2),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    setup_style()
    df = load_data()
    summary = create_visualizations(df)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
