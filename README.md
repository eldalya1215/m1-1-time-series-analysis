# M1-1 시계열 데이터 분석

1949~1960년 월별 국제선 승객 수(`AirPassengers`, 144개 관측치)의 추세와 계절성을 분석한 프로젝트입니다.

## 폴더 구조

```text
M1-1/
├─ analysis.py
├─ REPORT.md
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ air_passengers_raw.csv
│  ├─ air_passengers_clean.csv
│  └─ analysis_summary.json
└─ images/
   ├─ 01_trend_moving_average.png
   ├─ 02_yoy_growth.png
   ├─ 03_monthly_seasonality.png
   ├─ 04_decomposition.png
   └─ 05_seasonal_naive_forecast.png
```

## 실행 방법

Python 3.10 이상에서 다음 명령을 순서대로 실행합니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
python analysis.py
```

`analysis.py`는 원본 CSV가 없을 때 공개된 Rdatasets 미러에서 데이터를 내려받습니다. 실행 후 정제 CSV, 요약 JSON, PNG 시각화 5개가 다시 생성됩니다.

## 데이터 출처 및 이용 주의

- 데이터셋: R 기본 `datasets::AirPassengers`
- 설명: 1949년 1월~1960년 12월 월별 국제선 승객 수(천 명 단위), 총 144개 관측치
- 공식 문서: <https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html>
- 수집 CSV: <https://github.com/vincentarelbundock/Rdatasets/tree/master/csv/datasets>
- 원전: Box, Jenkins and Reinsel, *Time Series Analysis, Forecasting and Control*, Series G

학습·재현 목적의 공개 예제 데이터로 사용했습니다. 원출처의 권리와 인용 조건을 확인한 뒤 상업적 재배포에 사용해야 합니다.
