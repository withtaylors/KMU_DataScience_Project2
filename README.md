# 🎬 Netflix 콘텐츠 등급 분류 — Project 2

> **데이터사이언스실무 | 다중 분류 | NLP + XGBoost**

---

## 📌 비즈니스 문제

> 제목·줄거리·장르 텍스트만으로 콘텐츠 **연령 등급(rating)을 자동 분류**할 수 있는가?

넷플릭스 신규 콘텐츠 온보딩 시 등급 심사 자동화 → **인력 비용 절감 + 규제 리스크 대응**

---

## 📂 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle — Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) |
| 규모 | 8,807행 / 12컬럼 |
| 타깃 | `rating` (14개 등급 → 전처리 후 11개) |
| 문제 유형 | 다중 분류 (Multi-class Classification) |
| 라이선스 | CC0 Public Domain |

---

## 📊 핵심 EDA 결과

### 클래스 불균형
```
TV-MA  36.4%  ████████████████████
TV-14  24.5%  █████████████
TV-PG   9.8%  █████
R       9.1%  ████
...
NC-17   0.03% ▏  ← 불균형 비율 최대 1,069:1
```
→ **최대 클래스 TV-MA 36.4% — 불균형 데이터**, Accuracy 지표 사용 불가

### 국가별 등급 편중 (실측값)

| 국가 | Adult | Teen | 주요 인사이트 |
|------|-------|------|--------------|
| 스페인 | **81.8%** | 10.5% | 성인 콘텐츠 문화적 집중 |
| 멕시코 | **76.9%** | 11.9% | 동일 패턴 |
| 인도 | 26.4% | **56.3%** | 10대 인구 비중 + 넷플릭스 전략 |
| 미국 | 48.6% | 25.0% | 영국(52.4%)과 유사 |
| 한국 | 44.1% | 39.8% | 일본(35.3% Adult)과 유사 |

---

## 🔧 방법론

### 불균형 처리 파이프라인 (4단계)
```
1. class_weight='balanced'    → 소수 클래스 가중치 부여
2. SMOTE (Pipeline 내부)      → Train fold에만 적용, 데이터 누수 방지
3. Threshold Tuning           → TV-MA: 0.5 → 0.35 (Recall 우선)
4. Isotonic Calibration       → 확률 보정 후 threshold 신뢰도 향상
```

### 모델 전략

| 단계 | 모델 | 역할 |
|------|------|------|
| Baseline | Logistic Regression + TF-IDF | 빠른 벤치마크 |
| Model 1 | Random Forest + Genre OHE | 피처 중요도 해석 |
| **Model 2** | **XGBoost** + class_weight + Calibration | 최종 목표 모델 |
| 검증 | Stratified 5-Fold CV | 클래스 비율 유지 |

### 주요 피처
`description` TF-IDF · `title` TF-IDF · `listed_in` OHE · `type` · `country` · `release_year` · `director_missing` · `genre_count`

---

## 🎯 성공 기준

| 메트릭 | 목표 |
|--------|------|
| Macro F1 | ≥ 0.65 |
| Weighted Recall | ≥ 0.72 |
| ROC-AUC (OvR) | ≥ 0.85 |

> ⚠️ Accuracy 단독 사용 금지 — TV-MA만 예측해도 36.4% 달성

---

## ⚠️ 주요 리스크

| 리스크 | 대응 |
|--------|------|
| `director` 29.9% 결측 | 결측 여부 이진 피처로 변환 |
| NC-17·UR 각 3건 | 유사 등급 병합 처리 |
| 소비 국가 데이터 없음 | 외부 데이터 결합 검토 |
| 2021년 이전 데이터 | 최신 트렌드 반영 불가 — 한계 명시 |

---

## 🗓️ 타임라인

```
Week 1-2  ✅ 데이터 이해 & 기획안
Week 3-4     EDA → Feature Engineering → XGBoost + SMOTE Pipeline
Week 5       Threshold Tuning + Calibration + 추천 시스템 프로토타입
Week 6       Kaggle 제출 + 발표
```

---

## 📁 파일 구조

```
├── netflix_titles.csv        # 원본 데이터
├── netflix_1pager.ipynb      # 1-Pager 노트북 (본 파일)
└── README.md
```
