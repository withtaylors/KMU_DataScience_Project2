# 🎬 Netflix 카탈로그 구조 분석 & 등급 그룹 분류 — Project 2

> **데이터사이언스실무 | 다중 분류 | 카탈로그 구조 EDA + XGBoost**

---

## 📌 비즈니스 문제

> 텍스트(제목·줄거리·장르)로 콘텐츠 **연령 등급 그룹(rating_group)을 자동 분류**하여  
> 넷플릭스 콘텐츠 검수 비용을 줄이고, 위험 등급 오류를 빠르게 포착할 수 있는가?

```
Net Impact = TP × B_TP  -  (TP + FP) × C_review  -  FN × C_miss
→ C_miss(성인 등급 누락 리스크) >> C_review(검수 비용)
→ Adults Recall 극대화가 핵심 전략
```

> ⚠️ 이 데이터는 카탈로그 메타데이터 — 인기 분석·추천시스템·전략 인과해석은 이번 범위 외

---

## 📂 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle — Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) |
| 규모 | 8,807행 / 12컬럼 |
| 타깃 | `rating_group` — Adults / Teens / Kids / Unknown |
| 문제 유형 | 다중 분류 (원본 14클래스 → 4그룹으로 통합) |
| 라이선스 | CC0 Public Domain |

---

## 📊 핵심 EDA 결과

### 타깃 클래스 분포

```
Adults  45.6%  ████████████████████████
Teens   43.2%  ██████████████████████
Kids    10.3%  █████
Unknown  0.9%  ▏
불균형 비율: Adults:Kids = 4.4:1  (원본 rating 최대 1,069:1 대비 크게 개선)
```

### EDA 4축 인사이트

| 축 | 핵심 발견 |
|----|-----------|
| **카탈로그 구조** | Movie 중심(69.6%) + TV Show(30.4%) 보완 구조 |
| **시계열** | 2016년부터 본격 확장, Movie 유지하며 TV Show 함께 확대 |
| **국가별 포지션** | Spain(81.8%) / Mexico(76.9%) Adult 집중 / India Teen 56.3% 1위 / US≈UK / Korea≈Japan |
| **편입 시차** | TV Show 중앙값 0년(신작 중심) / Movie 중앙값 1년(구작 라이브러리도 포함) |

> `country`, `listed_in` 다중값 컬럼은 **가중치(1/n) 방식**으로 처리 (explode 대비 중복 카운팅 방지)

---

## 🔧 방법론

### 불균형 처리 파이프라인

```
1. class_weight='balanced'    → 소수 클래스(Kids) 가중치 부여
2. SMOTE (Pipeline 내부)      → Train fold에만 적용, 데이터 누수 방지
3. Threshold Tuning           → Adults: 0.5 → 0.35 (Net Impact 기반)
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
`description` TF-IDF · `title` TF-IDF · `listed_in` 가중치 OHE · `type` · `country` · `release_year` · `director_missing` · `genre_count`

---

## 🎯 성공 기준

| 메트릭 | 목표 |
|--------|------|
| Macro F1 | ≥ 0.70 |
| Adults Recall | ≥ 0.80 |
| ROC-AUC (OvR) | ≥ 0.88 |

> ⚠️ Accuracy 단독 사용 금지 — Adults만 예측해도 46% 달성

---

## ⚠️ 주요 리스크 & 한계

| 리스크 | 대응 |
|--------|------|
| `director` 29.9% 결측 | 결측 여부 이진 피처 변환 |
| 다중값 컬럼 (`country`, `listed_in`) | 가중치(1/n) 처리, 전처리 규칙 문서화 |
| Unknown(NR/UR) 83건 | 분류 제외 또는 별도 처리 검토 |
| Netflix Engagement Report 조인 | 매칭률 30% 미만 → 이번 단계 보류 |
| 소비 국가 데이터 없음 | 제작 국가 기준 해석, 한계 명시 |
| 2021년 이전 데이터 | 최신 트렌드 반영 불가 |

---

## 🗓️ 타임라인

```
Week 1-2  ✅ 데이터 이해 & 기획안
Week 3-4     EDA 4축 분석 → rating_group 변환 → XGBoost + SMOTE Pipeline
Week 5       Net Impact 기반 Threshold Tuning + 국가 유사도 분석
Week 6       Kaggle 제출 + 발표
```

---

## 📁 파일 구조

```
├── netflix_titles.csv          # 원본 데이터
├── netflix_1pager_v2.ipynb     # 1-Pager 노트북 (본 파일)
└── README.md
```
