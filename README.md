# XABSA: Cross-lingual Aspect-Based Sentiment Analysis

LLM Teacher 기반 Cross-lingual ABSA 시스템. 영어/한국어 리뷰에서 aspect-based sentiment triplet (term, category, polarity)를 추출합니다.

## 주요 특징

- **LLM Teacher**: GPT-4/Claude/Gemini를 사용한 pseudo-label 생성
- **Cross-lingual**: XLM-RoBERTa 기반 다국어 모델
- **Multi-task Learning**: ATE, Category, Polarity 동시 학습
- **필터링 시스템**: 5단계 pseudo-label 필터링
- **Config 기반 실험**: 재현 가능한 실험 관리

## 🚀 빠른 시작

### 요구사항
- **Python 3.11** (필수)
- CUDA 지원 GPU (권장)

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/jaewoosoul/XABSA.git
cd XABSA

# 가상환경 생성 (Python 3.11 사용)
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
# .env 파일 생성
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

**API 키 발급**: https://ai.google.dev/

### 3. 전체 파이프라인 실행

```bash
# 1. 한국어 데이터 생성 (CSV에서)
python scripts/create_korean_data.py \
  --csv data/raw/korean/reviews.csv \
  --text-column Review \
  --output data/processed/ko_raw.jsonl

# 2. Pseudo-label 생성
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl

# 3. 모델 학습
python scripts/train.py --config configs/experiments/ko_only.yaml

# 4. 모델 평가
python scripts/eval.py \
  --config configs/experiments/ko_only.yaml \
  --ckpt results/checkpoints/ko_only/best_model.pt
```

---

## 📊 프로젝트 상태

### ✅ Phase 1: 데이터 & Teacher (완료)
- 통합 데이터 파이프라인 (JSONL)
- LLM Teacher (OpenAI, Claude, Gemini)
- 5단계 필터링 시스템

### ✅ Phase 2: 모델 & 학습 (완료)
- XLM-RoBERTa 기반 Student 모델
- Multi-task learning (ATE + Category + Polarity)
- 평가 모듈 (Triplet F1, ATE F1 등)
- 체크포인트 관리 및 early stopping

### ⏳ Phase 3: 실험 & 보고서 (예정)
- Baseline 실험 (EN → KO zero-shot)
- Pseudo-label 효과 검증
- Few-shot 실험

---

## 프로젝트 구조

```
XABSA/
├── configs/                    # 설정 파일
│   ├── taxonomy.yaml           # 13개 카테고리 정의
│   ├── teacher.yaml            # LLM Teacher 설정
│   └── experiments/            # 실험 시나리오
│
├── src/
│   ├── data/                   # 데이터 처리
│   ├── teacher/                # LLM Teacher
│   ├── models/                 # Student 모델 ✅
│   ├── training/               # 학습 모듈 ✅
│   └── evaluation/             # 평가 모듈 ✅
│
├── scripts/
│   ├── create_korean_data.py  # 데이터 생성
│   ├── run_full_pipeline.py   # Pseudo-label 생성
│   ├── train.py                # 모델 학습 ✅
│   └── eval.py                 # 모델 평가 ✅
│
└── data/                       # 데이터 (gitignore)
```

---

## 📚 상세 가이드

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: 학습 및 평가 상세 가이드
- **[SCRIPTS.md](scripts/SCRIPTS.md)**: 모든 스크립트 CLI 옵션
- **[data/README.md](data/README.md)**: 데이터 포맷 및 준비 방법

## 데이터 포맷

```json
{
  "id": "ko_000001",
  "lang": "ko",
  "text": "배송은 빠른데 포장이 부실했어요.",
  "gold_triplets": [
    {"term": "배송", "category": "DELIVERY", "polarity": "positive"},
    {"term": "포장", "category": "PACKAGING", "polarity": "negative"}
  ],
  "split": "train"
}
```

**13개 카테고리**: PRICE, QUALITY, DELIVERY, SERVICE, DESIGN, PERFORMANCE, DURABILITY, USABILITY, PACKAGING, SIZE, RETURN, VALUE, ETC

전체 정의: [configs/taxonomy.yaml](configs/taxonomy.yaml)

---

## 실험 실행

```bash
# Baseline (영어만)
python scripts/train.py --config configs/baseline.yaml

# Pseudo-label 추가
python scripts/train.py --config configs/experiments/exp2_pseudo_added.yaml

# 한국어만
python scripts/train.py --config configs/experiments/ko_only.yaml
```

모든 실험 설정: `configs/experiments/`

---

## 라이선스

MIT License

## Citation

```bibtex
@software{xabsa2024,
  title={XABSA: Cross-lingual Aspect-Based Sentiment Analysis},
  author={Jaewoo Soul},
  year={2024},
  url={https://github.com/jaewoosoul/XABSA}
}
```

