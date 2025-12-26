# XABSA: Cross-lingual Aspect-Based Sentiment Analysis

LLM Teacher 기반 Cross-lingual ABSA 시스템. 영어/한국어 리뷰에서 aspect-based sentiment triplet (term, category, polarity)를 추출합니다.

## 주요 특징

- **LLM Teacher**: GPT-4/Claude/Gemini를 사용한 pseudo-label 생성
- **Cross-lingual**: XLM-RoBERTa 기반 다국어 모델
- **필터링 시스템**: 5단계 pseudo-label 필터링
- **실험 재현 가능**: Config 기반 실험 관리
- **통합 데이터 포맷**: JSONL 기반 통일된 인터페이스

## 🚀 빠른 시작 (3분)

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/jaewoosoul/XABSA.git
cd XABSA

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

**Gemini 사용 (권장 - 무료 60 req/min)**:
```bash
# .env 파일 생성
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

**API 키 발급**: https://ai.google.dev/

<details>
<summary>다른 LLM 사용하기 (OpenAI, Claude)</summary>

```bash
# OpenAI
echo "OPENAI_API_KEY=sk-your-key" > .env

# Claude
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env
```

**상세 가이드**: [SETUP.md](SETUP.md) 참조
</details>

### 3. 한국어 데이터 생성

```bash
# 대화형 입력
python scripts/create_korean_data.py --output data/processed/ko_raw.jsonl

# 또는 CSV에서 로드
python scripts/create_korean_data.py \
  --csv your_reviews.csv \
  --text-column review_text \
  --output data/processed/ko_raw.jsonl
```

### 4. Pseudo-label 생성

```bash
# 전체 파이프라인 실행
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl

# 소량 테스트 (10개)
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --max-samples 10

# 또는 대화형 스크립트
bash run.sh
```

### 5. 결과 확인

```bash
# 생성된 pseudo-label 확인
cat data/pseudo/ko_pseudo.jsonl | head -1 | jq

# 필터링 통계 확인
cat data/pseudo/summary.json | jq
```

**예상 출력**:
```json
{
  "id": "ko_000001",
  "lang": "ko",
  "text": "배송이 정말 빠르고 품질도 좋아요!",
  "gold_triplets": [
    {"term": "배송", "category": "DELIVERY", "polarity": "positive"},
    {"term": "품질", "category": "QUALITY", "polarity": "positive"}
  ],
  "split": "unlabeled"
}
```

---

## 📚 상세 사용 가이드

**[→ Scripts CLI 사용법 보기 (SCRIPTS.md)](SCRIPTS.md)**

모든 스크립트의 CLI 옵션과 상세한 사용 예시를 확인하세요:
- 데이터 준비 스크립트 (create_korean_data.py, prepare_data.py 등)
- Pseudo-label 생성 (run_full_pipeline.py, run_teacher.py)
- 모델 학습 및 평가 (train.py, eval.py)
- 일반적인 워크플로우

---

## 📊 현재 프로젝트 상태

### ✅ Phase 1: 데이터 & Teacher (완료)

- [x] 통합 데이터 파이프라인 (JSONL 포맷)
- [x] LLM Teacher 구현 (OpenAI, Claude, Gemini, Mock)
- [x] 5단계 Pseudo-label 필터링 시스템
- [x] 한국어/영어 데이터 파서
- [x] 실행 스크립트 및 문서화

**생성된 데이터 품질**:
- 12개 샘플 테스트: 26개 triplet 생성
- 필터링 통과율: 100%
- Term 추출 정확도: ✅ 원문에서 정확히 추출
- Category 분류: ✅ Taxonomy 준수
- Polarity 판단: ✅ 문맥 반영

### ⏳ Phase 2: 모델 & 학습 (예정)

- [ ] XLM-RoBERTa 기반 Student 모델 구현
- [ ] Training 모듈 (Multi-task learning)
- [ ] Evaluation 모듈 (Triplet F1, ATE F1)
- [ ] Contrastive learning (cross-lingual alignment)

### ⏳ Phase 3: 실험 & 보고서 (예정)

- [ ] Baseline 실험 (EN → KO zero-shot)
- [ ] Pseudo-label 효과 검증
- [ ] Filtering ablation study
- [ ] Few-shot 실험 (10/50/100 샘플)
- [ ] `report.md` 자동 생성

---

## 프로젝트 구조

```
XABSA/
├── configs/                    # 설정 파일
│   ├── taxonomy.yaml           # 13개 카테고리 정의
│   ├── teacher.yaml            # LLM Teacher 설정
│   ├── baseline.yaml           # Baseline 실험
│   └── experiments/            # 5가지 실험 시나리오
│
├── data/                       # 데이터 디렉토리
│   ├── raw/                    # 원본 데이터 (SemEval, 한국어)
│   ├── processed/              # JSONL 포맷
│   └── pseudo/                 # Pseudo-labels
│
├── src/                        # 소스 코드
│   ├── data/                   # 데이터 처리
│   │   ├── taxonomy.py         # Taxonomy 관리
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── semeval_parser.py   # SemEval 파서
│   │   └── korean_parser.py    # Korean 파서
│   │
│   ├── teacher/                # LLM Teacher ✅
│   │   ├── base.py             # Base Teacher
│   │   ├── openai_teacher.py   # OpenAI
│   │   ├── claude_teacher.py   # Claude
│   │   ├── gemini_teacher.py   # Gemini
│   │   ├── prompts.py          # Prompts
│   │   ├── validator.py        # Validation
│   │   └── filter.py           # 5단계 필터링
│   │
│   ├── models/                 # Student Models (예정)
│   ├── training/               # Training (예정)
│   └── evaluation/             # Evaluation (예정)
│
├── scripts/                    # 실행 스크립트
│   ├── create_korean_data.py   # 한국어 데이터 생성 ✅
│   ├── run_full_pipeline.py    # 전체 파이프라인 ✅
│   ├── run_teacher.py          # Pseudo-label 생성 ✅
│   ├── prepare_data.py         # 데이터 전처리
│   ├── train.py                # 학습 (예정)
│   └── eval.py                 # 평가 (예정)
│
├── results/                    # 실험 결과
├── logs/                       # 로그
└── README.md                   # 이 파일
```

---

## 데이터 포맷

### 통합 JSONL 포맷

모든 데이터는 다음 포맷으로 통일:

```json
{
  "id": "unique_id",
  "lang": "ko",
  "text": "배송은 빠른데 포장이 부실했어요.",
  "gold_triplets": [
    {"term": "배송", "category": "DELIVERY", "polarity": "positive"},
    {"term": "포장", "category": "PACKAGING", "polarity": "negative"}
  ],
  "split": "train"
}
```

### Category Taxonomy

13개 도메인 일반형 카테고리:

| Category | 설명 | 예시 |
|----------|------|------|
| PRICE | 가격, 비용 | "가격이 저렴해요" |
| QUALITY | 품질 | "품질이 우수합니다" |
| DELIVERY | 배송 | "배송이 빨라요" |
| SERVICE | 서비스 | "친절한 응대" |
| DESIGN | 디자인, 외관 | "디자인이 예뻐요" |
| PERFORMANCE | 성능 | "성능이 좋습니다" |
| DURABILITY | 내구성 | "오래 쓸 수 있어요" |
| USABILITY | 사용성 | "사용하기 편해요" |
| PACKAGING | 포장 | "포장이 꼼꼼해요" |
| SIZE | 크기 | "크기가 적당해요" |
| RETURN | 반품/교환 | "반품이 쉬워요" |
| VALUE | 가성비 | "가성비가 좋아요" |
| ETC | 기타 | - |

전체 정의: [configs/taxonomy.yaml](configs/taxonomy.yaml)

---

## LLM Teacher

### 지원 모델

- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Google**: Gemini 1.5 Pro, Flash (권장)
- **Mock**: API 없이 테스트용

### 5단계 필터링 시스템

1. **Term existence check**: term이 원문에 substring으로 존재하는지 확인
2. **Deduplication**: 중복 triplet 제거 (공백/조사 정규화)
3. **Category validation**: taxonomy에 정의된 카테고리만 허용
4. **Triplet count limit**: 과도한 triplet 제거 (기본: 최대 8개)
5. **Self-consistency** (옵션): 동일 문장 3회 생성 → 합의된 triplet만 채택

### 사용 예시

```bash
# Gemini (권장)
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --teacher gemini \
  --filter

# OpenAI GPT-4
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --teacher openai \
  --model gpt-4-turbo-preview \
  --filter

# Claude
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --teacher claude \
  --filter

# Mock (API 없이 테스트)
python scripts/run_teacher.py \
  --input examples/sample_korean_reviews.csv \
  --output data/pseudo/test.jsonl \
  --teacher mock \
  --max-samples 10
```

---

## 실험 설계

프로젝트는 5가지 주요 실험을 지원:

| 실험 | 설명 | Config |
|------|------|--------|
| **Exp1: Baseline** | EN gold만 사용, KO zero-shot 평가 | `exp1_baseline_en.yaml` |
| **Exp2: Pseudo-label** | EN gold + KO pseudo | `exp2_pseudo_added.yaml` |
| **Exp3: Filtering ablation** | 필터링 전략 비교 | `exp3_filtering_ablation.yaml` |
| **Exp4: Contrastive** | Cross-lingual alignment | `exp4_contrastive.yaml` |
| **Exp5: Few-shot** | KO gold 10/50/100 샘플 추가 | `exp5_fewshot.yaml` |

**실험 실행** (예정):
```bash
python scripts/train.py --config configs/experiments/exp1_baseline_en.yaml
```

---

## 고급 사용법

### SemEval 영어 데이터 준비

```bash
# SemEval 다운로드 가이드
python scripts/download_semeval.py --create-dirs

# 데이터 파싱
python scripts/prepare_data.py \
  --semeval data/raw/semeval/restaurant \
  --out data/processed
```

### 필터링 옵션

```bash
# 필터링 없이 raw만 생성
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo_raw.jsonl \
  --no-filter

# Self-consistency 적용 (3회 생성)
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --teacher gemini \
  --filter \
  --self-consistency 3
```

### Config 커스터마이징

```yaml
# configs/teacher.yaml
teacher:
  type: "gemini"
  model: "gemini-1.5-flash"
  temperature: 0.0

filtering:
  check_term_existence: true
  remove_duplicates: true
  max_triplets_per_text: 8
  self_consistency_rounds: 0  # 0: 비활성화, 3: 3회 생성
```

---

## 문제 해결

### "API key not found" 에러

```bash
# .env 파일 확인
cat .env

# API 키 설정 확인
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('GOOGLE_API_KEY') else 'NOT FOUND')"
```

### Rate limit 에러

Gemini 무료 티어는 60 requests/minute 제한이 있습니다.

```bash
# 소량으로 나눠서 실행
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo_part1.jsonl \
  --max-samples 50
```

### 데이터 파싱 오류

```bash
# 상세 로그 확인
python scripts/prepare_data.py \
  --semeval data/raw/semeval/restaurant \
  --log-level DEBUG
```

---

## 개발 가이드

### 새로운 Parser 추가

```python
from src.data.taxonomy import Taxonomy

class MyParser:
    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy

    def parse(self, input_path: str) -> List[Dict]:
        # Parse and return JSONL format
        pass
```

### 새로운 Teacher 추가

```python
from src.teacher.base import BaseTeacher

class MyTeacher(BaseTeacher):
    def generate_triplets(self, text: str, lang: str) -> List[Dict]:
        # Generate triplets
        pass
```

---

## 기여하기

Pull requests를 환영합니다! 자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

---

## 라이선스

MIT License

---

## Citation

```bibtex
@software{xabsa2024,
  title={XABSA: Cross-lingual Aspect-Based Sentiment Analysis},
  author={Jaewoo Soul},
  year={2024},
  url={https://github.com/jaewoosoul/XABSA}
}
```

---

## 문의

Issues: https://github.com/jaewoosoul/XABSA/issues
