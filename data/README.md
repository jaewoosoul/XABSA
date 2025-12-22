# XABSA Data Directory

이 디렉토리는 XABSA 프로젝트의 모든 데이터를 관리합니다.

## 디렉토리 구조

```
data/
├── raw/                    # 원본 데이터 (git에 포함 안 됨)
│   ├── semeval/           # SemEval ABSA 데이터
│   │   ├── 2014_restaurant/
│   │   ├── 2014_laptop/
│   │   └── ...
│   └── korean/            # 한국어 원본 데이터
│       ├── reviews.csv
│       └── ...
│
├── processed/             # 통합 JSONL 포맷 (git에 포함 안 됨)
│   ├── en_train.jsonl    # 영어 학습 데이터
│   ├── en_dev.jsonl      # 영어 검증 데이터
│   ├── en_test.jsonl     # 영어 테스트 데이터
│   ├── ko_raw.jsonl      # 한국어 unlabeled 데이터
│   └── ko_gold.jsonl     # 한국어 gold 데이터 (선택)
│
├── pseudo/               # Pseudo-label 데이터 (git에 포함 안 됨)
│   ├── ko_pseudo_raw.jsonl   # 필터링 전
│   ├── ko_pseudo.jsonl       # 필터링 후
│   └── summary.json          # 필터링 통계
│
└── README.md             # 본 파일
```

## 데이터 포맷

### 통합 JSONL 포맷

모든 데이터는 다음 포맷으로 통일됩니다:

```json
{
  "id": "unique_id",
  "lang": "ko",
  "text": "리뷰 텍스트",
  "gold_triplets": [
    {
      "term": "배송",
      "category": "DELIVERY",
      "polarity": "positive"
    }
  ],
  "split": "train"
}
```

**필드 설명**:
- `id`: 고유 식별자
- `lang`: 언어 (ko, en)
- `text`: 원본 텍스트
- `gold_triplets`: Triplet 리스트 (없으면 빈 배열)
- `split`: train, dev, test, unlabeled

### Triplet 포맷

각 triplet은 3개 필드로 구성:

```json
{
  "term": "string",      // 텍스트에서 추출된 aspect term
  "category": "CATEGORY", // Taxonomy의 카테고리
  "polarity": "positive"  // positive, negative, neutral
}
```

**중요**: `term`은 반드시 원본 텍스트에 substring으로 존재해야 합니다.

## 빠른 시작

### 샘플 데이터로 시작

```bash
# 예제 실행 (자동으로 데이터 준비)
bash examples/run_example.sh
```

### SemEval 데이터 준비

```bash
# 1. 다운로드 가이드
python3 scripts/download_semeval.py --create-dirs

# 2. 다운로드 (수동)
# http://alt.qcri.org/semeval2014/task4/

# 3. 파싱
python3 scripts/prepare_data.py \
  --semeval data/raw/semeval/2014_restaurant \
  --out data/processed
```

### 한국어 데이터 준비

```bash
python3 scripts/prepare_data.py \
  --korean data/raw/korean/reviews.csv \
  --korean-format csv \
  --out data/processed
```

## 상세 가이드

자세한 내용은 각 섹션을 참조하세요.

### Category Taxonomy

13개 도메인 일반형 카테고리 지원:
- PRICE, QUALITY, DELIVERY, SERVICE, DESIGN
- PERFORMANCE, DURABILITY, USABILITY, PACKAGING
- SIZE, RETURN, VALUE, ETC

`configs/taxonomy.yaml` 파일에서 관리됩니다.

### 데이터 통계 확인

```bash
# 샘플 수
wc -l data/processed/*.jsonl

# Triplet 통계
python3 << EOF
import json
with open('data/processed/en_train.jsonl') as f:
    samples = [json.loads(line) for line in f]
print(f"Samples: {len(samples)}")
print(f"Triplets: {sum(len(s['gold_triplets']) for s in samples)}")
EOF
```

## 참고 자료

- **configs/taxonomy.yaml**: Category taxonomy
- **scripts/prepare_data.py**: 데이터 준비
- **scripts/run_teacher.py**: Pseudo-label 생성
- **examples/**: 예제 데이터 및 스크립트
