# Scripts 사용 가이드

XABSA 프로젝트의 모든 스크립트 CLI 옵션을 정리한 문서입니다.

## 목차

1. [데이터 준비](#1-데이터-준비)
   - [create_korean_data.py](#create_korean_datapy) - 한국어 데이터 생성
   - [download_semeval.py](#download_semevalpy) - SemEval 다운로드 가이드
   - [prepare_data.py](#prepare_datapy) - 데이터 전처리
2. [Pseudo-label 생성](#2-pseudo-label-생성)
   - [run_full_pipeline.py](#run_full_pipelinepy) - 전체 파이프라인 실행
   - [run_teacher.py](#run_teacherpy) - LLM Teacher 실행
3. [모델 학습 및 평가](#3-모델-학습-및-평가)
   - [train.py](#trainpy) - 모델 학습
   - [eval.py](#evalpy) - 모델 평가

---

## 1. 데이터 준비

### create_korean_data.py

한국어 리뷰 데이터를 생성하여 JSONL 포맷으로 변환합니다.

#### 필수 인자

| 옵션 | 타입 | 설명 |
|------|------|------|
| `--output` | string | 출력 JSONL 파일 경로 |

#### 선택 인자

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--csv` | string | - | 입력 CSV 파일 경로 |
| `--text-column` | string | `text` | CSV의 텍스트 컬럼 이름 |
| `--id-column` | string | - | CSV의 ID 컬럼 이름 (선택사항) |
| `--txt` | string | - | 입력 텍스트 파일 경로 (한 줄에 하나씩) |

#### 사용 예시

```bash
# 대화형 입력
python scripts/create_korean_data.py --output data/processed/ko_raw.jsonl

# CSV 파일에서 로드
python scripts/create_korean_data.py \
  --csv reviews.csv \
  --text-column review_text \
  --output data/processed/ko_raw.jsonl

# 텍스트 파일에서 로드
python scripts/create_korean_data.py \
  --txt reviews.txt \
  --output data/processed/ko_raw.jsonl

# ID 컬럼 지정
python scripts/create_korean_data.py \
  --csv reviews.csv \
  --text-column content \
  --id-column review_id \
  --output data/processed/ko_raw.jsonl
```

---

### download_semeval.py

SemEval ABSA 데이터셋 다운로드 가이드 및 디렉토리 구조 생성.

#### 선택 인자

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--year` | string | `all` | SemEval 연도 (`2014`, `2015`, `2016`, `all`) |
| `--create-dirs` | flag | - | 다운로드용 디렉토리 구조 생성 |
| `--log-level` | string | `INFO` | 로그 레벨 (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

#### 사용 예시

```bash
# 모든 연도의 다운로드 가이드 출력
python scripts/download_semeval.py

# 2014년 데이터 가이드만 출력
python scripts/download_semeval.py --year 2014

# 디렉토리 구조 생성
python scripts/download_semeval.py --create-dirs

# 디렉토리 생성 + 특정 연도
python scripts/download_semeval.py --year 2016 --create-dirs
```

---

### prepare_data.py

SemEval/한국어 원본 데이터를 통합 JSONL 포맷으로 변환합니다.

#### 선택 인자

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--semeval` | string | - | SemEval 데이터 디렉토리 경로 (XML 파일 포함) |
| `--korean` | string | - | 한국어 데이터 파일 경로 (CSV/JSON/JSONL) |
| `--korean-format` | string | `auto` | 한국어 데이터 포맷 (`auto`, `csv`, `json`, `jsonl`) |
| `--korean-text-column` | string | `text` | CSV의 텍스트 컬럼 이름 |
| `--korean-has-labels` | flag | - | 한국어 데이터에 레이블 포함 여부 |
| `--out` | string | `data/processed` | 출력 디렉토리 |
| `--taxonomy` | string | `configs/taxonomy.yaml` | Taxonomy 파일 경로 |
| `--lang` | string | `en` | SemEval 데이터 언어 |
| `--log-level` | string | `INFO` | 로그 레벨 |

#### 사용 예시

```bash
# SemEval 데이터 파싱
python scripts/prepare_data.py \
  --semeval data/raw/semeval/restaurant \
  --out data/processed

# 한국어 데이터 파싱 (CSV)
python scripts/prepare_data.py \
  --korean data/raw/korean/reviews.csv \
  --korean-format csv \
  --out data/processed

# 레이블이 있는 한국어 데이터
python scripts/prepare_data.py \
  --korean data/raw/korean/labeled_reviews.json \
  --korean-has-labels \
  --out data/processed

# 둘 다 파싱
python scripts/prepare_data.py \
  --semeval data/raw/semeval/restaurant \
  --korean data/raw/korean/reviews.csv \
  --out data/processed

# 커스텀 텍스트 컬럼명
python scripts/prepare_data.py \
  --korean reviews.csv \
  --korean-text-column review_content \
  --out data/processed

# DEBUG 로그로 상세 정보 확인
python scripts/prepare_data.py \
  --semeval data/raw/semeval/restaurant \
  --log-level DEBUG
```

---

## 2. Pseudo-label 생성

### run_full_pipeline.py

**[추천]** Gemini를 사용한 전체 파이프라인 실행 (데이터 로드 → pseudo-label 생성 → 필터링 → 저장).

#### 필수 인자

| 옵션 | 타입 | 설명 |
|------|------|------|
| `--input` | string | 입력 JSONL 파일 (unlabeled 한국어 리뷰) |
| `--output` | string | 출력 JSONL 파일 (pseudo-labeled 데이터) |

#### 선택 인자

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | string | `configs/teacher.yaml` | 설정 파일 경로 |
| `--model` | string | `gemini-flash-latest` | Gemini 모델명 |
| `--max-samples` | int | - | 처리할 최대 샘플 수 (테스트용) |
| `--no-filter` | flag | - | 필터링 없이 raw pseudo-label만 생성 |
| `--log-level` | string | `INFO` | 로그 레벨 |

#### 사용 예시

```bash
# 기본 실행
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl

# 소량 테스트 (10개 샘플만)
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --max-samples 10

# Gemini Pro 모델 사용
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --model gemini-1.5-pro

# 필터링 없이 raw만 생성
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo_raw.jsonl \
  --no-filter

# 커스텀 config 사용
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --config configs/teacher_custom.yaml
```

---

### run_teacher.py

다양한 LLM Teacher (OpenAI, Claude, Gemini, Mock)를 사용한 pseudo-label 생성.

#### 필수 인자

| 옵션 | 타입 | 설명 |
|------|------|------|
| `--input` | string | 입력 JSONL 파일 (unlabeled 데이터) |
| `--output` | string | 출력 JSONL 파일 (pseudo-labeled 데이터) |

#### 선택 인자

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | string | - | Teacher 설정 파일 경로 (선택사항) |
| `--teacher` | string | `gemini` | Teacher 타입 (`openai`, `claude`, `gemini`, `mock`) |
| `--model` | string | - | 모델명 (config 오버라이드) |
| `--taxonomy` | string | `configs/taxonomy.yaml` | Taxonomy 파일 경로 |
| `--filter` | flag | - | 필터링 적용 |
| `--save-raw` | flag | - | 필터링 전 raw pseudo-label 저장 |
| `--max-samples` | int | - | 최대 샘플 수 (테스트용) |
| `--log-level` | string | `INFO` | 로그 레벨 |

#### 사용 예시

```bash
# Gemini (기본)
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
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
  --model claude-3-sonnet-20240229 \
  --filter

# Mock Teacher (API 없이 테스트)
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo_test.jsonl \
  --teacher mock \
  --max-samples 10

# Config 파일 사용
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --config configs/teacher.yaml \
  --filter

# Raw + Filtered 둘 다 저장
python scripts/run_teacher.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --teacher gemini \
  --filter \
  --save-raw
```

---

## 3. 모델 학습 및 평가

### train.py

XLM-RoBERTa 기반 Student 모델 학습.

#### 필수 인자

| 옵션 | 타입 | 설명 |
|------|------|------|
| `--config` | string | 설정 파일 경로 (YAML) |

#### 선택 인자

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--resume` | string | - | 재개할 체크포인트 경로 |

#### 사용 예시

```bash
# Baseline 실험 (EN gold만 사용)
python scripts/train.py \
  --config configs/experiments/exp1_baseline_en.yaml

# Pseudo-label 추가 실험
python scripts/train.py \
  --config configs/experiments/exp2_pseudo_added.yaml

# 필터링 ablation
python scripts/train.py \
  --config configs/experiments/exp3_filtering_ablation.yaml

# Contrastive learning 실험
python scripts/train.py \
  --config configs/experiments/exp4_contrastive.yaml

# Few-shot 실험
python scripts/train.py \
  --config configs/experiments/exp5_fewshot.yaml

# 체크포인트에서 재개
python scripts/train.py \
  --config configs/experiments/exp1_baseline_en.yaml \
  --resume results/checkpoints/baseline/checkpoint-epoch-5.pt
```

#### Config 파일 주요 설정

```yaml
# 예시: configs/experiments/exp1_baseline_en.yaml
experiment:
  name: "baseline_en"
  seed: 42

data:
  train_paths:
    - "data/processed/en_train.jsonl"
  eval_paths:
    - "data/processed/ko_test.jsonl"
  max_length: 128

model:
  backbone: "xlm-roberta-base"
  type: "joint"  # joint | pipeline
  dropout: 0.1
  freeze_backbone: false

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 500
  gradient_accumulation_steps: 1

output:
  output_dir: "results/baseline_en"
  save_best_by: "triplet_f1"
```

---

### eval.py

학습된 모델 평가.

#### 필수 인자

| 옵션 | 타입 | 설명 |
|------|------|------|
| `--config` | string | 설정 파일 경로 (YAML) |
| `--ckpt` | string | 평가할 체크포인트 경로 |

#### 선택 인자

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--split` | string | - | 평가할 특정 split (예: `test`) |
| `--save-predictions` | flag | - | 예측 결과 저장 |

#### 사용 예시

```bash
# 기본 평가 (config의 eval_paths 사용)
python scripts/eval.py \
  --config configs/experiments/exp1_baseline_en.yaml \
  --ckpt results/baseline_en/checkpoints/best.pt

# 특정 split만 평가
python scripts/eval.py \
  --config configs/experiments/exp1_baseline_en.yaml \
  --ckpt results/baseline_en/checkpoints/best.pt \
  --split test

# 예측 결과 저장
python scripts/eval.py \
  --config configs/experiments/exp1_baseline_en.yaml \
  --ckpt results/baseline_en/checkpoints/best.pt \
  --save-predictions

# 여러 체크포인트 비교
python scripts/eval.py \
  --config configs/experiments/exp1_baseline_en.yaml \
  --ckpt results/baseline_en/checkpoints/checkpoint-epoch-5.pt

python scripts/eval.py \
  --config configs/experiments/exp1_baseline_en.yaml \
  --ckpt results/baseline_en/checkpoints/checkpoint-epoch-10.pt
```

#### 출력 메트릭

평가 시 다음 메트릭이 계산됩니다:

- **ATE (Aspect Term Extraction)**
  - Precision, Recall, F1
- **Category Classification**
  - Accuracy, F1 (macro)
- **Polarity Classification**
  - Accuracy, F1 (macro)
- **Triplet Extraction (전체)**
  - Precision, Recall, F1

결과는 `{output_dir}/eval_metrics.json`에 저장됩니다.

---

## 일반적인 워크플로우

### 1. 데이터 준비 단계

```bash
# 1-1. 한국어 데이터 생성
python scripts/create_korean_data.py \
  --output data/processed/ko_raw.jsonl

# 1-2. (옵션) SemEval 데이터 준비
python scripts/download_semeval.py --create-dirs
# ... SemEval XML 파일 수동 다운로드 후 ...
python scripts/prepare_data.py \
  --semeval data/raw/semeval/2014_restaurant \
  --out data/processed
```

### 2. Pseudo-label 생성

```bash
# 2-1. 소량 테스트
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --max-samples 10

# 2-2. 전체 실행
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl
```

### 3. 모델 학습

```bash
# 3-1. Baseline (EN gold만)
python scripts/train.py \
  --config configs/experiments/exp1_baseline_en.yaml

# 3-2. Pseudo-label 추가
python scripts/train.py \
  --config configs/experiments/exp2_pseudo_added.yaml
```

### 4. 평가

```bash
# 4-1. Baseline 평가
python scripts/eval.py \
  --config configs/experiments/exp1_baseline_en.yaml \
  --ckpt results/baseline_en/checkpoints/best.pt \
  --save-predictions

# 4-2. Pseudo-label 모델 평가
python scripts/eval.py \
  --config configs/experiments/exp2_pseudo_added/checkpoints/best.pt \
  --ckpt results/pseudo_added/checkpoints/best.pt \
  --save-predictions
```

---

## 환경 변수

스크립트 실행 전 필요한 API 키를 `.env` 파일에 설정하세요:

```bash
# Gemini (권장)
GOOGLE_API_KEY=your-api-key-here

# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## 문제 해결

### API 키 에러

```bash
# .env 파일 확인
cat .env

# API 키 설정 확인
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if os.getenv('GOOGLE_API_KEY') else 'NOT FOUND')"
```

### Rate limit 에러

```bash
# 소량으로 나눠서 실행
python scripts/run_full_pipeline.py \
  --input data/processed/ko_raw.jsonl \
  --output data/pseudo/ko_pseudo.jsonl \
  --max-samples 50
```

### CUDA Out of Memory

```bash
# Config 파일에서 batch_size 줄이기
# configs/experiments/exp1_baseline_en.yaml
training:
  batch_size: 8  # 16 → 8로 감소
```

---

## 추가 자료

- [README.md](README.md) - 프로젝트 개요 및 빠른 시작
- [SETUP.md](SETUP.md) - 설치 및 환경 설정
- [CONTRIBUTING.md](CONTRIBUTING.md) - 기여 가이드
