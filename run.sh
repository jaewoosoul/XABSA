#!/bin/bash
# XABSA 프로젝트 실행 스크립트

set -e  # 에러 시 중단

echo "=========================================="
echo "XABSA Project - Quick Start"
echo "=========================================="
echo ""

# 1. API 키 확인
echo "Step 1: API 키 확인..."
if [ -z "$GOOGLE_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
    if [ ! -f .env ]; then
        echo "❌ .env 파일이 없습니다!"
        echo ""
        echo "다음 명령어를 실행하세요:"
        echo "  cp .env.example .env"
        echo "  # .env 파일을 열어서 GOOGLE_API_KEY 입력"
        exit 1
    fi

    # .env 파일 로드
    export $(cat .env | grep -v '^#' | xargs)

    if [ -z "$GOOGLE_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
        echo "❌ .env 파일에 API 키가 없습니다!"
        echo ""
        echo ".env 파일을 열어서 다음을 추가하세요:"
        echo "  GOOGLE_API_KEY=your-api-key-here"
        echo ""
        echo "API 키 발급: https://ai.google.dev/"
        exit 1
    fi
fi
echo "✓ API 키 확인 완료"
echo ""

# 2. 데이터 확인
echo "Step 2: 데이터 확인..."
if [ ! -f data/processed/ko_raw.jsonl ]; then
    echo "⚠️  한국어 데이터가 없습니다."
    echo ""
    echo "데이터를 생성하시겠습니까? (y/n)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        echo "데이터 생성 방법을 선택하세요:"
        echo "  1) 직접 입력 (대화형)"
        echo "  2) CSV 파일에서 로드"
        echo "  3) 텍스트 파일에서 로드"
        read -r choice

        case $choice in
            1)
                python scripts/create_korean_data.py --output data/processed/ko_raw.jsonl
                ;;
            2)
                echo "CSV 파일 경로를 입력하세요:"
                read -r csv_path
                echo "텍스트 컬럼 이름을 입력하세요 (기본값: text):"
                read -r text_col
                text_col=${text_col:-text}
                python scripts/create_korean_data.py \
                    --csv "$csv_path" \
                    --text-column "$text_col" \
                    --output data/processed/ko_raw.jsonl
                ;;
            3)
                echo "텍스트 파일 경로를 입력하세요:"
                read -r txt_path
                python scripts/create_korean_data.py \
                    --txt "$txt_path" \
                    --output data/processed/ko_raw.jsonl
                ;;
            *)
                echo "잘못된 선택입니다."
                exit 1
                ;;
        esac
    else
        echo "데이터 생성을 건너뜁니다."
        exit 0
    fi
else
    num_lines=$(wc -l < data/processed/ko_raw.jsonl)
    echo "✓ 데이터 확인 완료 (${num_lines}개 샘플)"
fi
echo ""

# 3. Pseudo-label 생성
echo "Step 3: Pseudo-label 생성 (Gemini API)..."
echo ""
echo "옵션:"
echo "  --max-samples N  : N개만 처리 (테스트용)"
echo "  --model MODEL    : 모델 선택 (gemini-1.5-flash, gemini-1.5-pro)"
echo ""
echo "추가 옵션이 필요하신가요? (없으면 Enter)"
read -r extra_args

python scripts/run_full_pipeline.py \
    --input data/processed/ko_raw.jsonl \
    --output data/pseudo/ko_pseudo.jsonl \
    --config configs/teacher.yaml \
    $extra_args

echo ""
echo "=========================================="
echo "✅ 완료!"
echo "=========================================="
echo ""
echo "생성된 파일:"
echo "  - data/pseudo/ko_pseudo_raw.jsonl  (필터링 전)"
echo "  - data/pseudo/ko_pseudo.jsonl      (필터링 후)"
echo "  - data/pseudo/summary.json         (통계)"
echo ""
echo "다음 단계:"
echo "  1. 데이터 확인: cat data/pseudo/ko_pseudo.jsonl | jq"
echo "  2. 통계 확인:   cat data/pseudo/summary.json | jq"
echo "  3. 모델 학습 (예정)"
