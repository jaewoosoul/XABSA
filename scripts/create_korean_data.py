#!/usr/bin/env python3
"""
한국어 리뷰 데이터 생성 스크립트

사용자가 직접 리뷰를 입력하거나 CSV 파일에서 로드하여 JSONL 포맷으로 변환합니다.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import save_jsonl, ensure_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_from_csv(csv_path: str, text_column: str, id_column: str = None) -> list:
    """
    CSV 파일에서 한국어 리뷰 로드

    Args:
        csv_path: CSV 파일 경로
        text_column: 리뷰 텍스트가 있는 컬럼 이름
        id_column: ID 컬럼 이름 (없으면 자동 생성)

    Returns:
        리뷰 데이터 리스트
    """
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available: {df.columns.tolist()}")

    reviews = []
    for idx, row in df.iterrows():
        review_id = row[id_column] if id_column and id_column in df.columns else f"ko_{idx:06d}"
        text = str(row[text_column]).strip()

        if not text or text == 'nan':
            continue

        reviews.append({
            "id": review_id,
            "lang": "ko",
            "text": text,
            "split": "unlabeled"
        })

    logger.info(f"Loaded {len(reviews)} reviews from CSV")
    return reviews


def create_from_text_file(txt_path: str) -> list:
    """
    텍스트 파일에서 한국어 리뷰 로드 (한 줄에 하나씩)

    Args:
        txt_path: 텍스트 파일 경로

    Returns:
        리뷰 데이터 리스트
    """
    logger.info(f"Loading text file from {txt_path}")
    reviews = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            text = line.strip()
            if not text or text.startswith('#'):  # 빈 줄이나 주석 무시
                continue

            reviews.append({
                "id": f"ko_{idx:06d}",
                "lang": "ko",
                "text": text,
                "split": "unlabeled"
            })

    logger.info(f"Loaded {len(reviews)} reviews from text file")
    return reviews


def create_interactive() -> list:
    """
    대화형으로 리뷰 입력받기

    Returns:
        리뷰 데이터 리스트
    """
    print("\n" + "="*60)
    print("한국어 리뷰 데이터 생성 - 대화형 모드")
    print("="*60)
    print("\n사용법:")
    print("- 한 줄에 하나의 리뷰를 입력하세요")
    print("- 입력 완료 후 빈 줄을 입력하면 종료됩니다")
    print("- Ctrl+C로 중단할 수 있습니다\n")

    reviews = []
    idx = 0

    try:
        while True:
            text = input(f"리뷰 {idx + 1}: ").strip()

            if not text:  # 빈 줄이면 종료
                if reviews:
                    break
                else:
                    print("최소 1개 이상의 리뷰를 입력해주세요.")
                    continue

            reviews.append({
                "id": f"ko_{idx:06d}",
                "lang": "ko",
                "text": text,
                "split": "unlabeled"
            })
            idx += 1

    except KeyboardInterrupt:
        print("\n\n입력이 중단되었습니다.")
        if not reviews:
            logger.warning("입력된 리뷰가 없습니다.")
            sys.exit(1)

    logger.info(f"총 {len(reviews)} 개의 리뷰가 입력되었습니다.")
    return reviews


def main():
    parser = argparse.ArgumentParser(
        description="한국어 리뷰 데이터 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 대화형 입력
  python scripts/create_korean_data.py --output data/processed/ko_raw.jsonl

  # CSV 파일에서 로드
  python scripts/create_korean_data.py \\
    --csv reviews.csv \\
    --text-column review_text \\
    --output data/processed/ko_raw.jsonl

  # 텍스트 파일에서 로드
  python scripts/create_korean_data.py \\
    --txt reviews.txt \\
    --output data/processed/ko_raw.jsonl
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 JSONL 파일 경로"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="입력 CSV 파일 경로"
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="CSV의 텍스트 컬럼 이름 (기본값: text)"
    )

    parser.add_argument(
        "--id-column",
        type=str,
        help="CSV의 ID 컬럼 이름 (선택사항)"
    )

    parser.add_argument(
        "--txt",
        type=str,
        help="입력 텍스트 파일 경로 (한 줄에 하나씩)"
    )

    args = parser.parse_args()

    # 데이터 생성
    if args.csv:
        reviews = create_from_csv(args.csv, args.text_column, args.id_column)
    elif args.txt:
        reviews = create_from_text_file(args.txt)
    else:
        reviews = create_interactive()

    if not reviews:
        logger.error("리뷰 데이터가 없습니다.")
        sys.exit(1)

    # 출력 디렉토리 생성
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    # JSONL로 저장
    save_jsonl(reviews, str(output_path))
    logger.info(f"✓ {len(reviews)}개의 리뷰를 {args.output}에 저장했습니다.")

    # 미리보기
    print("\n" + "="*60)
    print("저장된 데이터 미리보기 (첫 3개)")
    print("="*60)
    for i, review in enumerate(reviews[:3], 1):
        print(f"\n{i}. [{review['id']}]")
        print(f"   {review['text']}")

    if len(reviews) > 3:
        print(f"\n... 외 {len(reviews) - 3}개")


if __name__ == "__main__":
    main()
