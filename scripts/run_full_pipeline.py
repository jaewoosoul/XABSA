#!/usr/bin/env python3
"""
전체 파이프라인 실행 스크립트

1. 데이터 확인
2. LLM Teacher로 pseudo-label 생성 (Gemini)
3. 필터링 적용
4. 통계 출력
"""

import argparse
import logging
import sys
from pathlib import Path
import os

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.taxonomy import Taxonomy
from src.teacher import GeminiTeacher, PseudoLabelFilter
from src.utils import (
    setup_logging, load_jsonl, save_jsonl,
    save_json, ensure_dir
)

logger = logging.getLogger(__name__)


def check_api_key():
    """API 키 확인"""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(
            "\n❌ API 키를 찾을 수 없습니다!\n\n"
            "다음 중 하나를 설정해주세요:\n"
            "1. .env 파일에 GOOGLE_API_KEY=your-key 추가\n"
            "2. 환경 변수로 설정: export GOOGLE_API_KEY=your-key\n\n"
            "API 키 발급: https://ai.google.dev/\n"
        )
        sys.exit(1)

    logger.info(f"✓ API 키 확인 완료 (길이: {len(api_key)})")
    return api_key


def main():
    parser = argparse.ArgumentParser(
        description="전체 파이프라인 실행 (Gemini API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행 (config 파일 사용)
  python scripts/run_full_pipeline.py \\
    --input data/processed/ko_raw.jsonl \\
    --output data/pseudo/ko_pseudo.jsonl

  # 커스텀 설정
  python scripts/run_full_pipeline.py \\
    --input data/processed/ko_raw.jsonl \\
    --output data/pseudo/ko_pseudo.jsonl \\
    --model gemini-1.5-pro \\
    --max-samples 100

  # 필터링 없이 raw만 생성
  python scripts/run_full_pipeline.py \\
    --input data/processed/ko_raw.jsonl \\
    --output data/pseudo/ko_pseudo_raw.jsonl \\
    --no-filter
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 JSONL 파일 (unlabeled 한국어 리뷰)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 JSONL 파일 (pseudo-labeled 데이터)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/teacher.yaml",
        help="설정 파일 경로 (기본값: configs/teacher.yaml)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-flash-latest",
        help="Gemini 모델 (기본값: gemini-flash-latest)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="처리할 최대 샘플 수 (테스트용)"
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="필터링 없이 raw pseudo-label만 생성"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨"
    )

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.log_level)

    print("\n" + "="*70)
    print("🚀 XABSA Pseudo-Label Generation Pipeline (Gemini)")
    print("="*70 + "\n")

    # Step 1: API 키 확인
    logger.info("Step 1: API 키 확인...")
    api_key = check_api_key()

    # Step 2: 설정 로드
    logger.info(f"Step 2: 설정 로드... ({args.config})")
    config = load_config(args.config)

    # Step 3: Taxonomy 로드
    taxonomy_path = config.get("taxonomy_path", "configs/taxonomy.yaml")
    logger.info(f"Step 3: Taxonomy 로드... ({taxonomy_path})")
    taxonomy = Taxonomy(taxonomy_path)
    logger.info(f"  - {len(taxonomy.categories)} 카테고리: {', '.join(taxonomy.categories[:5])}...")

    # Step 4: 입력 데이터 로드
    logger.info(f"Step 4: 입력 데이터 로드... ({args.input})")
    if not Path(args.input).exists():
        logger.error(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        logger.info("\n먼저 데이터를 생성하세요:")
        logger.info("  python scripts/create_korean_data.py --output data/processed/ko_raw.jsonl")
        sys.exit(1)

    samples = load_jsonl(args.input)

    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"  - {len(samples)}개 샘플로 제한 (--max-samples={args.max_samples})")
    else:
        logger.info(f"  - 총 {len(samples)}개 샘플")

    # 샘플 미리보기
    if samples:
        logger.info(f"\n  첫 번째 샘플 미리보기:")
        logger.info(f"  ID: {samples[0]['id']}")
        logger.info(f"  Text: {samples[0]['text'][:100]}...")

    # Step 5: Gemini Teacher 생성
    logger.info(f"\nStep 5: Gemini Teacher 초기화... (model={args.model})")
    teacher_config = config.get("teacher", {})

    teacher = GeminiTeacher(
        model=args.model,
        api_key=api_key,
        temperature=teacher_config.get("temperature", 0.0),
        max_tokens=teacher_config.get("max_tokens", 1000),
        max_retries=teacher_config.get("max_retries", 3)
    )
    logger.info(f"  - {teacher}")

    # Step 6: Pseudo-label 생성
    logger.info(f"\nStep 6: Pseudo-label 생성 중...")
    logger.info(f"  (Gemini API 호출 중... 시간이 걸릴 수 있습니다)")

    from tqdm import tqdm

    pseudo_labeled = []
    errors = 0

    for sample in tqdm(samples, desc="  Generating"):
        try:
            triplets = teacher.generate_triplets(
                sample["text"],
                lang=sample.get("lang", "ko")
            )

            pseudo_sample = sample.copy()
            pseudo_sample["gold_triplets"] = triplets
            pseudo_labeled.append(pseudo_sample)

        except Exception as e:
            logger.debug(f"Error on {sample['id']}: {e}")
            errors += 1
            pseudo_sample = sample.copy()
            pseudo_sample["gold_triplets"] = []
            pseudo_labeled.append(pseudo_sample)

    logger.info(f"  - 생성 완료: {len(pseudo_labeled) - errors}개 성공, {errors}개 실패")

    # 통계
    total_triplets = sum(len(s["gold_triplets"]) for s in pseudo_labeled)
    avg_triplets = total_triplets / len(pseudo_labeled) if pseudo_labeled else 0
    logger.info(f"  - 총 {total_triplets}개 triplet 생성 (평균: {avg_triplets:.2f}개/샘플)")

    # Step 7: Raw 저장
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    raw_output = output_path.parent / f"{output_path.stem}_raw.jsonl"
    save_jsonl(pseudo_labeled, str(raw_output))
    logger.info(f"\nStep 7: Raw pseudo-label 저장 완료")
    logger.info(f"  - {raw_output}")

    # Step 8: 필터링 (옵션)
    if not args.no_filter:
        logger.info(f"\nStep 8: 필터링 적용 중...")

        filter_config = config.get("filtering", {})
        filter_obj = PseudoLabelFilter(
            taxonomy=taxonomy,
            check_term_existence=filter_config.get("check_term_existence", True),
            remove_duplicates=filter_config.get("remove_duplicates", True),
            normalize_whitespace=filter_config.get("normalize_whitespace", True),
            validate_category=filter_config.get("validate_category", True),
            map_to_etc=filter_config.get("map_to_etc", True),
            max_triplets_per_text=filter_config.get("max_triplets_per_text", 8)
        )

        filtered = filter_obj.filter_batch(pseudo_labeled)

        # 필터링 후 저장
        save_jsonl(filtered, str(output_path))
        logger.info(f"  - 필터링 완료: {args.output}")

        # 통계 저장
        stats = filter_obj.get_summary()
        stats_path = output_path.parent / "summary.json"
        save_json(stats, str(stats_path))
        logger.info(f"  - 통계 저장: {stats_path}")

        # 통계 출력
        print("\n" + "="*70)
        print("📊 필터링 결과")
        print("="*70)
        print(f"필터링 전: {stats['total_before']} triplets")
        print(f"필터링 후: {stats['total_after']} triplets")
        print(f"제거됨:     {stats['removed']} triplets ({100-stats['retention_rate']*100:.1f}%)")
        print(f"유지율:     {stats['retention_rate']*100:.1f}%")

        print("\n상세 breakdown:")
        for key, value in stats['breakdown'].items():
            print(f"  - {key}: {value}")
    else:
        # 필터링 없이 저장
        save_jsonl(pseudo_labeled, str(output_path))
        logger.info(f"\nStep 8: 필터링 건너뜀 (--no-filter)")
        logger.info(f"  - {args.output}")

    # 완료
    print("\n" + "="*70)
    print("✅ 파이프라인 완료!")
    print("="*70)
    print(f"\n출력 파일:")
    print(f"  - Raw:      {raw_output}")
    if not args.no_filter:
        print(f"  - Filtered: {output_path}")
        print(f"  - Stats:    {stats_path}")

    print("\n다음 단계:")
    print("  1. 생성된 데이터 확인")
    print("  2. 모델 학습 (예정)")
    print("  3. 평가 (예정)")


if __name__ == "__main__":
    main()
