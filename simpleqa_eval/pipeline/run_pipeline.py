"""SimpleQA 评测链路主流水线

分 3 阶段可独立运行：
  1. generate  - 对所有样本做多次采样
  2. evaluate  - 判错 + 实体抽取 + Jaccard 计算
  3. analyze   - 分桶统计 + 画图

用法：
  python -m simpleqa_eval.pipeline.run_pipeline --stage generate --first_n 200
  python -m simpleqa_eval.pipeline.run_pipeline --stage evaluate
  python -m simpleqa_eval.pipeline.run_pipeline --stage analyze
  python -m simpleqa_eval.pipeline.run_pipeline --stage all --first_n 200
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

# ── 日志配置 ──
def setup_logging(level: str = "INFO"):
    """配置日志格式"""
    fmt = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  阶段 1: GENERATE — 对 SimpleQA 样本做多次采样
# ═══════════════════════════════════════════════════════════════════════

def stage_generate(cfg):
    """阶段 1: 模型生成"""
    from simpleqa_eval.data.simpleqa_loader import load_simpleqa
    from simpleqa_eval.models.generator import LocalModelGenerator

    raw_path = cfg.outputs_dir / "raw_generations.jsonl"

    print()
    logger.info("=" * 70)
    logger.info("  🚀 阶段 1: GENERATE — 模型多次采样")
    logger.info("=" * 70)
    logger.info(f"  模型:       {cfg.model_name}")
    logger.info(f"  采样次数:    {cfg.n_samples}")
    logger.info(f"  温度:       {cfg.temperature}")
    logger.info(f"  最大 token: {cfg.max_new_tokens}")
    logger.info(f"  数据量:     前 {cfg.first_n} 条")
    logger.info(f"  输出文件:   {raw_path}")
    logger.info("=" * 70)
    print()

    # 加载数据
    data = load_simpleqa(first_n=cfg.first_n)
    logger.info(f"📋 加载了 {len(data)} 条 SimpleQA 样本")
    print()

    remaining = data
    logger.info(f"📝 待处理: {len(remaining)} 条")

    # 加载模型
    generator = LocalModelGenerator(
        model_name=cfg.model_name,
        device=cfg.device,
        max_new_tokens=cfg.max_new_tokens,
    )
    generator.load()
    print()

    # 逐条生成
    total = len(remaining)
    start_time = time.time()

    with open(raw_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(remaining):
            qid = sample["qid"]
            question = sample["question"]
            gt = sample["ground_truth"]

            logger.info(f"─── [{idx+1}/{total}] qid={qid} ───")
            logger.info(f"  ❓ 问题: {question[:80]}{'...' if len(question) > 80 else ''}")
            logger.info(f"  ✅ GT:   {gt[:60]}{'...' if len(gt) > 60 else ''}")

            # 多次采样
            logger.info(f"  🎲 开始 {cfg.n_samples} 次采样 (temperature={cfg.temperature})...")
            sample_start = time.time()
            results = generator.sample_answers(
                question, n=cfg.n_samples, temperature=cfg.temperature
            )
            sample_elapsed = time.time() - sample_start

            # 提取结果
            generations = [r[0] for r in results]
            confidences = [r[1] for r in results]

            # 打印每次采样结果
            for j, (ans, conf) in enumerate(results):
                logger.info(f"    [{j+1:2d}] conf={conf:.4f} | {ans[:60]}{'...' if len(ans) > 60 else ''}")

            logger.info(f"  ⏱️  耗时: {sample_elapsed:.1f}s")

            # 保存
            record = {
                "qid": qid,
                "question": question,
                "ground_truth": gt,
                "generations": generations,
                "confidences": confidences,
                "model_name": cfg.model_name,
                "temperature": cfg.temperature,
                "n_samples": cfg.n_samples,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            # 进度估算
            elapsed = time.time() - start_time
            avg_per_sample = elapsed / (idx + 1)
            remaining_est = avg_per_sample * (total - idx - 1)
            logger.info(f"  📊 进度: {idx+1}/{total} | 平均 {avg_per_sample:.1f}s/条 | 预计剩余 {remaining_est/60:.1f}min")
            print()

    total_elapsed = time.time() - start_time
    logger.info(f"✅ 阶段 1 完成！共处理 {total} 条，耗时 {total_elapsed/60:.1f} 分钟")
    logger.info(f"💾 结果已保存: {raw_path}")


# ═══════════════════════════════════════════════════════════════════════
#  阶段 2: EVALUATE — 判错 + 实体抽取 + Jaccard 计算
# ═══════════════════════════════════════════════════════════════════════

def stage_evaluate(cfg):
    """阶段 2: 评估与 Jaccard 计算"""
    from simpleqa_eval.eval.answer_match import llm_judge_correct
    from simpleqa_eval.eval.consistency import compute_consensus_llm
    from simpleqa_eval.entity.extractor import extract_entities
    from simpleqa_eval.entity.wiki_proxy import WikiCache, get_article_set_for_entities
    from simpleqa_eval.entity.jaccard import jaccard_similarity

    raw_path = cfg.outputs_dir / "raw_generations.jsonl"
    result_csv = cfg.outputs_dir / "sample_results.csv"

    print()
    logger.info("=" * 70)
    logger.info("  🔍 阶段 2: EVALUATE — 判错 + 实体 + Jaccard")
    logger.info("=" * 70)
    logger.info(f"  输入:        {raw_path}")
    logger.info(f"  输出:        {result_csv}")
    logger.info(f"  DeepSeek:    {cfg.deepseek_model}")
    logger.info(f"  Wiki 缓存:   {cfg.wiki_cache_db}")
    logger.info("=" * 70)
    print()

    # 检查 API key
    if not cfg.deepseek_api_key:
        logger.error("❌ 未设置 DEEPSEEK_API_KEY 环境变量！")
        logger.error("   请设置: export DEEPSEEK_API_KEY='your-key-here'")
        sys.exit(1)

    # 读取生成结果
    if not raw_path.exists():
        logger.error(f"❌ 未找到生成结果: {raw_path}")
        logger.error("   请先运行 --stage generate")
        sys.exit(1)

    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # 与 generate 阶段保持一致：支持 first_n 快速小样本验证
    if cfg.first_n > 0:
        original_n = len(records)
        records = records[:cfg.first_n]
        logger.info(f"✂️  first_n={cfg.first_n}：评估前 {len(records)} 条（原始 {original_n} 条）")

    logger.info(f"📋 加载了 {len(records)} 条生成记录")
    print()

    remaining = records
    existing_rows = []
    logger.info(f"📝 待处理: {len(remaining)} 条")
    if not remaining:
        logger.warning("⚠️ 无待评估样本，跳过此阶段")
        return
    print()

    # 初始化 Wiki 缓存
    wiki_cache = WikiCache(cfg.wiki_cache_db)

    # 统计计数器
    stats = {
        "correct": 0,
        "incorrect": 0,
        "entity_fail": 0,
        "wiki_fail_samples": 0,
        "wiki_fail_entities": 0,
        "wiki_total_entities": 0,
    }
    total = len(remaining)
    start_time = time.time()

    for idx, record in enumerate(remaining):
        qid = record["qid"]
        question = record["question"]
        gt = record["ground_truth"]
        generations = record["generations"]
        confidences = record.get("confidences", [0.0] * len(generations))

        logger.info(f"═══ [{idx+1}/{total}] qid={qid} ═══")
        logger.info(f"  ❓ 问题: {question[:80]}{'...' if len(question) > 80 else ''}")
        logger.info(f"  ✅ GT:   {gt[:60]}{'...' if len(gt) > 60 else ''}")
        logger.info(f"  📋 有 {len(generations)} 个生成结果")

        # ── 2.1 计算共识答案和自一致性 ──
        logger.info("  🔗 计算共识答案（LLM 聚类）...")
        consensus_answer, consensus_count, self_consistency = compute_consensus_llm(
            generations,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            model=cfg.deepseek_model,
            retry_times=cfg.api_retry_times,
            retry_delay=cfg.api_retry_delay,
        )
        logger.info(f"  📊 共识答案: '{consensus_answer[:50]}{'...' if len(consensus_answer) > 50 else ''}'")
        logger.info(f"     共识计数: {consensus_count}/{len(generations)} | self-consistency={self_consistency:.2f}")

        time.sleep(cfg.api_call_delay)

        # ── 2.2 判断 consensus 是否正确 ──
        logger.info("  ⚖️  LLM 判错...")
        is_correct = llm_judge_correct(
            prediction=consensus_answer,
            ground_truth=gt,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            model=cfg.deepseek_model,
            retry_times=cfg.api_retry_times,
            retry_delay=cfg.api_retry_delay,
        )
        is_hallucination = 0 if is_correct else 1
        label = "✅ 正确" if is_correct else "❌ 错误(幻觉)"
        logger.info(f"     判定: {label}")
        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1

        time.sleep(cfg.api_call_delay)

        # ── 2.3 实体抽取 ──
        logger.info("  🏷️  抽取问题实体...")
        question_entities = extract_entities(
            question,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            model=cfg.deepseek_model,
            retry_times=cfg.api_retry_times,
            retry_delay=cfg.api_retry_delay,
        )
        logger.info(f"     问题实体: {question_entities}")

        time.sleep(cfg.api_call_delay)

        logger.info("  🏷️  抽取答案实体...")
        consensus_entities = extract_entities(
            consensus_answer,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            model=cfg.deepseek_model,
            retry_times=cfg.api_retry_times,
            retry_delay=cfg.api_retry_delay,
        )
        logger.info(f"     答案实体: {consensus_entities}")

        time.sleep(cfg.api_call_delay)

        # ── 2.4 Wikipedia 文章集合 + Jaccard ──
        jaccard_proxy = float("nan")
        q_article_size = 0
        a_article_size = 0
        wiki_failed = 0
        wiki_failed_entities = 0
        wiki_total_entities = 0

        if question_entities and consensus_entities:
            logger.info("  🌐 查询 Wikipedia 文章集合...")
            q_articles, q_failed, q_total = get_article_set_for_entities(
                question_entities, cache=wiki_cache,
                search_limit=cfg.wiki_search_limit,
            )
            a_articles, a_failed, a_total = get_article_set_for_entities(
                consensus_entities, cache=wiki_cache,
                search_limit=cfg.wiki_search_limit,
            )
            wiki_failed_entities = q_failed + a_failed
            wiki_total_entities = q_total + a_total
            stats["wiki_fail_entities"] += wiki_failed_entities
            stats["wiki_total_entities"] += wiki_total_entities
            q_article_size = len(q_articles)
            a_article_size = len(a_articles)
            if wiki_failed_entities > 0:
                wiki_failed = 1
                stats["wiki_fail_samples"] += 1
                logger.warning(
                    "     ⚠️  Wikipedia 请求失败，Jaccard 设为 NaN "
                    f"(failed_entities={wiki_failed_entities}/{wiki_total_entities})"
                )
            else:
                jaccard_proxy = jaccard_similarity(q_articles, a_articles)

            logger.info(f"     问题文章集: {q_article_size} 篇")
            logger.info(f"     答案文章集: {a_article_size} 篇")
            if np.isnan(jaccard_proxy):
                logger.info("     Jaccard = NaN")
            else:
                logger.info(f"     Jaccard = {jaccard_proxy:.4f}")
        else:
            stats["entity_fail"] += 1
            logger.warning(f"     ⚠️  实体为空，Jaccard 设为 NaN（问题实体: {len(question_entities)}, 答案实体: {len(consensus_entities)}）")

        # ── 2.5 汇总平均 confidence ──
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        # ── 保存一条结果 ──
        row = {
            "qid": qid,
            "question": question,
            "ground_truth": gt,
            "consensus_answer": consensus_answer,
            "consensus_count": consensus_count,
            "self_consistency": self_consistency,
            "confidence": mean_confidence,
            "is_correct": int(is_correct),
            "is_hallucination": is_hallucination,
            "question_entities": json.dumps(question_entities, ensure_ascii=False),
            "consensus_entities": json.dumps(consensus_entities, ensure_ascii=False),
            "question_article_set_size": q_article_size,
            "answer_article_set_size": a_article_size,
            "jaccard_proxy": jaccard_proxy,
            "wiki_failed": wiki_failed,
            "wiki_failed_entities": wiki_failed_entities,
            "wiki_total_entities": wiki_total_entities,
            "model_name": record.get("model_name", ""),
            "temperature": record.get("temperature", 0),
            "n_samples": record.get("n_samples", 0),
        }
        existing_rows.append(row)

        # 每条都写一次 CSV（增量保存）
        _write_csv(result_csv, existing_rows)

        # 进度与统计
        elapsed = time.time() - start_time
        avg_per_sample = elapsed / (idx + 1)
        remaining_est = avg_per_sample * (total - idx - 1)
        total_processed = stats["correct"] + stats["incorrect"]
        halluc_rate = stats["incorrect"] / total_processed if total_processed > 0 else 0
        wiki_fail_rate = stats["wiki_fail_samples"] / total_processed if total_processed > 0 else 0

        logger.info(f"  📊 累计: 正确={stats['correct']} 错误={stats['incorrect']} "
                     f"幻觉率={halluc_rate:.2%} 实体失败={stats['entity_fail']} "
                     f"Wiki失败率={wiki_fail_rate:.2%}")
        logger.info(f"  ⏱️  平均 {avg_per_sample:.1f}s/条 | 预计剩余 {remaining_est/60:.1f}min")
        print()

    wiki_cache.close()

    total_elapsed = time.time() - start_time
    total_processed = stats["correct"] + stats["incorrect"]
    halluc_rate = stats["incorrect"] / total_processed if total_processed > 0 else 0
    wiki_fail_rate = stats["wiki_fail_samples"] / total_processed if total_processed > 0 else 0
    wiki_entity_fail_rate = (
        stats["wiki_fail_entities"] / stats["wiki_total_entities"]
        if stats["wiki_total_entities"] > 0 else 0
    )

    print()
    logger.info("=" * 70)
    logger.info("  📊 阶段 2 最终统计")
    logger.info("=" * 70)
    logger.info(f"  总样本:     {total_processed}")
    logger.info(f"  正确:       {stats['correct']}")
    logger.info(f"  错误(幻觉): {stats['incorrect']}")
    logger.info(f"  幻觉率:     {halluc_rate:.2%}")
    logger.info(f"  实体抽取失败: {stats['entity_fail']}")
    logger.info(f"  Wiki 样本失败率: {wiki_fail_rate:.2%} ({stats['wiki_fail_samples']}/{total_processed})")
    logger.info(
        "  Wiki 实体失败率: "
        f"{wiki_entity_fail_rate:.2%} ({stats['wiki_fail_entities']}/{stats['wiki_total_entities']})"
    )
    logger.info(f"  耗时:       {total_elapsed/60:.1f} 分钟")
    logger.info(f"  💾 结果:    {result_csv}")
    logger.info("=" * 70)


def _write_csv(path: Path, rows: List[Dict]):
    """将结果列表写入 CSV"""
    if not rows:
        return
    fieldnames = rows[0].keys()
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════
#  阶段 3: ANALYZE — 分桶统计 + 画图
# ═══════════════════════════════════════════════════════════════════════

def stage_analyze(cfg):
    """阶段 3: 分桶分析与可视化"""
    from simpleqa_eval.pipeline.aggregate import aggregate_buckets
    from simpleqa_eval.viz.plot_trends import plot_bucket_trends

    result_csv = cfg.outputs_dir / "sample_results.csv"
    bucket_csv = cfg.outputs_dir / "bucket_stats.csv"
    figures_dir = cfg.outputs_dir / "figures"

    print()
    logger.info("=" * 70)
    logger.info("  📊 阶段 3: ANALYZE — 分桶统计 + 可视化")
    logger.info("=" * 70)
    logger.info(f"  输入:   {result_csv}")
    logger.info(f"  桶统计: {bucket_csv}")
    logger.info(f"  图表:   {figures_dir}")
    logger.info("=" * 70)
    print()

    if not result_csv.exists():
        logger.error(f"❌ 未找到样本结果: {result_csv}")
        logger.error("   请先运行 --stage evaluate")
        sys.exit(1)

    # 分桶统计
    stats_df = aggregate_buckets(
        sample_csv=str(result_csv),
        output_csv=str(bucket_csv),
        n_buckets=5,
    )

    if stats_df.empty:
        logger.error("❌ 分桶统计为空，请检查数据")
        return

    # 画图
    print()
    plot_bucket_trends(
        bucket_csv=str(bucket_csv),
        output_dir=str(figures_dir),
    )

    print()
    logger.info("✅ 阶段 3 完成！")
    logger.info(f"   📄 桶统计: {bucket_csv}")
    logger.info(f"   📈 图表:   {figures_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="SimpleQA Real-world 评测链路",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "evaluate", "analyze", "all"],
        default="all",
        help="运行阶段:\n"
             "  generate  - 模型多次采样\n"
             "  evaluate  - 判错 + 实体 + Jaccard\n"
             "  analyze   - 分桶统计 + 画图\n"
             "  all       - 全部流程",
    )
    parser.add_argument("--first_n", type=int, default=200, help="取前 N 条样本 (默认 200)")
    parser.add_argument("--n_samples", type=int, default=10, help="每题采样次数 (默认 10)")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度 (默认 0.7)")
    parser.add_argument("--model", type=str, default=None, help="本地模型名称 (默认 Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--run_id", type=str, default=None, help="运行 ID；不传则自动按时间生成/选择最新")
    parser.add_argument("--output_dir", type=str, default=None, help="自定义输出目录")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别 (默认 INFO)")
    return parser.parse_args()


class PipelineConfig:
    """将 argparse + Config 统一成运行配置"""

    def __init__(self, args):
        from simpleqa_eval.config import Config, OUTPUTS_DIR

        base_cfg = Config()

        self.model_name = args.model or base_cfg.model_name
        self.device = base_cfg.resolve_device()
        self.first_n = args.first_n
        self.n_samples = args.n_samples
        self.temperature = args.temperature
        self.max_new_tokens = base_cfg.max_new_tokens

        self.deepseek_api_key = base_cfg.deepseek_api_key
        self.deepseek_base_url = base_cfg.deepseek_base_url
        self.deepseek_model = base_cfg.deepseek_model

        self.wiki_cache_db = base_cfg.wiki_cache_db
        self.wiki_search_limit = base_cfg.wiki_search_limit

        self.api_retry_times = base_cfg.api_retry_times
        self.api_retry_delay = base_cfg.api_retry_delay
        self.api_call_delay = base_cfg.api_call_delay

        if args.output_dir:
            self.output_root = Path(args.output_dir)
        else:
            self.output_root = OUTPUTS_DIR

        base_cfg.ensure_dirs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.runs_root = self.output_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.latest_run_path = self.output_root / "latest_run.txt"

        self.run_id = self._resolve_run_id(args.stage, args.run_id)
        self.outputs_dir = self.runs_root / self.run_id
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "figures").mkdir(parents=True, exist_ok=True)

        self.run_record_path = self.outputs_dir / "run_info.json"
        self.created_at = datetime.now().isoformat(timespec="seconds")
        if self.run_record_path.exists():
            try:
                with open(self.run_record_path, "r", encoding="utf-8") as f:
                    old_record = json.load(f)
                self.created_at = old_record.get("created_at", self.created_at)
            except Exception:
                pass

        if args.stage in ("generate", "all"):
            self.latest_run_path.write_text(self.run_id + "\n", encoding="utf-8")

    def _resolve_run_id(self, stage: str, requested_run_id: str | None) -> str:
        if requested_run_id:
            return requested_run_id

        if stage in ("generate", "all"):
            return datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.latest_run_path.exists():
            latest = self.latest_run_path.read_text(encoding="utf-8").strip()
            if latest and (self.runs_root / latest).exists():
                return latest

        run_dirs = sorted(p.name for p in self.runs_root.iterdir() if p.is_dir())
        if run_dirs:
            return run_dirs[-1]

        raise ValueError("未找到可用 run。请先运行 generate/all，或显式传入 --run_id。")


def _write_run_record(cfg: "PipelineConfig", args, status: str):
    record = {
        "run_id": cfg.run_id,
        "status": status,
        "stage": args.stage,
        "created_at": cfg.created_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(cfg.outputs_dir),
        "config": {
            "model_name": cfg.model_name,
            "device": cfg.device,
            "first_n": cfg.first_n,
            "n_samples": cfg.n_samples,
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens,
            "deepseek_model": cfg.deepseek_model,
        },
    }
    with open(cfg.run_record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    setup_logging(args.log_level)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SimpleQA Real-world 评测链路                                ║")
    print("║  复现: 'When Bias Pretends to Be Truth' 第4节               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    try:
        cfg = PipelineConfig(args)
    except ValueError as e:
        logger.error(f"❌ 配置错误: {e}")
        sys.exit(1)

    _write_run_record(cfg, args, status="running")

    logger.info("🔧 运行配置:")
    logger.info(f"   阶段:       {args.stage}")
    logger.info(f"   Run ID:     {cfg.run_id}")
    logger.info(f"   模型:       {cfg.model_name}")
    logger.info(f"   设备:       {cfg.device}")
    logger.info(f"   数据量:     前 {cfg.first_n} 条")
    logger.info(f"   采样次数:   {cfg.n_samples}")
    logger.info(f"   温度:       {cfg.temperature}")
    logger.info(f"   DeepSeek:   {cfg.deepseek_model}")
    logger.info(f"   输出目录:   {cfg.outputs_dir}")
    logger.info(f"   记录文件:   {cfg.run_record_path}")
    print()

    overall_start = time.time()
    try:
        if args.stage in ("generate", "all"):
            stage_generate(cfg)

        if args.stage in ("evaluate", "all"):
            stage_evaluate(cfg)

        if args.stage in ("analyze", "all"):
            stage_analyze(cfg)

        overall_elapsed = time.time() - overall_start
        _write_run_record(cfg, args, status="completed")
        print()
        logger.info(f"🏁 全部完成！总耗时: {overall_elapsed/60:.1f} 分钟")
        print()
    except Exception:
        _write_run_record(cfg, args, status="failed")
        raise


if __name__ == "__main__":
    main()
