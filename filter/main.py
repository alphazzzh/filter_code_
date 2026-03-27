# main.py
# ============================================================
# ASR 情报清洗与结构化抽取流水线 —— 全链路编排入口
#
# 流水线架构
# ─────────────────────────────────────────────────────────────
#             ┌─────────────┐
#   CSV/API → │ StageOneFilter│ → (过滤低价值)
#             └──────┬───────┘
#                    │ 接通 or 灰区记录
#                    ▼
#             ┌──────────────────┐
#             │ StageTwoPipeline │ → 拓扑 + 意图 + 角色
#             └──────┬───────────┘
#                    ▼
#             ┌────────────────────┐
#             │ IntelligenceScorer │ → 打分 + 标签 + 证据链
#             └──────┬─────────────┘
#                    ▼
#              JSONL 文件 + 控制台预览
#
# 工业级流转建议（详见 _route_record 的注释）
# ─────────────────────────────────────────────────────────────
# V5.0 重构：铲除 LITE 分流，采用 PASS / SKIP 二态策略：
#   PASS   → 正常接通记录，走完全部五个阶段
#   SKIP   → 极端噪声（空文本/纯噪音字符），直接丢弃，写日志
# UnconnectedDetector 已从 StageOneFilter 中铲除，
# 所有记录均标记为 CONNECTED，由后续拓扑引擎和打分器全权判定价值。
# ============================================================

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# ── 流水线模块 ────────────────────────────────────────────────
# 阶段一：启发式特征提取与高危拦截
from stage_one_filter import StageOneFilter
# 阶段一数据模型（Stub 模式下兼容真实 ASRRecord）
from models_stage2 import ASRRecord, StageTwoResult
# 阶段二：拓扑 + 意图 + 角色
from stage_two_pipeline import StageTwoPipeline
# 阶段三/四/五：规则打分
from stage_three_scorer import IntelligenceScorer, BotConfidenceEngine, AdvancedVoicemailDetector
from topology_engine import TopologyEngine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 日志配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers= [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("asr_pipeline.main")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全局配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RE_IMMUNITY = re.compile(r"国际形势|中美关系|国家政策|中共|法轮功|疫情|买卖毒品|冰毒|走私|涉密", re.IGNORECASE)
RE_IVR_BOT = re.compile(r"智能语音导航|请按[一二三四1234]|滴声后给.+留言|收取.+通话费用|输入结束请按井号键|您好.+客服|满意请按", re.IGNORECASE)

@dataclass(frozen=True)
class PipelineConfig:
    """
    全管道配置，集中管理所有可调参数。
    生产环境建议从 YAML/环境变量加载，此处给出合理默认值。
    """
    # 输入输出
    input_csv:       str = "data/asr_records.csv"
    output_jsonl:    str = "output/intelligence_results.jsonl"
    # 阶段一
    fasttext_model:  str = "models/lid.176.bin"
    # 阶段二
    bge_model_name:  str = "BAAI/bge-m3"
    use_fp16:        bool = True
    intent_threshold:float = 0.75
    # 漏斗路由：纯噪音记录直接丢弃
    # 控制台预览：每处理多少条打印一次摘要
    preview_every:   int  = 50


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 路由决策
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RouteDecision:
    PASS = "PASS"    # 正常放行，走全链路
    SKIP = "SKIP"    # 极端噪声，丢弃


def _route_record(records: list[ASRRecord]) -> str:
    """
    根据阶段一产出决定记录的流转路径。

    V5.0 重构：铲除 V1.0 的 p_unconnected / LITE 分流逻辑。
    改为基于全量 effective_text 拼接后的去重长度判断：
      SKIP：拼接去重后总长度 < 5，纯噪音（如"嗯"、"喂"等单字）
      PASS：其余一律放行，保留语气词供拓扑引擎分析
    """
    # 拼接所有记录的有效文本并去重
    all_text = " ".join(r.effective_text for r in records if r.effective_text)
    unique_chars = set(all_text.replace(" ", ""))
    
    if len(unique_chars) < 5:
        return RouteDecision.SKIP
    
    return RouteDecision.PASS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSV 读取 & ASRRecord 组装
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
'''
def _iter_csv_as_conversations(
    csv_path: str,
) -> Iterator[tuple[str, list[ASRRecord]]]:
    """
    从 CSV 文件中按 conversation_id 分组，逐组 yield (conv_id, records)。

    期望 CSV 列名（最小集）
    ──────────────────────────────────────────────────────────
    conversation_id : 会话标识（同一通话的多行共享此 ID）
    record_id       : 单条 ASR 片段唯一 ID
    speaker_id      : 发言方 ID（如 "A"/"B" 或 "wxid_xxx"）
    raw_text        : ASR 原始转写文本

    可选列（有则读取，无则留空）
    ──────────────────────────────────────────────────────────
    cleaned_text    : 阶段一已处理的容错文本
    source_lang_hint: 上游语种提示
    """
    current_conv_id: str | None = None
    current_records: list[ASRRecord] = []

    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"CSV 文件不存在：{csv_path}，将使用内置 Mock 数据。")
        yield from _iter_mock_conversations()
        return

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id:   str = row.get("conversation_id", "unknown")
            record_id: str = row.get("record_id", f"r_{id(row)}")
            speaker_id:str = row.get("speaker_id", "A")
            raw_text:  str = row.get("raw_text", "").strip()

            if not raw_text:
                continue  # 跳过空行

            record = ASRRecord(
                record_id        = record_id,
                speaker_id       = speaker_id,
                raw_text         = raw_text,
                cleaned_text     = row.get("cleaned_text") or None,
                source_lang_hint = row.get("source_lang_hint") or None,
            )

            if conv_id != current_conv_id:
                if current_records and current_conv_id:
                    yield current_conv_id, current_records
                current_conv_id  = conv_id
                current_records  = [record]
            else:
                current_records.append(record)

        # 最后一组
        if current_records and current_conv_id:
            yield current_conv_id, current_records

'''
def _iter_csv_as_conversations(csv_file_path: str):
    """
    读取 CSV 文件并使用生成器逐条产出会话记录。
    要求：CSV 第一列无用，第二列是包含整通对话的超长文本。
    """
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头

        for idx, row in enumerate(reader):
            # 确保至少有两列
            if len(row) < 2:
                continue
            
            # 第一列 idx 没用，我们直接用循环索引生成全局唯一会话 ID
            conv_id = f"conv_{idx+1:05d}"
            
            # 提取第二列的完整对话文本
            raw_transcript = row[1] 
            
            if not raw_transcript.strip():
                continue

            # 调用上面的正则拆解函数，将这段长文本化为结构化 List[ASRRecord]
            records = parse_transcript_cell(raw_transcript, conv_id)
            
            if records:
                yield conv_id, records


def _iter_mock_conversations() -> Iterator[tuple[str, list[ASRRecord]]]:
    """
    内置 Mock 数据，在无 CSV 文件时用于演示和测试。
    覆盖三种典型场景：
      conv_001 → 诈骗高压态势（非对称 + financial + urgency）
      conv_002 → 可疑对称协商（疑似杀猪盘前期）
      conv_003 → 纯未接通
    """
    mock_data: list[tuple[str, list[tuple[str, str]]]] = [
        # ── 场景 1：非对称诈骗态势 ────────────────────────────
        ("conv_001", [
            ("A", "您好，我这里是公安局经侦支队，您的账户涉嫌洗钱案件。"),
            ("B", "啊？什么？"),
            ("A", "你听清楚，你的账户现在马上就要被冻结了，必须立刻配合我们操作。"),
            ("B", "我没有做过什么违法的事情啊。"),
            ("A", "你知道逾期不处理会影响征信吗？你必须今天之内把资金转移到安全账户。"),
            ("B", "好，好的，那要怎么操作？"),
            ("A", "把你的银行卡号和验证码报给我，我们来帮你处理。"),
            ("B", "嗯，好的，我的卡号是……"),
        ]),
        # ── 场景 2：对称疑似杀猪盘前期 ──────────────────────
        ("conv_002", [
            ("X", "早安，今天天气怎么样？"),
            ("Y", "还不错，你吃早饭了吗"),
            ("X", "吃了，想到你了特地发消息，你最近工作怎么样，辛苦吗"),
            ("Y", "哈哈哈哈，还行吧，你真好"),
            ("X", "你上次说想投资，我有内部返利通道，朋友才能参加的"),
            ("Y", "哦，是什么"),
            ("X", "保本保息，年化百分之三十，你有兴趣可以了解一下"),
            ("Y", "好啊，可以，你说怎么操作"),
        ]),
        # ── 场景 3：未接通/语音信箱 ──────────────────────────
        ("conv_003", [
            ("A", "喂"),
            ("B", "嗯"),
            ("A", "喂"),
        ]),
    ]

    def _make(conv_id: str, sid: str, txt: str, idx: int) -> ASRRecord:
        return ASRRecord(
            record_id  = f"{conv_id}-{sid}-{idx:03d}",
            speaker_id = sid,
            raw_text   = txt,
            cleaned_text = txt,
        )

    for conv_id, turns in mock_data:
        records = [
            _make(conv_id, sid, txt, i)
            for i, (sid, txt) in enumerate(turns)
        ]
        yield conv_id, records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SKIP 路径：占位结果（极端噪声）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_skip_result(conv_id: str) -> dict[str, Any]:
    """为 SKIP 路径（极端噪声）构建占位结果，确保漏斗数据完整。"""
    return {
        "conversation_id":     conv_id,
        "final_score":         0,
        "tags":                ["extreme_noise_skipped"],
        "track_type":          "n/a",
        "roles":               {},
        "interaction_summary": {},
        "score_breakdown":     [{"delta": -50, "reason": "SKIP 路径：极端噪声，直接丢弃"}],
        "_route":              "SKIP",
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(config: PipelineConfig) -> None:
    """
    流水线主函数，负责：
    1. 初始化三个阶段的处理器（进程内单例模式）
    2. 逐会话读取 CSV → 路由决策 → 阶段一 → 阶段二 → 打分 → 写出
    3. 统计漏斗各层数据量，最终打印运营摘要
    """
    logger.info("=" * 60)
    logger.info("ASR 情报流水线启动")
    logger.info(f"  输入：{config.input_csv}")
    logger.info(f"  输出：{config.output_jsonl}")
    logger.info("=" * 60)

    # ── 初始化三个阶段的处理器 ────────────────────────────────
    logger.info("[初始化] 加载 StageOneFilter（fastText LID）…")
    stage1 = StageOneFilter(
        fasttext_model_path = config.fasttext_model,
    )

    logger.info("[初始化] 加载 StageTwoPipeline（BGE-M3）…")
    stage2 = StageTwoPipeline(
        bge_model_name   = config.bge_model_name,
        use_fp16         = config.use_fp16,
        intent_threshold = config.intent_threshold,
    )

    logger.info("[初始化] 加载 IntelligenceScorer（规则引擎）…")
    scorer = IntelligenceScorer()

    logger.info("[初始化] 加载 BotConfidenceEngine + AdvancedVoicemailDetector + TopologyEngine…")
    bot_engine = BotConfidenceEngine()
    voicemail_engine = AdvancedVoicemailDetector()
    topo_engine = TopologyEngine()

    # ── 确保输出目录存在 ──────────────────────────────────────
    output_path = Path(config.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 漏斗统计 ──────────────────────────────────────────────
    stats: dict[str, int] = {
        "total_conversations": 0,
        "total_records":       0,
        "routed_pass":         0,
        "routed_skip":         0,
        "stage2_processed":    0,
        "high_risk":           0,   # final_score >= 70
    }

    logger.info("[运行] 开始处理对话记录……")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for conv_id, raw_records in _iter_csv_as_conversations(config.input_csv):
            stats["total_conversations"] += 1
            stats["total_records"]       += len(raw_records)

            full_text = " ".join([r.raw_text for r in raw_records])
            has_immunity = bool(RE_IMMUNITY.search(full_text))
            
            if has_immunity:
                logger.warning(f"[🔥 警报] conv={conv_id} 发现极高危免疫特征！强制保送！")
            elif RE_IVR_BOT.search(full_text):
                logger.info(f"[🚫 拦截] conv={conv_id} 命中 IVR/机器客服废料，阶段零直接丢弃！")
                stats["routed_skip"] += 1
                continue

            # ── 阶段一：逐条处理 ────────────────────────────
            s1_records: list[ASRRecord] = []
            for rec in raw_records:
                try:
                    s1_rec = stage1.process(rec)
                    s1_records.append(s1_rec)
                except Exception as exc:
                    logger.warning(
                        f"[Stage1] conv={conv_id} rec={rec.record_id} "
                        f"处理失败，跳过：{exc}"
                    )

            if not s1_records:
                stats["routed_skip"] += 1
                continue

            # ── 路由决策：基于阶段一产出 ─────────────────────
            route: str = _route_record(s1_records)

            if route == RouteDecision.SKIP:
                stats["routed_skip"] += 1
                logger.debug(f"[SKIP] conv={conv_id}，极端噪声，已丢弃。")
                skip_result = _build_skip_result(conv_id)
                skip_result["_processed_at"] = datetime.utcnow().isoformat()
                _write_json_line(out_f, skip_result)
                continue

            # 阶段一 metadata 透传：将汇总的阶段一信号注入给阶段二 extra_metadata
            s1_aggregate_meta: dict[str, Any] = _aggregate_stage1_meta(s1_records)
            
            # 🚨 缺陷 1 修复：阶段一硬正则极高危透传
            if has_immunity:
                s1_aggregate_meta["stage_one_critical_hit"] = True
                s1_aggregate_meta["hit_keyword"] = "高危免疫特征"

            # ── PASS 路径：阶段二 + 打分 ─────────────────────
            stats["routed_pass"] += 1
            try:
                stage2_result: StageTwoResult = stage2.process_conversation(
                    conversation_id = conv_id,
                    records         = s1_records,
                    extra_metadata  = {"stage_one": s1_aggregate_meta},
                )
                stats["stage2_processed"] += 1
            except Exception as exc:
                logger.error(
                    f"[Stage2] conv={conv_id} 处理失败，降级输出：{exc}",
                    exc_info=True,
                )
                fallback_result = {
                    "conversation_id":     conv_id,
                    "final_score":         50,
                    "tags":                ["stage2_error_fallback"],
                    "track_type":          "n/a",
                    "roles":               {},
                    "interaction_summary": {},
                    "nlp_features_summary": {},
                    "score_breakdown":     [{"delta": 0, "tag": "stage2_error", "reason": f"阶段二处理异常，降级兜底：{exc}"}],
                    "_route":              "PASS",
                    "_error":              str(exc),
                    "_processed_at":       datetime.utcnow().isoformat(),
                }
                _write_json_line(out_f, fallback_result)
                continue

            # ── 阶段三/四/五：打分 ────────────────────────────
            intel: dict[str, Any] = scorer.evaluate(stage2_result)

            # ── 阶段 3.5：多维置信度辅助判定 ──────────────────
            topo_metrics = topo_engine.compute_metrics(stage2_result.dialogue_turns)
            bot_result = bot_engine.evaluate(stage2_result, filler_word_rate=topo_metrics.filler_word_rate)
            voicemail_result = voicemail_engine.evaluate(stage2_result, is_decoupled=topo_metrics.is_decoupled)

            intel["bot_confidence"] = {
                "bot_score": bot_result["bot_score"],
                "bot_label": bot_result["bot_label"].value,
                "veto_reason": bot_result["veto_reason"],
                "details": bot_result["details"],
            }
            intel["voicemail_detection"] = {
                "voicemail_score": voicemail_result["voicemail_score"],
                "is_voicemail": voicemail_result["is_voicemail"],
                "veto_reason": voicemail_result["veto_reason"],
                "details": voicemail_result["details"],
            }
            intel["topology_metrics"] = {
                "filler_word_rate": topo_metrics.filler_word_rate,
                "max_sentence_length": topo_metrics.max_sentence_length,
                "avg_sentence_length": topo_metrics.avg_sentence_length,
                "is_decoupled": topo_metrics.is_decoupled,
            }

            intel["_route"]        = "PASS"
            intel["_processed_at"] = datetime.utcnow().isoformat()

            _write_json_line(out_f, intel)

            if intel["final_score"] >= 70:
                stats["high_risk"] += 1

            if stats["total_conversations"] % config.preview_every == 0:
                _print_preview(intel)

    # ── 漏斗摘要 ──────────────────────────────────────────────
    _print_funnel_summary(stats)
    logger.info(f"[完成] 结果已写入：{output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _aggregate_stage1_meta(records: list[ASRRecord]) -> dict[str, Any]:
    """
    将一组 ASRRecord 的阶段一元信息聚合为会话级别的单一字典。

    V5.0 重构：移除 p_unconnected / entity_matched（UnconnectedDetector 已铲除），
    仅保留 bot_label 和 dominant_lang。
    """
    any_bot:      bool  = False
    lang_counter: dict[str, int] = {}

    for rec in records:
        meta = getattr(rec, "metadata", {})
        s1   = meta.get("stage_one", {}) if isinstance(meta, dict) else {}

        if str(s1.get("bot_label", "")).lower() == "bot":
            any_bot = True

        lang = getattr(rec, "lang", None) or "zh"
        lang_counter[lang] = lang_counter.get(lang, 0) + 1

    # 主语种：出现次数最多的语种
    dominant_lang: str = max(lang_counter, key=lambda k: lang_counter[k]) \
        if lang_counter else "zh"

    return {
        "bot_label":      "bot" if any_bot else "human",
        "dominant_lang":  dominant_lang,
    }


def _write_json_line(file: Any, data: dict[str, Any]) -> None:
    """将字典序列化为 JSON 行写入文件（JSONL 格式）。"""
    file.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def _print_preview(intel: dict[str, Any]) -> None:
    """控制台预览一条情报结果的核心字段。"""
    logger.info(
        f"[预览] conv={intel.get('conversation_id')} | "
        f"score={intel.get('final_score')} | "
        f"route={intel.get('_route', '?')} | "
        f"tags={intel.get('tags')} | "
        f"roles={intel.get('roles')}"
    )


def _print_funnel_summary(stats: dict[str, int]) -> None:
    """打印漏斗各层统计摘要。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("【漏斗摘要】")
    logger.info(f"  总对话数    : {stats['total_conversations']}")
    logger.info(f"  总 ASR 碎片 : {stats['total_records']}")
    logger.info(f"  路由 PASS   : {stats['routed_pass']}")
    logger.info(f"  路由 SKIP   : {stats['routed_skip']}")
    logger.info(f"  阶段二完成  : {stats['stage2_processed']}")
    logger.info(f"  高风险(≥70) : {stats['high_risk']}")
    pass_total = max(stats["routed_pass"], 1)
    logger.info(f"  高风险占比  : {stats['high_risk'] / pass_total:.1%}")
    logger.info("=" * 60)

import re
from models_stage2 import ASRRecord

def parse_transcript_cell(raw_text: str, conv_id: str) -> list[ASRRecord]:
    """
    带记忆状态的单元格解析器，完美处理多行断句和奇葩标点。
    """
    records = []
    lines = raw_text.strip().splitlines()
    
    # 🚨 升级点1：扩充标点集，加入全/半角分号 [；;]
    pattern = re.compile(r"^(.+?)说[：:,，；;]\s*(.*)$")
    
    current_speaker = None
    current_text_blocks = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = pattern.match(line)
        if match:
            # 🚨 升级点2：如果匹配到新角色发言，先把上一个角色的【所有多行文本】合并保存
            if current_speaker is not None and current_text_blocks:
                joined_text = " ".join(current_text_blocks).strip()
                if joined_text:
                    records.append(ASRRecord(
                        record_id=f"{conv_id}_{len(records):04d}",
                        speaker_id=current_speaker,
                        raw_text=joined_text
                    ))
            
            # 更新当前正在说话的角色，并开启新的文本块收集
            current_speaker = match.group(1).strip()
            current_text_blocks = [match.group(2).strip()]
        else:
            # 🚨 升级点3：如果这行没匹配到“XXX说：”（比如孤立的“过来”），
            # 直接把它追加到当前正在说话的人的文本块里！
            if current_speaker is not None:
                current_text_blocks.append(line)
            else:
                # 极端兜底：如果文本第一行就没有“XXX说：”
                current_speaker = "Unknown"
                current_text_blocks = [line]
                
    # 循环结束后，不要忘记把最后一个人收集到的文本保存入库
    if current_speaker is not None and current_text_blocks:
        joined_text = " ".join(current_text_blocks).strip()
        if joined_text:
            records.append(ASRRecord(
                record_id=f"{conv_id}_{len(records):04d}",
                speaker_id=current_speaker,
                raw_text=joined_text
            ))
            
    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # 生产环境建议用 argparse 或 Hydra 从 CLI/YAML 加载配置
    # 此处使用 dataclass 默认值快速启动
    cfg = PipelineConfig(
        input_csv     = "data/asr_records.csv",   # 不存在时自动切换 Mock 数据
        output_jsonl  = "/home/zzh/923/output/intelligence_results.jsonl",
        fasttext_model= "/home/zzh/923/model/lid.176.bin",
        bge_model_name= "/home/zzh/923/model/dir",
        use_fp16      = False,   # 无 GPU 时设为 False
        intent_threshold = 0.75,
        preview_every = 1,       # 演示时每条都预览
    )
    run_pipeline(cfg)
