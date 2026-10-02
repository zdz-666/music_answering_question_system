"""用 RGB 中文数据集（zh_refine.json）评测本项目的 RAG 链路。

数据集：https://github.com/chen700564/RGB —— 300 条中文样本，字段
`id / query / answer / positive / negative`。

评测时先把样本自带的 positive + negative 合并成一篇文本，再用项目自己的
SemanticChunker（dynamic_chunk.py）切成块，把这些块当作知识库内容传给
rag.get_result_evaluate(query, external_docs=...)：通道规划、集合路由、
网络搜索被跳过，查询重写 → 混合检索 → 重排 → 补充检索 → 自反思 →
生成这整条链路保持与线上一致。

评测方法：
- 逐条调 rag.get_result_evaluate(query, external_docs=...)，拿回 (检索到的上下文, 生成答案)；
- 用 RGB 官方 checkanswer 做子串匹配：参考答案里的每一条都出现在生成答案中才算命中；
- 汇总准确率（对齐 RGB 的 all_rate）与拒答率，逐条明细落 JSONL，支持断点续跑。

默认只有进度条与最终汇总，链路日志压到 WARNING；需要逐步排查时加 --verbose。

用法：
    python rgb_eval.py                   # 跑全部 300 条
    python rgb_eval.py --limit 20        # 先跑 20 条试水
    python rgb_eval.py --offset 20 --limit 20
    python rgb_eval.py --chunk-size 500   # 换一个切块上限
    python rgb_eval.py --verbose         # 额外打印完整链路 JSON 日志
"""

import argparse
import json
import logging
import os
import time
import urllib.request

from tqdm import tqdm

import rag
from dynamic_chunk import SemanticChunker
from observability import (
    ROOT_LOGGER_NAME,
    logger,
    new_request_id,
    reset_request_id,
    set_request_id,
)

# RGB 仓库默认分支是 master（不是 main）
RGB_RAW_URL = "https://raw.githubusercontent.com/chen700564/RGB/master/data/zh_refine.json"
DEFAULT_DATASET = os.path.join("data", "zh_refine.json")
DEFAULT_OUTPUT = os.path.join("eval_results", "rgb_zh_predictions.jsonl")


def ensure_dataset(path: str) -> str:
    """数据集不在本地时自动下载；下载失败给出可照做的提示。"""
    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    logger.info("rgb_eval.download", extra={"fields": {"url": RGB_RAW_URL, "path": path}})
    try:
        urllib.request.urlretrieve(RGB_RAW_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"数据集下载失败（{type(exc).__name__}: {exc}）。"
            f"请手动下载 {RGB_RAW_URL} 保存到 {path} 后重试。"
        ) from exc

    return path


def load_instances(path: str) -> list:
    """zh_refine.json 是 JSONL：一行一条样本。"""
    instances = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def normalize(text: str) -> str:
    """匹配前去掉所有空白并转小写，避免「12 人」与「12人」这类空格差异导致漏判。"""
    return "".join(str(text).split()).lower()


def check_answer(prediction: str, ground_truth) -> list:
    """RGB 官方 checkanswer：参考答案逐条子串匹配，返回每条是否命中（0/1）。"""
    prediction = normalize(prediction)
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]

    labels = []
    for instance in ground_truth:
        if isinstance(instance, list):
            # 参考答案本身是一组同义写法，命中任一即可
            hit = any(normalize(item) in prediction for item in instance)
        else:
            hit = normalize(instance) in prediction
        labels.append(int(hit))
    return labels


def is_refusal(prediction: str) -> bool:
    """沿用 RGB 的拒答标记；本项目检索不到内容时也可能走到拒答分支。"""
    return "信息不足" in prediction


def load_records(path: str) -> list:
    """读回已落盘的逐条结果；最后一行若因中断而不完整，跳过即可。"""
    records = []
    if not os.path.exists(path):
        return records

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def build_corpus(instance: dict, chunker: SemanticChunker) -> list:
    """样本的 positive + negative 合并成一篇文本，再用项目的切分算法切成检索单元。

    positive 是能回答该问题的文档，negative 是语义相近但答不上的干扰文档；
    合并后交给 SemanticChunker，产生的每一块才是一个检索单元——这样评测与
    线上「整篇文档先切分再入库」的粒度一致。
    """
    parts = []
    for field in ("positive", "negative"):
        value = instance.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(item for item in value if str(item).strip())

    if not parts:
        return []

    # 段落之间用换行连接，还原线上读入整篇文档（TextLoader）后的文本形态
    text = "\n".join(parts)
    return [doc.page_content for doc in chunker.chunk_document(text)]


def evaluate_one(instance: dict, chunker: SemanticChunker) -> dict:
    """跑一条样本。单条失败不中断整轮评测，错误照常记进明细。"""
    query = instance["query"]
    # 每个 query 单独一个 request_id，跑完可用 trace_view.py 按 id 还原该条的调用树
    request_id = new_request_id()
    token = set_request_id(request_id)
    started = time.perf_counter()

    record = {
        "id": instance["id"],
        "query": query,
        "answers": instance.get("answer"),
        "request_id": request_id,
    }

    try:
        # 切块本身要调嵌入接口，可能被网络抖动打挂，放进 try 才不会让整轮白跑
        corpus = build_corpus(instance, chunker)
        context, prediction = rag.get_result_evaluate(query, external_docs=corpus)
    except Exception as exc:  # 单条异常（上游抖动等）不该让整轮白跑
        logger.error(
            "rgb_eval.query_failed",
            extra={
                "fields": {
                    "id": instance["id"],
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            },
        )
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
        )
        return record
    finally:
        reset_request_id(token)

    labels = check_answer(prediction, instance.get("answer"))
    record.update(
        {
            "prediction": prediction,
            "context": context,
            # 合并后切出的块数，也就是这次的检索单元总数
            "corpus_size": len(corpus),
            "labels": labels,
            # 对齐 RGB 的 all_rate 判定：参考答案全部命中才算这一条正确
            "correct": bool(labels) and all(label == 1 for label in labels),
            "refusal": is_refusal(prediction),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
    )
    return record


def summarize(records: list) -> dict:
    scored = [record for record in records if not record.get("error")]
    correct = sum(1 for record in scored if record.get("correct"))
    refusal = sum(1 for record in scored if record.get("refusal"))

    return {
        "total": len(records),
        "scored": len(scored),
        "errors": len(records) - len(scored),
        "correct": correct,
        # all_rate 与 RGB 论文口径一致：全部参考答案都命中的比例
        "all_rate": round(correct / len(scored), 4) if scored else 0.0,
        "refusal_rate": round(refusal / len(scored), 4) if scored else 0.0,
        "avg_elapsed_s": (
            round(sum(record["elapsed_s"] for record in scored) / len(scored), 2)
            if scored
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="用 RGB 中文数据集评测本项目的 RAG 链路")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="zh_refine.json 路径，缺失时自动下载"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="逐条结果落盘路径（JSONL，已完成的 id 会自动跳过）"
    )
    parser.add_argument("--limit", type=int, default=0, help="最多评测多少条，0 表示全部")
    parser.add_argument("--offset", type=int, default=0, help="从数据集第几条开始")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="合并文本的语义切块上限，与线上 collection_create 的 max_chunk_size 一致",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="每条之间的间隔秒数，防上游限流")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印完整链路 JSON 日志（默认压到 WARNING，只留进度条与最终汇总）",
    )
    args = parser.parse_args()

    # 链路日志由 observability 的 music_rag 日志器输出，这里统一调级：
    # 默认只放 WARNING 及以上（检索降级、单条失败这类仍会显示），
    # --verbose 恢复 INFO，才能看到 step.start/step.done/llm.usage。
    logging.getLogger(ROOT_LOGGER_NAME).setLevel(
        logging.INFO if args.verbose else logging.WARNING
    )

    instances = load_instances(ensure_dataset(args.dataset))[args.offset :]
    if args.limit > 0:
        instances = instances[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done = {record["id"] for record in load_records(args.output)}
    if done:
        logger.info("rgb_eval.resume", extra={"fields": {"finished": len(done)}})

    pending = [instance for instance in instances if instance["id"] not in done]
    logger.info(
        "rgb_eval.start",
        extra={"fields": {"selected": len(instances), "pending": len(pending)}},
    )

    # 切块器只建一次：它内部持有嵌入模型，每条样本重建一遍没有意义
    chunker = SemanticChunker(overlap_size=0, max_chunk_size=args.chunk_size)

    hits = 0
    failures = 0
    with open(args.output, "a", encoding="utf-8") as handle, tqdm(
        pending, desc="RGB 评测", unit="条", dynamic_ncols=True
    ) as bar:
        for instance in bar:
            record = evaluate_one(instance, chunker)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()  # 中途 Ctrl+C 也不丢已完成的结果

            if record.get("error"):
                failures += 1
            elif record.get("correct"):
                hits += 1
            bar.set_postfix_str(f"命中 {hits} 失败 {failures}")

            if args.sleep:
                time.sleep(args.sleep)

    summary = summarize(load_records(args.output))
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐条明细：{args.output}\n汇总指标：{summary_path}")
    return summary


if __name__ == "__main__":
    main()