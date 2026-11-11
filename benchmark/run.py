"""Benchmark hallucination scorers on HaluEval (Li et al., 2023).

    python -m benchmark.run            # ~10-15 min on an M-series Mac

QA: evidence = the knowledge passage, output = "question + answer"; each item gives one faithful
and one hallucinated answer. Summarization: evidence = the source document, output = a faithful or
hallucinated summary, scored sentence by sentence against overlapping document windows (the
clause-level idea from detection.py, done with a cross-encoder).

Thresholds for accuracy are picked on a dev split and applied to a disjoint test split.
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from benchmark.scorers import BERT_MNLI, DEBERTA_NLI, NLI, Embedder, chunks, cosine, novel_token_rate, sentences

OUT = Path("benchmark/results")
N_QA = 2000
N_SUM = 200
SEED = 0


def qa_pairs():
    d = load_dataset("pminervini/HaluEval", "qa")["data"].shuffle(seed=SEED).select(range(N_QA))
    ev, out, y, item = [], [], [], []
    for i, r in enumerate(d):
        for ans, lab in ((r["right_answer"], 0), (r["hallucinated_answer"], 1)):
            ev.append(r["knowledge"])
            out.append(f"{r['question']} {ans}")
            y.append(lab)
            item.append(i)
    return ev, out, np.array(y), np.array(item)


def sum_pairs():
    d = load_dataset("pminervini/HaluEval", "summarization")["data"].shuffle(seed=SEED).select(range(N_SUM))
    ev, out, y, item = [], [], [], []
    for i, r in enumerate(d):
        for s, lab in ((r["right_summary"], 0), (r["hallucinated_summary"], 1)):
            ev.append(r["document"])
            out.append(s)
            y.append(lab)
            item.append(i)
    return ev, out, np.array(y), np.array(item)


def score_qa(ev, out, bert, deberta, emb):
    s = {}
    # 1. the original detection.py method: each text through the classifier alone, cosine of the
    #    two 3-way logit vectors, higher similarity = more faithful
    s["original: cosine of nli logits"] = -cosine(bert.single_logits(ev), bert.single_logits(out))
    # 2. the same bert-mnli used as intended, as a cross-encoder over (evidence, output)
    p = bert.probs(ev, out)
    s["bert-mnli cross-encoder, 1 - p(entail)"] = 1 - p[:, bert.ent]
    s["bert-mnli cross-encoder, p(contradict)"] = p[:, bert.con]
    # 3. a stronger nli model
    p = deberta.probs(ev, out)
    s["deberta-v3 nli, 1 - p(entail)"] = 1 - p[:, deberta.ent]
    s["deberta-v3 nli, p(contradict)"] = p[:, deberta.con]
    # 4. sentence-embedding similarity and 5. lexical novelty baselines
    s["embedding cosine (minilm)"] = -cosine(emb.encode(ev), emb.encode(out))
    s["novel-token rate"] = novel_token_rate(ev, out)
    return s


def score_sum(ev, out, nli):
    """Per summary sentence: best entailment over document windows. Summary score: worst sentence."""
    scores_min, scores_mean, n_pairs = [], [], 0
    for doc, summ in zip(ev, out):
        wins = chunks(doc, nli.tok)
        sents = sentences(summ)
        P = [w for _ in sents for w in wins]
        H = [s for s in sents for _ in wins]
        n_pairs += len(P)
        ent = nli.probs(P, H)[:, nli.ent].reshape(len(sents), len(wins)).max(1)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        scores_min.append(1 - ent.min())
        scores_mean.append(1 - ent.mean())
    return np.array(scores_min), np.array(scores_mean), n_pairs


def evaluate(scores, y, item):
    """auroc/auprc on all pairs; accuracy with a threshold fit on the dev half of items."""
    dev = item % 2 == 0
    out = {}
    for name, s in scores.items():
        fpr, tpr, thr = roc_curve(y[dev], s[dev])
        t = thr[np.argmax(tpr - fpr)]
        test = ~dev
        out[name] = {
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "test_acc_at_dev_threshold": float(np.mean((s[test] >= t) == y[test])),
            # pairwise: is the hallucinated answer scored above the faithful one for the same item?
            "pairwise_acc": float(np.mean(s[y == 1] > s[y == 0])),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bert, deberta, emb = NLI(BERT_MNLI), NLI(DEBERTA_NLI), Embedder()

    # the qa pass is the slow one; reuse its scores if a previous run saved them
    if (OUT / "qa_scores.npz").exists():
        z = np.load(OUT / "qa_scores.npz")
        y, item = z["y"], z["item"]
        qa = {k: z[k] for k in z.files if k not in ("y", "item")}
    else:
        ev, out, y, item = qa_pairs()
        qa = score_qa(ev, out, bert, deberta, emb)
        np.savez(OUT / "qa_scores.npz", y=y, item=item, **{k: v for k, v in qa.items()})

    # long document windows: small batches keep memory flat on a 16 GB machine
    bert.batch = deberta.batch = 8

    # the original thresholds: >0.8 entailment, <0.4 contradiction, applied to the raw cosine
    raw_cos = -qa["original: cosine of nli logits"]
    bands = {
        "faithful": {
            b: float(np.mean(f(raw_cos[y == 0])))
            for b, f in [
                (">0.8", lambda c: c > 0.8),
                ("0.4-0.8", lambda c: (c >= 0.4) & (c <= 0.8)),
                ("<0.4", lambda c: c < 0.4),
            ]
        },
        "hallucinated": {
            b: float(np.mean(f(raw_cos[y == 1])))
            for b, f in [
                (">0.8", lambda c: c > 0.8),
                ("0.4-0.8", lambda c: (c >= 0.4) & (c <= 0.8)),
                ("<0.4", lambda c: c < 0.4),
            ]
        },
    }

    sev, sout, sy, sitem = sum_pairs()
    smin, smean, n_pairs = score_sum(sev, sout, deberta)
    bmin, bmean, _ = score_sum(sev, sout, bert)
    whole = 1 - deberta.probs([chunks(d, deberta.tok)[0] for d in sev], sout)[:, deberta.ent]
    summ = {
        "deberta-v3, sentence-level (worst sentence)": smin,
        "deberta-v3, sentence-level (mean)": smean,
        "deberta-v3, whole summary vs first window": whole,
        "bert-mnli, sentence-level (worst sentence)": bmin,
        "embedding cosine (minilm)": -cosine(emb.encode(sev), emb.encode(sout)),
        "novel-token rate": novel_token_rate(sev, sout),
    }
    np.savez(OUT / "sum_scores.npz", y=sy, item=sitem, **summ)

    res = {
        "qa": {"n_items": N_QA, "n_pairs": len(y), "methods": evaluate(qa, y, item), "original_threshold_bands": bands},
        "summarization": {
            "n_items": N_SUM,
            "n_pairs": len(sy),
            "nli_pairs_scored": n_pairs,
            "methods": evaluate(summ, sy, sitem),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    write_table(res)
    from benchmark.plots import make_all

    make_all()


def write_table(res):
    lines = []
    for task in ("qa", "summarization"):
        lines += [f"### {task}", "", "| method | auroc | auprc | test acc | pairwise acc |", "|---|---|---|---|---|"]
        for name, m in sorted(res[task]["methods"].items(), key=lambda kv: -kv[1]["auroc"]):
            lines.append(
                f"| {name} | {m['auroc']:.3f} | {m['auprc']:.3f} | {m['test_acc_at_dev_threshold']:.1%} | {m['pairwise_acc']:.1%} |"
            )
        lines.append("")
    b = res["qa"]["original_threshold_bands"]
    lines += [
        "original 0.8 / 0.4 bands on qa (share of answers in each band):",
        "",
        '| answers | >0.8 "entailment" | 0.4-0.8 "neutral" | <0.4 "contradiction" |',
        "|---|---|---|---|",
    ]
    for k in ("faithful", "hallucinated"):
        lines.append(f"| {k} | {b[k]['>0.8']:.1%} | {b[k]['0.4-0.8']:.1%} | {b[k]['<0.4']:.1%} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
