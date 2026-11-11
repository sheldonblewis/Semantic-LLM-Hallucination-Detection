### qa

| method | auroc | auprc | test acc | pairwise acc |
|---|---|---|---|---|
| deberta-v3 nli, 1 - p(entail) | 0.794 | 0.837 | 74.8% | 77.8% |
| deberta-v3 nli, p(contradict) | 0.732 | 0.794 | 69.5% | 72.0% |
| bert-mnli cross-encoder, 1 - p(entail) | 0.613 | 0.626 | 58.5% | 60.9% |
| bert-mnli cross-encoder, p(contradict) | 0.611 | 0.645 | 57.8% | 62.5% |
| novel-token rate | 0.581 | 0.577 | 55.0% | 72.5% |
| embedding cosine (minilm) | 0.554 | 0.546 | 54.9% | 67.1% |
| original: cosine of nli logits | 0.514 | 0.483 | 52.6% | 48.6% |

### summarization

| method | auroc | auprc | test acc | pairwise acc |
|---|---|---|---|---|
| novel-token rate | 0.751 | 0.721 | 67.0% | 77.5% |
| deberta-v3, whole summary vs first window | 0.651 | 0.671 | 57.0% | 68.5% |
| bert-mnli, sentence-level (worst sentence) | 0.644 | 0.653 | 60.0% | 63.0% |
| deberta-v3, sentence-level (mean) | 0.627 | 0.639 | 58.0% | 63.0% |
| deberta-v3, sentence-level (worst sentence) | 0.617 | 0.627 | 55.5% | 62.5% |
| embedding cosine (minilm) | 0.507 | 0.514 | 47.5% | 49.5% |

original 0.8 / 0.4 bands on qa (share of answers in each band):

| answers | >0.8 "entailment" | 0.4-0.8 "neutral" | <0.4 "contradiction" |
|---|---|---|---|
| faithful | 33.5% | 20.6% | 46.0% |
| hallucinated | 27.0% | 24.2% | 48.8% |
