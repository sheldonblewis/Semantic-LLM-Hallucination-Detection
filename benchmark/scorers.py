"""Hallucination scorers. Each returns a score per (evidence, output) pair; higher = more likely
hallucinated. All run locally on CPU or Apple MPS."""

import re

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
BERT_MNLI = "textattack/bert-base-uncased-MNLI"  # same recipe as model_tuning.py: bert-base fine-tuned on MNLI
DEBERTA_NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


class NLI:
    def __init__(self, name, batch=32):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
        self.batch = batch
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        if "entailment" in labels:
            self.ent, self.con = labels["entailment"], labels["contradiction"]
        else:
            # textattack's MNLI checkpoints ship LABEL_0..2 in the GLUE order (0 contradiction, 1 entailment, 2 neutral)
            self.con, self.ent = 0, 1

    @torch.no_grad()
    def probs(self, premises, hypotheses):
        """P(label | premise, hypothesis) as a cross-encoder, shape (n, 3)."""
        out = []
        for i in range(0, len(premises), self.batch):
            enc = self.tok(
                premises[i : i + self.batch],
                hypotheses[i : i + self.batch],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def single_logits(self, texts):
        """What detection.py does: run each text through the classifier *alone* and keep the logits."""
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(
                texts[i : i + self.batch], truncation=True, max_length=512, padding=True, return_tensors="pt"
            ).to(DEVICE)
            out.append(self.model(**enc).logits.float().cpu())
        return torch.cat(out).numpy()


class Embedder:
    def __init__(self, name=MINILM, batch=64):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(DEVICE).eval()
        self.batch = batch

    @torch.no_grad()
    def encode(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(
                texts[i : i + self.batch], truncation=True, max_length=256, padding=True, return_tensors="pt"
            ).to(DEVICE)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out).numpy()


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


WORD = re.compile(r"[a-z0-9]+")
STOP = set(
    "the a an of in on at to for and or is was were are be by with as from that this it its his her their".split()
)


def novel_token_rate(evidence, outputs):
    """Share of content words in the output that never appear in the evidence."""
    scores = []
    for e, o in zip(evidence, outputs):
        ev = set(WORD.findall(e.lower()))
        words = [w for w in WORD.findall(o.lower()) if w not in STOP]
        scores.append(np.mean([w not in ev for w in words]) if words else 0.0)
    return np.array(scores)


SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"])')


def sentences(text):
    return [s.strip() for s in SENT.split(text.strip()) if len(s.strip()) > 3] or [text.strip()]


def chunks(text, tok, max_tokens=380, stride=300):
    """Overlapping windows of a long document, so every part of it fits beside a hypothesis."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [text]
    return [tok.decode(ids[i : i + max_tokens]) for i in range(0, max(1, len(ids) - max_tokens + stride), stride)]
