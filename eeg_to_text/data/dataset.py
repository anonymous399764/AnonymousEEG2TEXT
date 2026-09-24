"""
ZuCo EEG-to-Text PyTorch Dataset and Collate Function

Loads pre-extracted (eeg_matrix, sentence) pairs, tokenises text with BART
tokenizer, pads EEG to max_words, and returns ready-to-train batches.

This module works with the EXISTING pickle format (8-band, 840-dim) produced
by the project's Data.py / create_pickle_file scripts.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from transformers import BartTokenizer


class ZuCoEEGDataset(Dataset):
    """
    PyTorch dataset that wraps pre-extracted (eeg_matrix, text) pairs.

    Each sample contains:
        eeg          – float tensor  (max_words, eeg_dim)
        eeg_mask     – long tensor   (max_words,)  1=real word, 0=padding
        input_ids    – long tensor   (max_text_len,)
        attention_mask – long tensor (max_text_len,)
        labels       – long tensor   (max_text_len,)  pad positions → -100
        raw_text     – original sentence string (kept for evaluation)
    """

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, str]],
        tokenizer: BartTokenizer,
        max_words: int = 56,
        max_text_len: int = 56,
        augment: bool = False,
        noise_std: float = 0.1,
        channel_drop: float = 0.05,
        time_shift: int = 1,
    ):
        """
        Args:
            samples:      List of (eeg_matrix, sentence_text) from preprocessing.
            tokenizer:    BART tokenizer instance.
            max_words:    Pad/truncate EEG to this many words.
            max_text_len: Pad/truncate text tokens to this length.
            augment:      Enable stochastic EEG augmentation (training only).
            noise_std:    Gaussian noise std for augmentation.
            channel_drop: Probability of zeroing individual feature channels.
            time_shift:   Max word positions to shift EEG (circular).
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_text_len = max_text_len
        self.augment = augment
        self.noise_std = noise_std
        self.channel_drop = channel_drop
        self.time_shift = time_shift

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        eeg_np, text = self.samples[idx]
        num_words, eeg_dim = eeg_np.shape

        # ── EEG: pad / truncate to max_words ────────────────────────────
        if num_words > self.max_words:
            eeg_np = eeg_np[: self.max_words]
            num_words = self.max_words

        padded = np.zeros((self.max_words, eeg_dim), dtype=np.float32)
        padded[:num_words] = eeg_np
        eeg_tensor = torch.from_numpy(padded)                  # (max_words, D)

        eeg_mask = torch.zeros(self.max_words, dtype=torch.long)
        eeg_mask[:num_words] = 1                               # 1 = real, 0 = pad

        # ── EEG augmentation (training only) ────────────────────────────
        if self.augment:
            # 1. Additive Gaussian noise
            if self.noise_std > 0:
                noise = torch.randn_like(eeg_tensor) * self.noise_std
                noise[num_words:] = 0  # don't add noise to padding
                eeg_tensor = eeg_tensor + noise

            # 2. Feature/channel dropout (zero random features)
            if self.channel_drop > 0:
                drop_mask = (torch.rand(eeg_dim) < self.channel_drop).float()
                # Scale up surviving features to keep expected value
                scale = 1.0 / (1.0 - self.channel_drop + 1e-8)
                eeg_tensor = eeg_tensor * (1.0 - drop_mask) * scale

            # 3. Time shift (circular shift of word positions)
            if self.time_shift > 0 and num_words > 2:
                shift = np.random.randint(-self.time_shift, self.time_shift + 1)
                if shift != 0:
                    eeg_data = eeg_tensor[:num_words]
                    eeg_data = torch.roll(eeg_data, shifts=shift, dims=0)
                    eeg_tensor[:num_words] = eeg_data

        # ── Text: tokenise with BART tokenizer ──────────────────────────
        tok = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = tok["input_ids"].squeeze(0)                # (max_text_len,)
        attention_mask = tok["attention_mask"].squeeze(0)       # (max_text_len,)

        # Labels: copy of input_ids with padding set to -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "eeg": eeg_tensor,
            "eeg_mask": eeg_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_text": text,
        }


def eeg_collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """
    Custom collate: stacks tensors and gathers raw_text into a list.
    """
    return {
        "eeg": torch.stack([b["eeg"] for b in batch]),                # (B, W, D)
        "eeg_mask": torch.stack([b["eeg_mask"] for b in batch]),      # (B, W)
        "input_ids": torch.stack([b["input_ids"] for b in batch]),    # (B, T)
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),          # (B, T)
        "raw_text": [b["raw_text"] for b in batch],
    }


# ── Train / Dev / Test helpers ──────────────────────────────────────────────

def split_samples(
    samples: List[Tuple[np.ndarray, str]],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split by UNIQUE SENTENCE TEXT, then assign all EEG recordings of each
    sentence to the same split.  This prevents data leakage where the BART
    decoder memorises evaluation sentences during training.

    Without this fix, ZuCo's multi-subject design means the same sentence
    appears ~10× (once per subject).  A naive random split leaks nearly 100%
    of dev/test sentences into training → inflated scores.
    """
    from collections import defaultdict

    rng = np.random.RandomState(seed)

    # Group all samples by their sentence text
    text_to_samples: Dict[str, List[Tuple[np.ndarray, str]]] = defaultdict(list)
    for eeg, text in samples:
        text_to_samples[text].append((eeg, text))

    # Shuffle unique texts, then split
    unique_texts = list(text_to_samples.keys())
    rng.shuffle(unique_texts)

    n = len(unique_texts)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train_texts = set(unique_texts[:train_end])
    dev_texts = set(unique_texts[train_end:dev_end])
    test_texts = set(unique_texts[dev_end:])

    train, dev, test = [], [], []
    for text in train_texts:
        train.extend(text_to_samples[text])
    for text in dev_texts:
        dev.extend(text_to_samples[text])
    for text in test_texts:
        test.extend(text_to_samples[text])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    # Verify no leakage
    assert len(train_texts & dev_texts) == 0, "Train-Dev text overlap!"
    assert len(train_texts & test_texts) == 0, "Train-Test text overlap!"
    assert len(dev_texts & test_texts) == 0, "Dev-Test text overlap!"

    print(f"[Split] {n} unique sentences -> "
          f"train={len(train)} ({len(train_texts)} unique), "
          f"dev={len(dev)} ({len(dev_texts)} unique), "
          f"test={len(test)} ({len(test_texts)} unique)")
    print(f"[Split] Sentence overlap: train&dev={len(train_texts & dev_texts)}, "
          f"train&test={len(train_texts & test_texts)}, "
          f"dev&test={len(dev_texts & test_texts)}")
    return train, dev, test


def split_samples_by_subject(
    samples: List[Tuple[np.ndarray, str, str]],
    seed: int = 42,
    n_test: int = 2,
    n_dev: int = 1,
) -> Tuple[List, List, List]:
    """
    Split by SUBJECT: hold out some subjects for dev/test.
    Same sentences appear in train and test from different subjects.
    This makes the task 'EEG-conditioned retrieval' rather than
    'novel generation', which forces the model to actually rely on
    EEG patterns.  Noise baseline should fail dramatically.

    ZuCo 1.0 subjects (Z*): 12 subjects across 3 tasks

    Default split (seed=42):
        Test:  2 subjects
        Dev:   1 subject
        Train: 9 subjects

    Returns (eeg, text) pairs with subject info dropped.
    """
    rng = np.random.RandomState(seed)

    all_subjects = sorted(set(s for _, _, s in samples))
    rng.shuffle(all_subjects)

    # Assign subjects to splits
    test_subjects = set(all_subjects[:n_test])
    dev_subjects = set(all_subjects[n_test:n_test + n_dev])
    train_subjects = set(all_subjects) - test_subjects - dev_subjects

    # Split samples (drop subject from output tuples)
    train, dev, test = [], [], []
    for eeg, text, subj in samples:
        pair = (eeg, text)
        if subj in train_subjects:
            train.append(pair)
        elif subj in dev_subjects:
            dev.append(pair)
        elif subj in test_subjects:
            test.append(pair)

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    # Stats
    train_texts = set(t for _, t in train)
    dev_texts = set(t for _, t in dev)
    test_texts = set(t for _, t in test)

    print(f"[SubjectSplit] {len(all_subjects)} subjects -> "
          f"train={len(train_subjects)} subj ({len(train)} samples, "
          f"{len(train_texts)} unique sents), "
          f"dev={len(dev_subjects)} subj ({len(dev)} samples, "
          f"{len(dev_texts)} unique sents), "
          f"test={len(test_subjects)} subj ({len(test)} samples, "
          f"{len(test_texts)} unique sents)")
    print(f"[SubjectSplit] Train: {sorted(train_subjects)}")
    print(f"[SubjectSplit] Dev:   {sorted(dev_subjects)}")
    print(f"[SubjectSplit] Test:  {sorted(test_subjects)}")
    shared = train_texts & test_texts
    

    return train, dev, test
