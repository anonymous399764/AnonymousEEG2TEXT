import run_subject_split as rss
import os

# Apply HPO-03 overrides to the subject split config
original_build_config = rss.build_config
def patched_build_config():
    cfg = original_build_config()
    cfg.phase1_epochs = 30
    cfg.phase2_epochs = 70
    cfg.phase1_lr = 8e-05
    cfg.phase2_lr = 2.5e-05
    cfg.lambda_contrastive = 3.0
    cfg.lambda_attn_entropy = 0.15
    cfg.word_dropout_start = 0.35
    cfg.word_dropout_end = 0.85
    cfg.eeg_prior_lambda = 1.2
    cfg.eeg_noise_std = 0.05
    cfg.eeg_channel_drop = 0.02
    cfg.self_attn_scale = 0.0
    return cfg

rss.build_config = patched_build_config
out_dir = r"C:\kaggle2\EEG-To-text\test\hpo_sentence_disjoint\03_bleu1_selector_strong_eeg\subject_split"
os.makedirs(out_dir, exist_ok=True)
rss.OUT_DIR = out_dir
rss.log_path = os.path.join(out_dir, "training.log")

if __name__ == "__main__":
    rss.main()
