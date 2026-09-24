import os
import json
from transformers import BartTokenizer

import run_ablation_hpo as ra_hpo

def main():
    out_dir = r"C:\kaggle2\EEG-To-text\test\hpo_sentence_disjoint\03_bleu1_selector_strong_eeg\sa_ablation"
    os.makedirs(out_dir, exist_ok=True)
    ra_hpo.ABLATION_DIR = out_dir
    
    # We will use the HPO-03 overrides which are already in ra_hpo.build_config
    base_overrides = {"data_dir": "dataset"}
    
    base_cfg = ra_hpo.build_config(base_overrides)
    train_s, dev_s, test_s, pre = ra_hpo.load_and_split_data(base_cfg)
    tokenizer = BartTokenizer.from_pretrained(ra_hpo.BART, local_files_only=True)
    
    scales = [0.1, 0.3, 0.5, 1.0]
    
    results = {}
    out_json = os.path.join(out_dir, "sa_ablation_results.json")
    
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            results = json.load(f)
            
    for scale in scales:
        name = f"sa_scale_{scale}"
        cfg_overrides = {"self_attn_scale": scale, "data_dir": "dataset"}
        
        if name in results and results[name] is not None:
            print(f"Skipping {name}, already completed.")
            continue
                
        metrics = ra_hpo.run_single_ablation(
            name, cfg_overrides, {}, tokenizer, train_s, dev_s, test_s, pre
        )
        results[name] = metrics
        
        # Save intermediate
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
            
    print("\nDONE. Results:")
    for k, v in results.items():
        if v:
            print(f"{k}: BLEU-1={v.get('greedy_bleu1', 0):.4f}, BLEU-4={v.get('greedy_bleu4', 0):.4f}, ROUGE-1={v.get('greedy_rouge1', 0):.4f}")

if __name__ == "__main__":
    main()
