import os
import pickle
import scipy.io as io
import numpy as np
from glob import glob
from tqdm import tqdm

def process_mat_files(mat_files, task_name):
    """Process all .mat files into a structured dataset for ZuCo v1."""
    dataset_dict = {}
    
    for mat_file in tqdm(mat_files, desc=f"Processing {task_name}"):
        subject_name = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
        dataset_dict[subject_name] = []

        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
        
        for sent in matdata:
            if isinstance(sent.word, float):
                dataset_dict[subject_name].append(None)
                continue
            
            sent_obj = {'content': sent.content}
            sent_obj['sentence_level_EEG'] = {
                'mean_t1': sent.mean_t1, 'mean_t2': sent.mean_t2,
                'mean_a1': sent.mean_a1, 'mean_a2': sent.mean_a2,
                'mean_b1': sent.mean_b1, 'mean_b2': sent.mean_b2,
                'mean_g1': sent.mean_g1, 'mean_g2': sent.mean_g2
            }

            if task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {
                    'answer_mean_t1': sent.answer_mean_t1, 'answer_mean_t2': sent.answer_mean_t2,
                    'answer_mean_a1': sent.answer_mean_a1, 'answer_mean_a2': sent.answer_mean_a2,
                    'answer_mean_b1': sent.answer_mean_b1, 'answer_mean_b2': sent.answer_mean_b2,
                    'answer_mean_g1': sent.answer_mean_g1, 'answer_mean_g2': sent.answer_mean_g2
                }
            
            sent_obj['word'] = []
            word_tokens_all = []
            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            
            for word in sent.word:
                # Handle nFixations which can be scalar, array, or other types
                nfix = word.nFixations
                if isinstance(nfix, np.ndarray):
                    nfix = int(nfix.flatten()[0]) if nfix.size > 0 else 0
                elif not isinstance(nfix, (int, float)):
                    nfix = 0
                else:
                    nfix = int(nfix)
                    
                word_obj = {'content': word.content, 'nFixations': nfix}
                
                if nfix > 0:
                    word_obj['word_level_EEG'] = {
                        'FFD': {f'FFD_{freq}': getattr(word, f'FFD_{freq}') for freq in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']},
                        'TRT': {f'TRT_{freq}': getattr(word, f'TRT_{freq}') for freq in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']},
                        'GD': {f'GD_{freq}': getattr(word, f'GD_{freq}') for freq in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']}
                    }
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append('[MASK]')
                
                word_tokens_all.append(word.content)

            sent_obj.update({
                'word_tokens_has_fixation': word_tokens_has_fixation,
                'word_tokens_with_mask': word_tokens_with_mask,
                'word_tokens_all': word_tokens_all
            })
            
            dataset_dict[subject_name].append(sent_obj)
    
    return dataset_dict

def main():
    # Process all three tasks
    tasks = {
        'task1-SR': 'd:/kaggle2/zuco_data/task1-SR/',
        'task2-NR': 'd:/kaggle2/zuco_data/task2-NR/',
        'task3-TSR': 'd:/kaggle2/zuco_data/task3-TSR/'
    }
    
    output_dir = 'd:/kaggle2/EEG-To-text/dataset/'
    os.makedirs(output_dir, exist_ok=True)
    
    for task_name, input_dir in tasks.items():
        print(f'\n{"="*50}')
        print(f'Processing {task_name}...')
        print(f'{"="*50}')
        
        mat_files = sorted(glob(os.path.join(input_dir, '*.mat')))
        
        if not mat_files:
            print(f'No .mat files found in {input_dir}')
            continue
        
        print(f'Found {len(mat_files)} .mat files')
        
        dataset_dict = process_mat_files(mat_files, task_name)
        
        # Save to pickle
        output_file = os.path.join(output_dir, f'{task_name}-dataset.pickle')
        with open(output_file, 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f'✓ Saved to {output_file}')
        print(f'  Subjects: {len(dataset_dict)}')
        print(f'  Subject IDs: {list(dataset_dict.keys())}')

if __name__ == '__main__':
    main()
