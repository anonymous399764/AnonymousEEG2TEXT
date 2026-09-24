import os
import numpy as np
import h5py
from glob import glob
from tqdm import tqdm
import pickle


def load_matlab_string(matlab_string_obj):
    """Load string from MATLAB h5py reference object"""
    try:
        # Handle different data types
        if isinstance(matlab_string_obj, str):
            return matlab_string_obj
        elif isinstance(matlab_string_obj, bytes):
            return matlab_string_obj.decode('utf-8')
        elif isinstance(matlab_string_obj, h5py.Reference):
            return ''.join(chr(c) for c in matlab_string_obj)
        elif hasattr(matlab_string_obj, 'dtype'):
            # It's a numpy array - handle both 1D and 2D arrays
            arr = np.squeeze(matlab_string_obj)
            if arr.ndim == 0:
                # Scalar
                return str(arr.item())
            else:
                # Array of character codes
                return ''.join(chr(int(c)) for c in arr.flatten())
        else:
            return str(matlab_string_obj)
    except Exception as e:
        # Fallback: return string representation
        return str(matlab_string_obj)


def extract_word_level_data(f, word_ref):
    """Extract word level data from MATLAB structure"""
    word_data = []
    word_tokens_all = []
    word_tokens_has_fixation = []
    word_tokens_with_mask = []
    
    try:
        # Dereference to get the actual word data object
        word_data_obj = f[word_ref]
        
        # Check if it's a Group (has keys) or Dataset
        if not isinstance(word_data_obj, h5py.Group):
            # It's a Dataset, not a structured group
            return [], [], [], []
        
        word_keys = list(word_data_obj.keys())
        
        # Check if we have the expected keys
        if 'content' not in word_keys:
            return None, [], [], []
        
        contentData = word_data_obj['content']
        nFixData = word_data_obj['nFixations']
        
        # Get optional EEG data
        has_ffd = 'FFD_t1' in word_keys
        has_gd = 'GD_t1' in word_keys
        has_trt = 'TRT_t1' in word_keys
        
        num_words = len(contentData)
        
        for widx in range(num_words):
            # Get word content
            word_content_ref = contentData[widx][0]
            word_content = load_matlab_string(f[word_content_ref])
            word_tokens_all.append(word_content)
            
            # Get number of fixations - handle various formats
            try:
                nFix_ref = nFixData[widx][0]
                nFix_data = np.squeeze(f[nFix_ref][()])
                # Handle scalar vs array
                if isinstance(nFix_data, np.ndarray):
                    if nFix_data.size == 1:
                        nFix = int(nFix_data.item())
                    else:
                        # If array has multiple values, take first or default to 0
                        nFix = int(nFix_data.flatten()[0]) if nFix_data.size > 0 else 0
                else:
                    nFix = int(nFix_data)
            except Exception as e:
                print(f"      Warning: Could not extract nFix for word '{word_content}': {e}")
                nFix = 0
            
            data_dict = {'content': word_content, 'nFix': nFix}
            
            if nFix > 0 and has_gd and has_ffd and has_trt:
                # Extract EEG features for each frequency band
                gd_eeg = []
                ffd_eeg = []
                trt_eeg = []
                
                freq_bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
                
                try:
                    for band in freq_bands:
                        # GD (Gaze Duration)
                        gd_key = f'GD_{band}'
                        if gd_key in word_data_obj:
                            gd_ref = word_data_obj[gd_key][widx][0]
                            gd_eeg.append(np.squeeze(f[gd_ref][()]))
                        
                        # FFD (First Fixation Duration)
                        ffd_key = f'FFD_{band}'
                        if ffd_key in word_data_obj:
                            ffd_ref = word_data_obj[ffd_key][widx][0]
                            ffd_eeg.append(np.squeeze(f[ffd_ref][()]))
                        
                        # TRT (Total Reading Time)
                        trt_key = f'TRT_{band}'
                        if trt_key in word_data_obj:
                            trt_ref = word_data_obj[trt_key][widx][0]
                            trt_eeg.append(np.squeeze(f[trt_ref][()]))
                    
                    if len(gd_eeg) == 8 and len(ffd_eeg) == 8 and len(trt_eeg) == 8:
                        data_dict['GD_EEG'] = gd_eeg
                        data_dict['FFD_EEG'] = ffd_eeg
                        data_dict['TRT_EEG'] = trt_eeg
                        word_tokens_has_fixation.append(word_content)
                        word_tokens_with_mask.append(word_content)
                    else:
                        word_tokens_with_mask.append('[MASK]')
                except Exception as e:
                    # If EEG extraction fails for this word, skip it
                    word_tokens_with_mask.append('[MASK]')
            else:
                word_tokens_with_mask.append('[MASK]')
            
            word_data.append(data_dict)
        
        return word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask
    
    except Exception as e:
        print(f"Error extracting word data: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], []


def process_zuco_v2_task2_tsr(rootdir, output_dir):
    """Process ZuCo 2.0 task2-TSR MATLAB files into pickle format"""
    
    task = "NR"
    
    print('='*60)
    print(f'Processing ZuCo 2.0 task3-NR...')
    print(f'Input directory: {rootdir}')
    print('='*60)
    
    dataset_dict = {}
    
    # Find all TSR.mat files
    mat_files = sorted(glob(os.path.join(rootdir, f'*{task}.mat')))
    
    if not mat_files:
        print(f'❌ No .mat files found in {rootdir}')
        return None
    
    print(f'Found {len(mat_files)} .mat files')
    print(f'Files: {[os.path.basename(f) for f in mat_files]}')
    print()
    
    for file_path in tqdm(mat_files, desc='Processing files'):
        filename = os.path.basename(file_path)
        
        # Extract subject ID from filename (e.g., "resultsYAC_TSR.mat" -> "YAC")
        try:
            subject = filename.split("ts")[1].split("_")[0]
        except:
            # Alternative parsing: results{SUBJECT}_TSR.mat
            subject = filename.replace('results', '').replace(f'_{task}.mat', '').replace('.mat', '')
        
        # Exclude YMH due to incomplete data (dyslexia)
        if subject == 'YMH':
            print(f'  Skipping subject YMH (incomplete data)')
            continue
        
        if subject in dataset_dict:
            print(f'  Warning: Duplicate subject {subject}, skipping...')
            continue
        
        dataset_dict[subject] = []
        
        try:
            # Load MATLAB file using h5py (for v7.3 format)
            f = h5py.File(file_path, 'r')
            
            sentence_data = f['sentenceData']
            
            # Get sentence-level EEG references
            mean_t1_objs = sentence_data['mean_t1']
            mean_t2_objs = sentence_data['mean_t2']
            mean_a1_objs = sentence_data['mean_a1']
            mean_a2_objs = sentence_data['mean_a2']
            mean_b1_objs = sentence_data['mean_b1']
            mean_b2_objs = sentence_data['mean_b2']
            mean_g1_objs = sentence_data['mean_g1']
            mean_g2_objs = sentence_data['mean_g2']
            
            contentData = sentence_data['content']
            wordData = sentence_data['word']
            
            num_sentences = len(contentData)
            
            for idx in range(num_sentences):
                # Get sentence string
                obj_reference_content = contentData[idx][0]
                sent_string = load_matlab_string(f[obj_reference_content])
                
                sent_obj = {'content': sent_string}
                
                # Get sentence level EEG
                sent_obj['sentence_level_EEG'] = {
                    'mean_t1': np.squeeze(f[mean_t1_objs[idx][0]][()]),
                    'mean_t2': np.squeeze(f[mean_t2_objs[idx][0]][()]),
                    'mean_a1': np.squeeze(f[mean_a1_objs[idx][0]][()]),
                    'mean_a2': np.squeeze(f[mean_a2_objs[idx][0]][()]),
                    'mean_b1': np.squeeze(f[mean_b1_objs[idx][0]][()]),
                    'mean_b2': np.squeeze(f[mean_b2_objs[idx][0]][()]),
                    'mean_g1': np.squeeze(f[mean_g1_objs[idx][0]][()]),
                    'mean_g2': np.squeeze(f[mean_g2_objs[idx][0]][()])
                }
                
                sent_obj['word'] = []
                
                # Get word level data - pass the reference, not dereferenced object
                word_ref = wordData[idx][0]
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = \
                    extract_word_level_data(f, word_ref)
                
                if not word_data or len(word_data) == 0:
                    print(f'    Missing sentence: subj:{subject} content:{sent_string[:50]}..., append None')
                    dataset_dict[subject].append(None)
                    continue
                elif len(word_tokens_all) == 0:
                    print(f'    No word features: subj:{subject} content:{sent_string[:50]}..., append None')
                    dataset_dict[subject].append(None)
                    continue
                else:
                    for widx in range(len(word_data)):
                        data_dict = word_data[widx]
                        word_obj = {'content': data_dict['content'], 'nFixations': data_dict['nFix']}
                        
                        if 'GD_EEG' in data_dict:
                            gd = data_dict["GD_EEG"]
                            ffd = data_dict["FFD_EEG"]
                            trt = data_dict["TRT_EEG"]
                            
                            if len(gd) == len(trt) == len(ffd) == 8:
                                word_obj['word_level_EEG'] = {
                                    'GD': {'GD_t1': gd[0], 'GD_t2': gd[1], 'GD_a1': gd[2], 'GD_a2': gd[3],
                                          'GD_b1': gd[4], 'GD_b2': gd[5], 'GD_g1': gd[6], 'GD_g2': gd[7]},
                                    'FFD': {'FFD_t1': ffd[0], 'FFD_t2': ffd[1], 'FFD_a1': ffd[2], 'FFD_a2': ffd[3],
                                           'FFD_b1': ffd[4], 'FFD_b2': ffd[5], 'FFD_g1': ffd[6], 'FFD_g2': ffd[7]},
                                    'TRT': {'TRT_t1': trt[0], 'TRT_t2': trt[1], 'TRT_a1': trt[2], 'TRT_a2': trt[3],
                                           'TRT_b1': trt[4], 'TRT_b2': trt[5], 'TRT_g1': trt[6], 'TRT_g2': trt[7]}
                                }
                                sent_obj['word'].append(word_obj)
                    
                    sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                    sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                    sent_obj['word_tokens_all'] = word_tokens_all
                    
                    dataset_dict[subject].append(sent_obj)
            
            f.close()
            
        except Exception as e:
            print(f'  ❌ Error processing {filename}: {e}')
            continue
    
    # Save output
    if dataset_dict == {}:
        print(f'\n❌ No data processed for task2-TSR-2.0')
        return None
    
    task_name = 'task2-TSR-2.0'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{task_name}-dataset.pickle')
    
    with open(output_file, 'wb') as handle:
        pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'\n{"="*60}')
    print(f'✓ Successfully saved to: {output_file}')
    print(f'{"="*60}')
    print(f'Subjects: {list(dataset_dict.keys())}')
    print(f'Number of subjects: {len(dataset_dict)}')
    
    # Print statistics for each subject
    for subject, sentences in dataset_dict.items():
        valid_sentences = [s for s in sentences if s is not None]
        print(f'  {subject}: {len(valid_sentences)} valid sentences (out of {len(sentences)} total)')
    
    return dataset_dict


def main():
    # UPDATE THIS PATH to your ZuCo 2.0 task2-TSR directory
    input_dir = 'd:/kaggle2/EEG-To-text/task2.0-NR/'  # or wherever your .mat files are
    output_dir = 'd:/kaggle2/EEG-To-text/dataset/'
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f'❌ ERROR: Input directory not found: {input_dir}')
        print(f'\nPlease update the input_dir variable to point to your ZuCo 2.0 task2-TSR folder')
        print(f'Expected files: resultsYAC_TSR.mat, resultsYAG_TSR.mat, etc.')
        return
    
    # Process the data
    dataset_dict = process_zuco_v2_task2_tsr(input_dir, output_dir)
    
    if dataset_dict:
        print('\n✅ Conversion completed successfully!')
    else:
        print('\n❌ Conversion failed')


if __name__ == '__main__':
    main()
