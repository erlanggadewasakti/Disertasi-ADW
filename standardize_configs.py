import json
import re
import os

files = [
    'mvsa-bert-bilstm.ipynb',
    'mvsa-bert-lstm.ipynb',
    'mvsa-bilstm.ipynb',
    'mvsa-lstm.ipynb',
    'mvsa-vit-bilstm.ipynb',
    'mvsa-vit-lstm.ipynb'
]

replacements = {
    r'("epochs"\s*:\s*)\d+': r'\g<1>30',
    r'("batch_size"\s*:\s*\d+)': r'"batch_size": 16',
    r'("weight_decay"\s*:\s*)[\d\.e-]+': r'\g<1>0.01',
    r'("early_stop_patience"\s*:\s*)\d+': r'\g<1>7',
    r'("gradient_clip"\s*:\s*)[\d\.e-]+': r'\g<1>1.0',
    r'("classifier_dropout"\s*:\s*)[\d\.e-]+': r'\g<1>0.25',
    r'("lr_main"\s*:\s*)[\d\.e-]+': r'\g<1>1e-4',
    r'("lr_vit"\s*:\s*)[\d\.e-]+': r'\g<1>1e-5',
    r'("bert_lr"\s*:\s*)[\d\.e-]+': r'\g<1>2e-5',
    r'("vision_lr"\s*:\s*)[\d\.e-]+': r'\g<1>1e-5',
    r'("new_params_lr"\s*:\s*)[\d\.e-]+': r'\g<1>1e-4',
    r'("vit_trainable_blocks"\s*:\s*)\d+': r'\g<1>4',
    r'("image_trainable_blocks"\s*:\s*)\d+': r'\g<1>4',
    r'("co_attn_layers"\s*:\s*)\d+': r'\g<1>3',
    r'("co_attn_heads"\s*:\s*)\d+': r'\g<1>8',
    r'("co_attn_dropout"\s*:\s*)[\d\.e-]+': r'\g<1>0.10',
    r'("vit_dropout"\s*:\s*)[\d\.e-]+': r'\g<1>0.10',
    r'("image_dropout"\s*:\s*)[\d\.e-]+': r'\g<1>0.10',
    r'("text_dropout"\s*:\s*)[\d\.e-]+': r'\g<1>0.10',
    r'("train_ratio"\s*:\s*)[\d\.e-]+': r'\g<1>0.80',
    r'("val_ratio"\s*:\s*)[\d\.e-]+': r'\g<1>0.10',
    r'("test_ratio"\s*:\s*)[\d\.e-]+': r'\g<1>0.10',
    r'("seed"\s*:\s*)\d+': r'\g<1>42',
    r'("grad_accum_steps"\s*:\s*)\d+': r'\g<1>2',
    r'("warmup_ratio"\s*:\s*)[\d\.e-]+': r'\g<1>0.10',
    r'("label_smoothing"\s*:\s*)[\d\.e-]+': r'\g<1>0.05',
    r'("use_amp"\s*:\s*)(True|False)': r'\g<1>True',
    r'("use_ema"\s*:\s*)(True|False)': r'\g<1>True',
    r'("use_rdrop"\s*:\s*)(True|False)': r'\g<1>True',
    r'("ema_decay"\s*:\s*)[\d\.e-]+': r'\g<1>0.999',
    r'("rdrop_alpha"\s*:\s*)[\d\.e-]+': r'\g<1>1.0',
}

for fname in files:
    path = os.path.join(r"e:\Coding\Project\Disertasi", fname)
    if not os.path.exists(path):
        print(f"Skipping {fname}, does not exist.")
        continue

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        
        source = cell.get('source', [])
        source_text = ''.join(source)
        
        # Determine if it's the config cell
        if "CONFIG" in source_text and "{" in source_text:
            orig_text = source_text
            for pat, repl in replacements.items():
                source_text = re.sub(pat, repl, source_text)
            
            if source_text != orig_text:
                # restore back to list of strings
                
                # To maintain exactly how it was, we split by lines and append \n
                lines = source_text.split('\n')
                new_source = []
                for i, line in enumerate(lines):
                    if i < len(lines) - 1:
                        new_source.append(line + '\n')
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                modified = True
                print(f"Modified config in {fname}")
                
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Saved {fname}")
        
print("Done.")
