import sys, os
from tqdm import tqdm

metalst = sys.argv[1]
wav_dir = sys.argv[2]
wav_res_ref_text = sys.argv[3]
lang = sys.argv[4] if len(sys.argv) > 4 else None

f = open(metalst)
lines = f.readlines()
f.close()

f_w = open(wav_res_ref_text, 'w')

for line in tqdm(lines):
    parts = line.strip().split('|')

    if len(parts) == 5:
        if lang == "th":
            utt, prompt_text, prompt_wav, infer_text, norm_infer_text = line.strip().split('|')
        else:
            utt, prompt_text, prompt_wav, infer_text, infer_wav = line.strip().split('|')
    elif len(parts) == 4:
        utt, prompt_text, prompt_wav, infer_text = line.strip().split('|')
    elif len(parts) == 2:
        utt, infer_text = line.strip().split('|')
    elif len(parts) == 3:
        utt, infer_text, prompt_wav = line.strip().split('|')
        if utt.endswith(".wav"): 
            utt = utt[:-4]
    else:
        raise NotImplementedError(f"Malformed line: {line.strip()}")
    if not os.path.exists(os.path.join(wav_dir, utt + ".wav")):
        continue

    # tmp
    #prompt_wav = infer_wav

    if not os.path.isabs(prompt_wav):
        prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

    # if not os.path.isabs(infer_wav):
    #     infer_wav = os.path.join(os.path.dirname(metalst), infer_wav)

    if len(parts) == 2:
        out_line = '|'.join([os.path.join(wav_dir, utt + ".wav"), infer_text])
    else:
        if lang == "th":
            out_line = '|'.join([os.path.join(wav_dir, utt + ".wav"), prompt_wav, norm_infer_text])
        else:
            out_line = '|'.join([os.path.join(wav_dir, utt + ".wav"), prompt_wav, infer_text])
    f_w.write(out_line + '\n')
f_w.close()
