import sys, os, torch, string, warnings, scipy, zhconv, jiwer
import numpy as np
import soundfile as sf
from tqdm import tqdm
from zhon.hanzi import punctuation
from funasr import AutoModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
warnings.filterwarnings("ignore")

punctuation_all = punctuation + string.punctuation

def load_en_model(device):
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def load_zh_model():
    model = AutoModel(model="paraformer-zh")
    return model

def load_th_model(device):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_id = "typhoon-ai/typhoon-whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True, 
        token=os.environ.get("HF_TOKEN")
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=444,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            device=device)
    return pipe

def is_silence(path, threshold=2e-3):
    """Returns True if the audio max amplitude is below the threshold."""
    try:
        data, _ = sf.read(path)
        return np.max(np.abs(data)) < threshold
    except Exception:
        return True

def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'': continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    elif lang == "th":
        truth = truth.lower().replace(' ', '').replace('\'', '')
        hypo = hypo.lower().replace(' ', '').replace('\'', '')
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    else: 
        raise NotImplementedError

    measures = jiwer.process_words(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures.wer
    subs = measures.substitutions / len(ref_list)
    dele = measures.deletions / len(ref_list)
    inse = measures.insertions / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)

def run_asr(params, res_path, lang):
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if lang == "en":   
        processor, model = load_en_model(device)
    elif lang == "zh": 
        model = load_zh_model()
    elif lang == "th": 
        pipe = load_th_model(device)
    else:
        raise NotImplementedError(f"Unsupported language: {lang}")

    with open(res_path, "w", encoding="utf-8") as fout:
        for wav_res_path, text_ref in tqdm(params, desc=f"ASR ({lang})"):
            
            if not is_silence(wav_res_path):
                if lang == "en":
                    wav, sr = sf.read(wav_res_path)
                    if sr != 16000: wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
                    input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
                    input_features = input_features.to(device)
                    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
                    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                elif lang == "th":
                    duration = sf.info(wav_res_path).duration
                    if duration >= 30.0:
                        # Generated clips should stay short; longer audio is treated as a generator error.
                        transcription = ""
                    else:
                        wav, sr = sf.read(wav_res_path)
                        if sr != 16000: wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
                        transcription = pipe({"sampling_rate": 16000, "raw": wav}, generate_kwargs={"language": "thai"})["text"]
                        transcription = transcription.strip()

                        chk_text_ref, chk_transcription = text_ref, transcription
                        for x in punctuation_all:
                            chk_text_ref = chk_text_ref.replace(x, '')
                            chk_transcription = chk_transcription.replace(x, '')
                        truth_chars = len(chk_text_ref.replace(" ", ""))
                        hypo_chars = len(chk_transcription.replace(" ", ""))
                        ratio_threshold = 2.5  # If output is 2.5x longer than truth, it's a loop
                        if truth_chars > 0:
                            ratio = hypo_chars / truth_chars
                            if ratio > ratio_threshold: 
                                transcription = ""  
                        elif hypo_chars > 30:  # If ground truth is empty but the model hallucinates a long string
                            transcription = ""
                
                elif lang == "zh":
                    res = model.generate(input=wav_res_path, batch_size_s=300)
                    transcription = res[0]["text"]
                    transcription = zhconv.convert(transcription, 'zh-cn')

                else: 
                    transcription = ""
            else: 
                transcription = ""

            raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref, lang)
            fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
            fout.flush()

if __name__ == '__main__':
    wav_res_text_path = sys.argv[1]
    res_path = sys.argv[2]
    lang = sys.argv[3] # zh or en or th

    params =[]
    with open(wav_res_text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split('|')

            if len(parts) == 2:
                wav_res_path, text_ref = parts
            elif len(parts) == 3:
                wav_res_path, wav_ref_path, text_ref = parts
            elif len(parts) == 4: # for edit
                wav_res_path, _, text_ref, wav_ref_path = parts
            else: 
                raise NotImplementedError(f"Malformed line: {line}")

            if not os.path.exists(wav_res_path): 
                continue
            
            params.append((wav_res_path, text_ref))

    if not params:
        print("No valid lines to process. Exiting.")
        sys.exit(0)

    print(f"Processing {len(params)} files...")
    run_asr(params, res_path, lang)
    print("Inference completed successfully!")
