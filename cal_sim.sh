set -euxo pipefail

meta_lst=$1
output_dir=$2
checkpoint_path=$3
lang=${4:-th}

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.sim

python3 get_wav_res_ref_text.py "$meta_lst" "$output_dir" "$output_dir/wav_res_ref_text" "$lang"

workdir=$(cd $(dirname $0); pwd)
verify_script="$workdir/thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py"
average_script="$workdir/thirdparty/UniSpeech/downstreams/speaker_verification/average.py"

timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_$timestamp/
mkdir -p "$thread_dir"
num_job=${ARNOLD_WORKER_GPU:-1}
if [ "$num_job" -lt 1 ]; then
    num_job=1
fi
num=$(wc -l < "$wav_wav_text")
num_per_thread=$((num / num_job + 1))
split -l "$num_per_thread" -a 4 -d "$wav_wav_text" "$thread_dir/thread-"
out_dir=/tmp/thread_metas_$timestamp/results/
mkdir -p "$out_dir"

job_idx=0
for chunk_file in "$thread_dir"/thread-*; do
    [ -f "$chunk_file" ] || continue

    rank=$((job_idx % num_job))
    chunk_name=$(basename "$chunk_file")
    sub_score_file="$out_dir/$chunk_name.sim.out"
    PYTHONPATH="$workdir" CUDA_VISIBLE_DEVICES=$rank python3 "$verify_script" "$chunk_file" \
        --model_name wavlm_large \
        --checkpoint "$checkpoint_path" \
        --scores "$sub_score_file" \
        --wav1_start_sr 0 \
        --wav2_start_sr 0 \
        --wav1_end_sr -1 \
        --wav2_end_sr -1 &
    job_idx=$((job_idx + 1))
done
wait

rm -f "$wav_wav_text"
rm -f "$out_dir/merge.out"

grep -h -v "avg score" "$out_dir"/thread-*.sim.out >> "$out_dir/merge.out"
python3 "$average_script" "$out_dir/merge.out" "$score_file"
