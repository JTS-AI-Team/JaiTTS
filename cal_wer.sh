set -euxo pipefail

meta_lst=$1
output_dir=$2
lang=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

workdir=$(cd $(dirname $0); cd ../; pwd)

python3 get_wav_res_ref_text.py "$meta_lst" "$output_dir" "$wav_wav_text" "$lang"

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
    sub_score_file="$out_dir/$chunk_name.wer.out"
    CUDA_VISIBLE_DEVICES=$rank python3 run_wer.py "$chunk_file" "$sub_score_file" "$lang" &
    job_idx=$((job_idx + 1))
done
wait

rm -f "$wav_wav_text"
rm -f "$out_dir/merge.out"

cat "$out_dir"/thread-*.wer.out >> "$out_dir/merge.out"
python3 average_wer.py "$out_dir/merge.out" "$score_file" "$lang"
