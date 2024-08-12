date=`date +%s`
date=1723168378
mkdir -p exp/$date/cache/s1
mkdir -p exp/$date/cache/s2
mkdir -p exp/$date/cache/retrieval
mkdir -p exp/$date/prediction/s1
mkdir -p exp/$date/prediction/s2
mkdir -p exp/$date/prediction/valid
mkdir -p exp/$date/output/s1
mkdir -p exp/$date/output/s2
mkdir -p exp/$date/output/retrieval
mkdir -p summary/s1
mkdir -p summary/s1

batch_size=12
max_source_length=512
max_target_length=100
output_dir=./exp/$date/output
res_dir=./exp/$date/prediction
cache_path=./exp/$date/cache
task=java
learning_rate=5e-5
devices=0
epoch=10
data_num=-1

# echo ">>>>>>>>>>>>>>>>>>>> s1"
# CUDA_VISIBLE_DEVICES=$devices \
#   python run_gen.py \
#   --do_train --do_eval \
#   --task summarize --sub_task $task --model_type codet5 --data_num -1 \
#   --data_type s1 --num_train_epochs 5 --warmup_steps 1000 --learning_rate $learning_rate \
#   --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --data_dir /mnt/ssd2/rambo/data/come_data \
#   --cache_path $cache_path/s1 --output_dir $output_dir/s1 --summary_dir ./summary/s1 \
#   --save_last_checkpoints --always_save_model --res_dir $res_dir/s1 \
#   --train_batch_size $batch_size --eval_batch_size $batch_size --max_source_length $max_source_length --max_target_length $max_target_length

# cp ./config.json ./exp/$date/output/s1/checkpoint-best-ppl/

echo ">>>>>>>>>>>>>>>>>>>> s2 without s1"
CUDA_VISIBLE_DEVICES=$devices \
  python run_gen.py \
  --do_train --do_eval --do_eval_bleu --do_test --split_tag test\
  --task summarize --sub_task $task --model_type codet5 --data_num $data_num \
  --data_type s2 --num_train_epochs $epoch --warmup_steps 1000 --learning_rate $learning_rate \
  --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --data_dir /mnt/ssd2/rambo/data/come_data \
  --cache_path $cache_path/s2 --output_dir $output_dir/s2 --summary_dir ./summary/s2 \
  --save_last_checkpoints --always_save_model --res_dir $res_dir/s2 \
  --train_batch_size $batch_size --eval_batch_size $batch_size --max_source_length $max_source_length --max_target_length $max_target_length

cp ./config.json ./exp/$date/output/s2/checkpoint-best-bleu/
rm -r exp/$date/cache/s2
mkdir exp/$date/cache/s2

echo ">>>>>>>>>>>>>>>>>>>> run valid test"
CUDA_VISIBLE_DEVICES=$devices \
  python run_gen.py \
  --do_test --test_file /mnt/ssd2/rambo/data/come_data/summarize/$task/valid.jsonl --split_tag valid \
  --task summarize --sub_task $task --model_type codet5 --data_num $data_num \
  --data_type s2 --num_train_epochs $epoch --warmup_steps 1000 --learning_rate $learning_rate \
  --tokenizer_name=Salesforce/codet5-base --model_name_or_path=./exp/$date/output/s2/checkpoint-best-bleu --data_dir /mnt/ssd2/rambo/data/come_data \
  --cache_path $cache_path/s2 --output_dir $output_dir/s2 --summary_dir ./summary/s2 \
  --save_last_checkpoints --always_save_model --res_dir $res_dir/valid \
  --train_batch_size $batch_size --eval_batch_size $batch_size --max_source_length $max_source_length --max_target_length $max_target_length

echo ">>>>>>>>>>>>>>>>>>>> run retrieve for valid"
CUDA_VISIBLE_DEVICES=$devices \
  python -W ignore run_gen.py \
  --do_retrieval --retrieval_file valid \
  --task summarize --sub_task $task --model_type codet5 --data_num $data_num \
  --tokenizer_name=Salesforce/codet5-base --model_name_or_path=./exp/$date/output/s2/checkpoint-best-bleu/ --data_dir /mnt/ssd2/rambo/data/come_data \
  --data_type s2 --output_dir $output_dir/retrieval \
  --cache_path $cache_path/retrieval --summary_dir ./summary/s1 --res_dir $res_dir/s1 \
  --train_batch_size 32 --eval_batch_size 32 --max_source_length $max_source_length

echo ">>>>>>>>>>>>>>>>>>>> run retrieve for test"
CUDA_VISIBLE_DEVICES=$devices \
  python -W ignore run_gen.py \
  --do_retrieval --retrieval_file test \
  --task summarize --sub_task $task --model_type codet5 --data_num $data_num \
  --tokenizer_name=Salesforce/codet5-base --model_name_or_path=./exp/$date/output/s2/checkpoint-best-bleu/ --data_dir /mnt/ssd2/rambo/data/come_data \
  --data_type s2 --output_dir $output_dir/retrieval \
  --cache_path $cache_path/retrieval --summary_dir ./summary/s1 --res_dir $res_dir/s1 \
  --train_batch_size 32 --eval_batch_size 32 --max_source_length $max_source_length

echo ">>>>>>>>>>>>>>>>>>>> run svm"
python -W ignore svm.py \
    -valid_retrieval_msg ./exp/$date/output/retrieval/valid.output \
    -valid_retrieval_bleu ./exp/$date/output/retrieval/valid.score \
    -valid_generate_msg ./exp/$date/prediction/valid/test_best-bleu.output \
    -valid_generate_score ./exp/$date/prediction/valid/test_best-bleu.score \
    -ground_truth ./exp/$date/prediction/valid/test_best-bleu.gold \
    -test_retrieval_msg ./exp/$date/output/retrieval/test.output \
    -test_retrieval_bleu ./exp/$date/output/retrieval/test.score \
    -test_generate_msg ./exp/$date/prediction/s2/test_best-bleu.output \
    -test_generate_score ./exp/$date/prediction/s2/test_best-bleu.score \
    -test_ground_truth ./exp/$date/prediction/s2/test_best-bleu.gold \
    -output  ./exp/$date/prediction/svm

echo ">>>>>>>>>>>>>>>>>>>> run evaluate"
echo "*************s2"
python3 evaluate.py --refs_filename  ./exp/$date/prediction/s2/test_best-bleu.gold --preds_filename ./exp/$date/prediction/s2/test_best-bleu.output
echo "*************retrieve"
python3 evaluate.py --refs_filename  ./exp/$date/prediction/s2/test_best-bleu.gold --preds_filename ./exp/$date/output/retrieval/test.output
echo "*************svm"
python3 evaluate.py --refs_filename  ./exp/$date/prediction/s2/test_best-bleu.gold --preds_filename ./exp/$date/prediction/svm

