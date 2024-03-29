CUDA_VISIBLE_DEVICES=1,2,3,4 python learn.py\
    --corpus_name douban\
    --train_context_path ../data/Douban/train_context.txt\
    --train_true_response_path ../data/Douban/train_true_response_id.txt\
    --response_index_path ../data/Douban/train_response_index.txt\
    --train_context_vec_file ../data/Douban/douban_train_context_vec.pkl\
    --all_response_vec_file ../data/Douban/douban_all_response_vec.pkl\
    --dev_path ../data/Douban/test.txt\
    --bert_path ./bert-base-chinese\
    --batch_size_per_gpu 1\
    --number_of_gpu 4\
    --gradient_accumulation_steps 2\
    --pretrain_total_steps 300000\
    --finetune_total_steps 100000\
    --print_every 100\
    --eval_every 1000\
    --ckpt_path ./ckpt/
