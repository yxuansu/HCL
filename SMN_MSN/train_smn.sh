CUDA_VISIBLE_DEVICES=5 python train.py\
    --train_context_path ../data/Douban/train_context.txt\
    --train_true_response_path ../data/Douban/train_true_response_id.txt\
    --response_index_path ../data/Douban/train_response_index.txt\
    --train_context_vec_file ../data/Douban/douban_train_context_vec.pkl\
    --all_response_vec_file ../data/Douban/douban_all_response_vec.pkl\
    --dev_path ../data/Douban/test.txt\
    --word2id_path ./embeddings/smn/word2id.txt\
    --model_type SMN\
    --embedding_path ./embeddings/smn/embedding.pkl\
    --batch_size 256\
    --loss_margin 0.3\
    --lr 3e-5\
    --gradient_accumulation_steps 1\
    --pretrain_total_steps 100000\
    --finetune_total_steps 100000\
    --warmup_steps 2000\
    --print_every 100\
    --eval_every 1000\
    --ckpt_path ./ckpt/smn/\
