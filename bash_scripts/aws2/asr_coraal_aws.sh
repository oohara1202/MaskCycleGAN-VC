python -W ignore::UserWarning -m asr.main \
    --name debug \
    --data_dir ~/data \
    --save_dir ~/results \
    --coraal \
    --voc \
    --converted \
    --num_epochs 100 \
    --batch_size 1 \
    --gpu_ids 0 \
    --num_workers 1 \
    --n_feats 80 \
    --epochs_per_save 2 \
    --pretrained_ckpt_path ~/results/librispeech/ckpts/best.pth.tar \
    --converted_source_ids 0 \