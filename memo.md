# メモ

## Data Preprocessing

```bash
python data_preprocessing/preprocess_studies.py \
    --data_directory STUDIES_reconst/tr_no_dev \
    --preprocessed_data_directory STUDIES_reconst_preprocessed/tr_no_dev \
    --speaker_ids Teacher FStudent MStudent
```

```bash
python data_preprocessing/preprocess_studies.py \
    --data_directory STUDIES_reconst/eval1 \
    --preprocessed_data_directory STUDIES_reconst_preprocessed/eval1 \
    --speaker_ids Teacher FStudent MStudent
```

## Training

```bash
python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_FStudent_Teacher \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir STUDIES_reconst_preprocessed/tr_no_dev \
    --speaker_A_id FStudent \
    --speaker_B_id Teacher \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --batch_size 1 \
    --generator_lr 2e-4 \
    --discriminator_lr 1e-4 \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 1
```
