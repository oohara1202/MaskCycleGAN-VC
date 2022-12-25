# メモ

使ったコマンドを記録する．

モデルについて，

- デフォルトの[公式MelGAN](https://github.com/descriptinc/melgan-neurips)で学習したもの（方式1）と，
- STUDIES-Teacherで学習された[MelGAN or HiFiGAN](https://github.com/descriptinc/melgan-neurips)（方式2）

以上2種類を検討する．
これらは，メルスペクトログラムの算出パラメータが異なる．

## 方式1

1. 話者（とGPU）を指定

    ```bash
    spk_A=FStudent
    spk_A=MStudent
    export CUDA_VISIBLE_DEVICES="0"
    ```

2. Data Preprocessing

    ```bash
    python data_preprocessing/preprocess_studies.py \
        --data_directory STUDIES_22k_reconst_without_angry/tr_no_dev \
        --preprocessed_data_directory STUDIES_22k_reconst_without_angry_preped/tr_no_dev \
        --speaker_ids Teacher FStudent MStudent

    python data_preprocessing/preprocess_studies.py \
        --data_directory STUDIES_22k_reconst_without_angry/eval1 \
        --preprocessed_data_directory STUDIES_22k_reconst_without_angry_preped/eval1 \
        --speaker_ids Teacher FStudent MStudent

    python data_preprocessing/preprocess_studies.py \
        --data_directory STUDIES_22k_reconst_without_angry/dev \
        --preprocessed_data_directory STUDIES_22k_reconst_without_angry_preped/dev \
        --speaker_ids Teacher FStudent MStudent
    ```

3. Training

    ```bash
    python -W ignore::UserWarning -m mask_cyclegan_vc.train_studies \
        --name ${spk_A}_Teacher \
        --seed 0 \
        --save_dir results/ \
        --preprocessed_data_dir STUDIES_22k_reconst_without_angry_preped/tr_no_dev \
        --speaker_A_id ${spk_A} \
        --speaker_B_id Teacher \
        --epochs_per_save 100 \
        --epochs_per_plot 10 \
        --num_epochs 1000 \
        --batch_size 1 \
        --generator_lr 2e-4 \
        --discriminator_lr 1e-4 \
        --decay_after 1e4 \
        --sample_rate 22050 \
        --num_frames 64 \
        --max_mask_len 25 \
        --gpu_ids 0
    ```

4. Testing

    ```bash
    python -W ignore::UserWarning -m mask_cyclegan_vc.test_studies \
        --name ${spk_A}_Teacher \
        --save_dir results/ \
        --preprocessed_data_dir STUDIES_22k_reconst_without_angry_preped/eval1 \
        --gpu_ids 0 \
        --speaker_A_id ${spk_A} \
        --speaker_B_id Teacher \
        --ckpt_dir results/${spk_A}_Teacher/ckpts \
        --load_epoch 1000 \
        --model_name generator_A2B
    ```

## 方式2

1. 話者（とGPU）を指定

    ```bash
    spk_A=FStudent
    spk_A=MStudent
    export CUDA_VISIBLE_DEVICES="0"
    ```

2. Data Preprocessing

    ```bash
    python data_preprocessing/preprocess_studies_pwgan.py \
        --data_directory STUDIES_22k_reconst_without_angry/tr_no_dev \
        --preprocessed_data_directory STUDIES_22k_reconst_without_angry_preped_pwgan/tr_no_dev \
        --speaker_ids Teacher FStudent MStudent

    python data_preprocessing/preprocess_studies_pwgan.py \
        --data_directory STUDIES_22k_reconst_without_angry/eval1 \
        --preprocessed_data_directory STUDIES_22k_reconst_without_angry_preped_pwgan/eval1 \
        --speaker_ids Teacher FStudent MStudent

    python data_preprocessing/preprocess_studies_pwgan.py \
        --data_directory STUDIES_22k_reconst_without_angry/dev \
        --preprocessed_data_directory STUDIES_22k_reconst_without_angry_preped_pwgan/dev \
        --speaker_ids Teacher FStudent MStudent
    ```

3. Training

    ```bash
    python -W ignore::UserWarning -m mask_cyclegan_vc.train_studies_pwgan \
        --name ${spk_A}_Teacher_pwgan \
        --seed 0 \
        --save_dir results/ \
        --preprocessed_data_dir STUDIES_22k_reconst_without_angry_preped_pwgan/tr_no_dev \
        --speaker_A_id ${spk_A} \
        --speaker_B_id Teacher \
        --epochs_per_save 100 \
        --epochs_per_plot 10 \
        --num_epochs 1000 \
        --batch_size 1 \
        --generator_lr 2e-4 \
        --discriminator_lr 1e-4 \
        --decay_after 1e4 \
        --sample_rate 22050 \
        --num_frames 64 \
        --max_mask_len 25 \
        --gpu_ids 0
    ```

4. Testing

    ```bash
    python -W ignore::UserWarning -m mask_cyclegan_vc.test_studies_pwgan \
        --name ${spk_A}_Teacher_pwgan \
        --save_dir results/ \
        --preprocessed_data_dir STUDIES_22k_reconst_without_angry_preped_pwgan/eval1 \
        --gpu_ids 0 \
        --speaker_A_id ${spk_A} \
        --speaker_B_id Teacher \
        --ckpt_dir results/${spk_A}_Teacher_pwgan/ckpts \
        --load_epoch 1000 \
        --model_name generator_A2B
    ```
