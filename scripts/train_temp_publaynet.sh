python3 train_temp.py \
    --device cuda \
    --dataset publaynet \
    --lr 1e-4 \
    --batch_size 512 \
    --n_save_epoch 40 \
    --num_frame 4 \
    --project_name "LACE-temporal" \
    --align_weight 1 \
    --experiment_name "publaynet-standard-tf-static" \
    --freeze_original_model \
    --save_dir "plot_train" \
    --aug_type "static" \
    --wandb
    # --resume_from_ckpt \
    # --resume_ckpt_path "/home/youmong1204/code/Dynamic-Layout-Generation/model_trained/publaynet-standard-tf-static-wo-laloss_2024-06-25T04-28-10/epoch=000240.pt" \
    # --resume_id r8ugs1va \