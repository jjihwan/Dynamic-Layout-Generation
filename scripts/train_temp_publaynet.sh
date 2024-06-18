python3 train_temp.py \
    --device cuda \
    --dataset publaynet \
    --lr 1e-6 \
    --n_save_epoch 10 \
    --num_frame 4 \
    --project_name "LACE-temporal" \
    --experiment_name "publaynet-only-dloss-train-all" \
    --no-freeze_original_model \
    --save_dir "plot_train" \
    --wandb