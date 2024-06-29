python inference_temp.py \
        --dataset publaynet \
        --experiment c \
        --pretrained_model_path /home/kjh26720/code/LACE/model_trained/publaynet-standard-tf-static-wo-laloss_2024-06-28T11-06-42/epoch=000360.pt \
        --device cuda \
        --dim_transformer 1024 \
        --nhead 16 \
        --batch_size 32 \
        --beautify \
        --plot