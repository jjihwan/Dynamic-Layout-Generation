python inference_temp.py \
        --dataset publaynet \
        --experiment c \
        --pretrained_model_path model_trained/publaynet_2024-06-18T11-03-00/epoch=000030.pt \
        --device cuda \
        --dim_transformer 1024 \
        --nhead 16 \
        --batch_size 2048 \
        --beautify \
        --plot