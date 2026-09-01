# CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
#   --metadata_path /root/workspace/train_metadata.json \
#   --model_dir /mnt/oss/data/Wan2.1-T2V-1.3B \
#   --raft_checkpoint /mnt/oss/data/raft/raft-sintel.pth \
#   --output_dir ./outputs/test_stage1


CUDA_VISIBLE_DEVICES=0 python train_stage3.py \
  --metadata_path /root/workspace/train_metadata.json \
  --model_dir /mnt/oss/data/Wan2.1-T2V-1.3B \
  --raft_checkpoint /mnt/oss/data/raft/raft-sintel.pth \
  --base_lora_checkpoint  /root/workspace/WanHDR/outputs/stage1/epoch-300.safetensors \
  --decoder_checkpoint /root/workspace/WanHDR/train/decoder/checkpoint-00016900.pt \
  --output_dir ./outputs/test_stage3
