export CUDA_VISIBLE_DEVICES=4
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name eyetrack \
    --train_data_paths C:\Users\daanb\PycharmProjects\EyetrackPredRNN\Data \
    --valid_data_paths C:\Users\daanb\PycharmProjects\EyetrackPredRNN\Data \
    --save_dir checkpoints/eye_predrnn_v2 \
    --gen_frm_dir results/eye_predrnn_v2 \
    --model_name image_cond_predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 224 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 1 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 5000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2000 \
    --lr 0.0001 \
    --batch_size 4 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --num_action_ch 3 \
    --conv_on_input 1 \
#    --pretrained_model ./checkpoints/kth_predrnn_v2/kth_model.ckpt