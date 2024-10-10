python full_main.py \
    --batch_size 4 \
    --version cfd \
    --dlv3   \
    --name cfd_training \
    --gpu_id 0 \
    --test \
    --imsize 256 \
    --model_name 500_cfd_dlv3_500_500_0.0001_0.9_0.999.pth \
    --csv_path './dice_scores/dice_score_test_cfd.csv'


# python full_main.py  \
#     --batch_size 4 \
#     --version celeb \
#     --dlv3   \
#     --name celeb_training \
#     --gpu_id 0 \
#     --test \
#     --imsize 256 \
#     --model_name '500_celeb_dlv3_500_500_0.0001_0.9_0.999.pth' \
#     --csv_path './dice_scores/dice_score_test_celeb.csv'


python full_main.py \
    --batch_size 4 \
    --version combined \
    --dlv3   \
    --name combined_training \
    --gpu_id 0 \
    --test \
    --imsize 256 \
    --model_name '500_combined_dlv3_500_500_0.0001_0.9_0.999.pth' \
    --csv_path './dice_scores/dice_score_test_combined.csv'

    