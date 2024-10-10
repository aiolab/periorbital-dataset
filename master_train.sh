python full_main.py --total_step 500 \
    --batch_size 16 \
    --version cfd \
    --lr .0001 \
    --dlv3   \
    --name cfd_training \
    --gpu_id 0 \
    --train \
    --imsize 256 
    

# python full_main.py --total_step 500 \
#     --batch_size 16 \
#     --version celeb \
#     --lr .0001 \
#     --dlv3   \
#     --name celeb_training \
#     --gpu_id 0 \
#     --train \
#     --imsize 256 
    

python full_main.py --total_step 500 \
    --batch_size 16 \
    --version combined \
    --lr .0001 \
    --dlv3   \
    --name combined_training \
    --gpu_id 0 \
    --train \
    --imsize 256 
    









