# actors_train='W028 W024 W025 W029 W026 M003 W009 W040 M013 M024 W015 M033 M041 M022 W011 M005 M019 W021 M011 W018 M042 M032 M009 W036 M040 M012 M039 M025 W035 M026 W038 M028 M030 W033 M031 W023 M029 W014 M007 M035'
# actors_val='W019 W016 M023 M034 M037 W037 M027'

actors_train='M003'
actors_val='M003'

python manipulator/train2.py \
    --train_root MEAD-sim \
    --selected_actors $actors_train \
    --selected_actors_ref M009 \
    --selected_actors_val $actors_val \
    --selected_emotions neutral happy \
    --selected_emotions_ref happy \
    --class_names neutral happy \
    --dist_file exp/similarity/dists/dists.pkl \
    --checkpoints_dir exp/manipulator/single_simi \
    --niter 20 --niter_decay 20 \
    --paired --lambda_paired 10

# python train_union.py \
#     --train_root MEAD-sim \
#     --selected_actors $actors_train \
#     --selected_actors_ref M009 \
#     --selected_actors_val $actors_val \
#     --selected_emotions neutral \
#     --selected_emotions_ref happy \
#     --class_names neutral happy \
#     --dist_file exp/similarity/dists/dists.pkl \
#     --checkpoints_dir exp/union/single_simi4 \
#     --niter 5 --lambda_feat 10 \
#     --lambda_paired 1 --lambda_paired_renderer 1 \
#     --manipulator_pretrain_weight exp/manipulator/single_simi/20_nets.pth \
#     --rendererG_pretrain_weight exp/renderer/M003_happy/20_net_G.pth \
#     --rendererD_pretrain_weight exp/renderer/M003_happy/20_net_D.pth


# python renderer/train.py \
#     --celeb celebrities/train/W037_happy \
#     --checkpoints_dir exp/renderer/W037_happy \
#     --niter 20 --batch_size 4
