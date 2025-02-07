actor=${1:-M003}
reference=${2:-M009_happy}
manipulator_ckpt=${3:-single_simi}
renderer_ckpt=${4:-M003_happy}
manipulator_epoch=${5:-20}
renderer_epoch=${6:-20}
tag=${7:-simi0}

rm -rf MEAD-sim/${actor}/render/neutral/level_1/reference_on_${reference}_${tag}

python manipulator/test.py --celeb MEAD-sim/${actor}/render/neutral/level_1 --checkpoints_dir exp/manipulator/${manipulator_ckpt}/ --which_epoch ${manipulator_epoch} --ref_dirs celebrities/reference/${reference} --exp_name reference_on_${reference}_${tag}

./scripts/postprocess.sh MEAD-sim/${actor}/render/neutral/level_1 reference_on_${reference}_${tag} exp/renderer/${renderer_ckpt}/ ${renderer_epoch}

#mkdir -p celebrities/out_videos/${actor}

#python postprocessing/images2video.py --imgs_path MEAD-sim/${actor}/render/neutral/level_1/reference_on_${reference}_${tag}/full_frames --out_path celebrities/out_videos/${actor}/${actor}_reference_on_${reference}_${tag}
