actor=M003_neutral
ref=M009_happy
python postprocessing/vis.py \
    --celeb celebrities/test/${actor} \
    --reference celebrities/reference/${ref}/images/000004.png \
    --exp_name reference_on_${ref}_4 \
    --out_path celebrities/out_videos/vis/${actor}_reference_on_${ref}_base.mp4