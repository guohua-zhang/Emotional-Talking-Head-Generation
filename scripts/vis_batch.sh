for ref in M009 W016
do
  for actor in M003 M027 W019 W037
  do
  python postprocessing/vis.py \
      --celeb celebrities/test/${actor}_neutral \
      --reference celebrities/reference/${ref}_happy/images/000004.png \
      --exp_name reference_on_${ref}_happy \
      --out_path celebrities/out_videos/vis/${actor}_neutral_reference_on_${ref}_happy.mp4
  done
done