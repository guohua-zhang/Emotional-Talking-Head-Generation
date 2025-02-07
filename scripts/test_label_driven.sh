for object in FrancesMcDormand JamieFoxx JuliaRoberts MatthewMcConaughey Pacino Tarantino
do
    ./preprocess.sh ./celebrities/test/${object}/ test
    for emotion in angry disgusted fear happy sad surprised
    do
        python manipulator/test.py --celeb ./celebrities/test/${object}/ --checkpoints_dir ./checkpoints/manipulator/ --trg_emotions ${emotion} --exp_name ./${emotion}

        ./postprocess.sh ./celebrities/test/${object}/ ./${emotion} ./checkpoints/renderer/${object}/

        python postprocessing/images2video.py --imgs_path ./celebrities/test/${object}/${emotion}/full_frames --out_path ./celebrities/out_videos/${object}/${emotion}_${object}.mp4 --audio ./celebrities/test/${object}/videos/${object}_t.mp4
    done
done
