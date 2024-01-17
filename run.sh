cd /main_frame/resize3
python3 run.py /images /frame_main/resize3
cd /mvs
mv depthmap /restore/resize2/mvs
mv normalmap /restore/resize2/mvs
cd /restore/resize2
python3 run.py /images /restore/resize2
cd mvs
mv depthmap /frame_main/resize2/mvs
mv normalmap /frame_main/resize2/mvs
cd /frame_main/resize2
python3 run.py /images /frame_main/resize2
cd mvs
mv depthmap /restore/resize1/mvs
mv normalmap /restore/resize1/mvs
cd /restore/resize1
python3 run.py /images /restore/resize1
cd mvs
mv depthmap /frame_main/resize1/mvs
mv normalmap /frame_main/resize1/mvs
cd /frame_main/resize1
python3 run.py /images /frame_main/resize1

