cd ~/carla/
./CarlaUE4.sh /Game/Carla/Maps/Town04 -windowed -quality-level=Low -opengl

conda activate carla
cd PythonAPI/pilotnet
python minimal.py
python naive_lane_detection.py