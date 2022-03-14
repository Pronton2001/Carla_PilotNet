import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from collections import deque
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import model as md

# Remove warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load pretrained model
model = md.getPilotNetModel()
model.load_weights('model/model-weights.h5')

# Constants
actor_list = []
# W, H = 320, 180
W, H = 200, 188

# TODO: predict steering angle using trained model
def predict_steering(input_img, pretrained = False):
    if pretrained:
        input_img = resize(input_img, (66, 200))
    model = load_model('model/baseline3_new.h5') 
    input_img = expand_dims(input_img, 0)  # Create batch axis
    steering_pred= model.predict(input_img)[0][0]
    return steering_pred

# TODO: call_back function for RGB image
def action(image):
    angle = predict_steering(image)
    current_steering = vehicle.get_control().steer
    new_steering = pid(angle, current_steering, **args_lateral_dict)
    vehicle.apply_control(carla.VehicleControl(throttle=0.4,steer = float(new_steering)))
    print("Steering angle: ", angle)

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((H, W, 4)) # RGBA
    i3 = i2[:, :, :3]  # RGBA -> RGB
    # cv2.imshow('', i3)
    # cv2.waitKey(1)
    # img_norm = i3/255.0
    y = 75
    h = 88
    crop_img = i3[y:y+h, :]
    action(crop_img)
    # path = '_out6/'  + str(image.frame) + '.jpg'
    # cv2.imwrite(path, crop_img)

def clip_throttle(throttle, curr_speed, target_speed = 8.3):
    return np.clip(
        throttle - 0.01 * (curr_speed-target_speed),
        0.4,
        0.9
    )

error_buffer =deque(maxlen=10) 
def pid(target_steering, current_steering, dt = 0.03, K_P = 1.0, K_I = 0.0, K_D = 0.0):
    error = target_steering - current_steering
    error_buffer.append(error)
    if len(error_buffer) >= 2:
        _de = (error_buffer[-1] - error_buffer[-2]) / dt
        _ie = sum(error_buffer) * dt
    else:
        _de = 0.0
        _ie = 0.0
    _pid = K_P * error + K_I * _ie + K_D * _de
    return np.clip(_pid, -1, 1)

args_lateral_dict = {
    'K_P': 1.95,
    'K_D': 0.2,
    'K_I': 0.07

    ,'dt': 1.0 / 10.0
}

args_long_dict = {
    'K_P': 1,
    'K_D': 0.0,
    'K_I': 0.75
    ,'dt': 1.0 / 10.0
}
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) # seconds

    world = client.load_world('Town01_Opt')
    # world = client.get_world()
    world.unload_map_layer(carla.MapLayer.All)

    blueprint_library = world.get_blueprint_library()

    # bp = random.choice(blueprint_library.filter('vehicle'))
    bp = blueprint_library.filter('model3')[0]
    transform = world.get_map().get_spawn_points()[80]

    client.start_recorder("./recording_baseline3_Town1_pid.log")
    vehicle = world.spawn_actor(bp, transform)
    # vehicle.set_autopilot(True)

    actor_list.append(vehicle)
    print('created %s' % vehicle.type_id)

    # Find the blueprint of the sensor.
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    camera_bp.set_attribute('image_size_x', str(W))
    camera_bp.set_attribute('image_size_y', str(H))
    camera_bp.set_attribute('fov', '40')

    # Set the time in seconds between sensor captures
    camera_bp.set_attribute('sensor_tick', '1.0')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    actor_list.append(camera)
    print('created %s' % camera.type_id)

    cc = carla.ColorConverter.Raw
    camera.listen(lambda image: process_img(image))
    # while True:
    #     # vehicle.apply_control(carla.VehicleControl(throttle=0.9, steer = 1))
    #     # vehicle.enable_constant_velocity(carla.Vector3D(8.3,0,0)) 
    #     time.sleep(0.01)
    time.sleep(30)
    client.stop_recorder()
finally:

    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print('done.')
