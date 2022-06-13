import glob
import os
import string
import sys
import time
from urllib.parse import MAX_CACHE_SIZE
from urllib.response import addinfo
from matplotlib.animation import FuncAnimation


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# from agents.navigation.global_route_planner import globalrouteplanner

# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import carla
from collections import deque
import numpy as np

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
actor_list = []
W, H = 800, 600 
cnt = 0

def get_roi(img):
    """Retrieve the region of interests."""
    # create a zero array
    stencil = np.zeros_like(img[:,:,0])

    # specify coordinates of the polygon
    polygon = np.array([[100,600], [300,420], [500,420], [700,600]])


    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)

    img = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)

def kmeanLine(lines):
    """Find 2 cluster of Hough transform lines"""
    _, _, means = cv2.kmeans(data=np.asarray(lines, dtype =np.float32), K=2, bestLabels=None,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, 
    flags=cv2.KMEANS_PP_CENTERS)
    return means

# def isEqual(a,b):
#     pass

def getAngle(lines, dmy): # lines must contain only left and right lines
#   if (isEqual(lines[0], lines[1])):
#       return 0
  p1 = [0,0]
  p2 = [0,0]
  b = [0,0]
  slope =[0,0]

  for i in range(len(lines)):
    x1, y1, x2, y2 = lines[i]
    slope[i] = (y2 - y1) / (x2 - x1)
    b[i]= y2 - slope[i] * x2
    p1[i] = np.float32((H - b[i])/slope[i])
    p2[i] = np.float32((0-b[i])/slope[i])
  
  c = (p1[0] + p1[1]) /2  # center bottom
  x_vp = (b[0] - b[1])/(slope[1] - slope[0]) # vanish point
  y_vp = slope[1] * x_vp + b[1]

  # left and right lane
  cv2.line(dmy,  (int(p1[0]), H),  (int(x_vp), int(y_vp)),  (0, 0, 255),  3 )
  cv2.line(dmy,  (int(p1[1]), H),  (int(x_vp), int(y_vp)),  (0, 255, 0),  3 )

  # center lane
  cv2.line(dmy,  (int(x_vp), int(y_vp)),  (int(c), H),  (255, 0, 0),  3 )
  
  # Target point
  x_tp = (x_vp + c) / 2
  y_tp = (y_vp + H) / 2
  cte = W/2 - x_tp
  cv2.circle(dmy, (int(x_tp), int(y_tp)), radius=5, color=(0, 0, 255), thickness=-1)
  
  slope = (H-y_vp)/ (c - x_vp)
  angle = 180 / math.pi * math.atan2(H-y_vp, x_vp - c) 
  return angle, cte

max_steer = 0.5
past_steering = 0
def action(image,cnt):
    global past_steering
    angle, cte = predict_steering(image, cnt)
    # veh_steering = vehicle.get_control().steer
    # current_steering = pid(angle, veh_steering, **args_lateral_dict)
    current_steering = pid(0, cte, **args_lateral_dict)
    if current_steering > past_steering + 0.1:
            current_steering = past_steering + 0.1
    elif current_steering < past_steering - 0.1:
        current_steering = past_steering - 0.1

    if current_steering >= 0:
        steering = min(max_steer, current_steering)
    else:
        steering = max(-max_steer, current_steering)

    current_steering = steering
    past_steering = steering

    vehicle.apply_control(carla.VehicleControl(throttle=0.3,steer = float(current_steering)))
    print("cte: ", cte)
    # print("Steering angle: ", angle)
    print('pid steering:', current_steering)

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((H, W, 4)) # RGBA
    i3 = i2[:, :, :3]  # RGBA -> RGB
    # crop_img = i3[y:y+h, :]
    action(i3, image.frame)

error_buffer =deque(maxlen=10) 
def pid(target, current, dt = 0.03, K_P = 1.0, K_I = 0.0, K_D = 0.0):
    error = target - current
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
    'K_P': 0.005,
    'K_D': 0.0,
    'K_I': 0.0

    ,'dt': 1.0 / 10.0
}

args_long_dict = {
    'K_P': 1,
    'K_D': 0.0,
    'K_I': 0.75
    ,'dt': 1.0 / 10.0
}

def predict_steering(img, cnt):
    """img = col_images[idx][:,:,2]"""
    img_gray = img[:,:,1]
    stencil = np.zeros_like(img_gray)

    # specify coordinates of the polygon
    polygon = np.array([[0,600], [0,420], [800,420], [800,600]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)

    # apply frame mask
    masked = cv2.bitwise_and(img_gray, img_gray, mask=stencil)

    # WHITE boundary
    # _, thresh1 = cv2.threshold(masked, 230, 255, cv2.THRESH_BINARY)

    def binary_threshold(img, thresh):
        return np.array(np.where(img <= thresh, 0, 255), dtype="uint8")

    img_inv = cv2.bitwise_not(img_gray)
    img_inv = cv2.bitwise_and(img_inv, img_inv, mask=stencil)

    # ret, thresh2 = cv2.threshold(imagem, 90, 255, cv2.THRESH_BINARY)
    thresh2 = binary_threshold(img_inv, thresh=110)

    # lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold = 10, minLineLength = 30, maxLineGap=200)
    lines = cv2.HoughLinesP(thresh2, 1, np.pi/180, threshold = 20, minLineLength = 50, maxLineGap=200)
    if lines == []:
        return 0,0

    lines = lines.reshape(-1, 4)
    dmy = img_gray.copy()

    # Plot detected lines
    try:
        means = kmeanLine(lines)
        angle, cte = getAngle(means, dmy) 
        path = '_out/' +str(cnt) + '.jpg'
        cv2.imwrite(path, dmy)
        path = '_out2/' + str(cnt) +  '.jpg'
        cv2.imwrite(path, img)
        return (angle - 90) / 180, cte

    except TypeError: 
        return 0, 0


try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) # seconds

    # world = client.load_world('Town04_Opt')
    # world.unload_map_layer(carla.MapLayer.All)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    transform = world.get_map().get_spawn_points()[2]

    client.start_recorder("./recording_naive_lane_detection2.log")
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
    camera_transform = carla.Transform(carla.Location(x=1.5, z=12))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    actor_list.append(camera)
    print('created %s' % camera.type_id)

    # cnt = 0
    # for transform in world.get_map().get_spawn_points():
    #     # print(type(transform.location))
    #     world.debug.draw_string(location = transform.location, text = str(cnt), life_time=120.0)
    #     # world.debug.draw_point(transform.location,color=carla.Color(r=255, g=0, b=0),size=1.6 ,life_time=120.0)
    #     cnt+=1
    time.sleep(5)
    cc = carla.ColorConverter.Raw
    camera.listen(lambda image: process_img(image))
    
    time.sleep(120)
    client.stop_recorder()
finally:

    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print('done.')