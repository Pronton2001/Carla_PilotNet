import glob
import os
import sys
import time
from xml.dom.expatbuilder import theDOMImplementation

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
import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
actor_list = []
W, H = 800, 600 

def read_image():
    """Read image from ego vehicle's camera."""
    pass

def get_roi(img):
    """Retrieve the region of interests."""
    # create a zero array
    stencil = np.zeros_like(img[:,:,0])

    # specify coordinates of the polygon
    polygon = np.array([[100,600], [200,400], [550,400], [700,600]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)

    img = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)

    # plot polygon
    plt.figure(figsize=(10,10))
    plt.imshow(stencil, cmap= "gray")
    plt.show()

# def threshold(img):
#     # img = col_images[idx]
#     # apply image thresholding
#     _, thresh1 = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
#     def binary_threshold(img, thresh):
#         return np.array(np.where(img <= thresh, 0, 255), dtype="uint8")

#     img_inv = cv2.bitwise_not(col_images[idx][:,:,1])
#     img = cv2.bitwise_and(img_inv, img_inv, mask=stencil)

#     # ret, thresh2 = cv2.threshold(imagem, 90, 255, cv2.THRESH_BINARY)
#     thresh2 = binary_threshold(img, thresh=80)

#     # gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

#     # plot image
#     plt.figure(figsize=(10,10))
#     plt.imshow(thresh, cmap="gray")
#     plt.show()

def kmeanLine(lines):
    """Find 2 cluster of Hough transform lines"""
    _, _, means = cv2.kmeans(data=np.asarray(lines, dtype =np.float32), K=2, bestLabels=None,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, 
    flags=cv2.KMEANS_PP_CENTERS)
    return means

def getAngle(lines, dmy): # lines must contain only left and right lines
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
  
  slope = (H-y_vp)/ (c - x_vp)
  angle = 180 / math.pi * math.atan2(H-y_vp, x_vp - c) 
  return angle


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
    # y = 75
    # h = 88
    # crop_img = i3[y:y+h, :]
    action(i3[:,:,2])

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

def predict_steering(img):
    """img = col_images[idx][:,:,2]"""
    stencil = np.zeros_like(img)

    # specify coordinates of the polygon
    polygon = np.array([[100,600], [200,400], [550,400], [700,600]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)

    # stencil = np.zeros_like(img[:,:,0])
    # apply frame mask

    masked = cv2.bitwise_and(img, img, mask=stencil)

    # apply image thresholding
    _, thresh1 = cv2.threshold(masked, 230, 255, cv2.THRESH_BINARY)

    def binary_threshold(img, thresh):
        return np.array(np.where(img <= thresh, 0, 255), dtype="uint8")

    img_inv = cv2.bitwise_not(img)
    img_inv = cv2.bitwise_and(img_inv, img_inv, mask=stencil)

    # ret, thresh2 = cv2.threshold(imagem, 90, 255, cv2.THRESH_BINARY)
    thresh2 = binary_threshold(img_inv, thresh=80)

    thresh_total = thresh1 | thresh2
    # apply Hough Line Transformation
    lines = cv2.HoughLinesP(thresh_total, 1, np.pi/180, 30, maxLineGap=200)
    # lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold = 10, minLineLength = 30, maxLineGap=200)
    lines = lines.reshape(-1, 4)
    dmy = img.copy()

    # Plot detected lines
    try:
        # for line in lines:
        #   x1, y1, x2, y2 = line[0]
        #   cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3) 
        means = kmeanLine(lines)
        angle = getAngle(means, dmy) 
        
        # cv2.imwrite('detected/'+str(cnt)+'.png',dmy)
        # cv2.imshow("", dmy)
        # cv2.waitKey(1)
        path = '_out/'  + str(angle)+  '.jpg'
        cv2.imwrite(path, dmy)
        print(-(angle - 90) / 140)
        return -(angle - 90) / 140

    except TypeError: 
        return 0


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

    client.start_recorder("./recording_naive_lane_detection1.log")
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