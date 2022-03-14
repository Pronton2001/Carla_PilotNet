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
actor_list = []
W, H = 320, 180
SHOW_CAM = True
front_camera = None
throttle = 0.5

try:
	client = carla.Client('localhost', 2000)
	client.set_timeout(10.0) # seconds

	world = client.get_world()
	world.unload_map_layer(carla.MapLayer.All)

	blueprint_library = world.get_blueprint_library()

	# bp = random.choice(blueprint_library.filter('vehicle'))
	bp = blueprint_library.filter('model3')[0]
	transform = world.get_map().get_spawn_points()[100]

	vehicle = world.spawn_actor(bp, transform)
	# vehicle.set_autopilot(True)

	# actor_list.append(vehicle)
	# print('created %s' % vehicle.type_id)

	# # Find the blueprint of the sensor.
	# camera_bp = blueprint_library.find('sensor.camera.rgb')
	# # Modify the attributes of the blueprint to set image resolution and field of view.
	# camera_bp.set_attribute('image_size_x', str(W))
	# camera_bp.set_attribute('image_size_y', str(H))
	# camera_bp.set_attribute('fov', '40')

	# # # Set the time in seconds between sensor captures
	# camera_bp.set_attribute('sensor_tick', '1.0')
	# camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
	# camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

	# actor_list.append(camera)
	# print('created %s' % camera.type_id)

	client.replay_file("./recording_baseline3_Town1_pid.log", 1, 100, vehicle.id)

finally:

    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print('done.')