import random
import carla
import math
#import detect_pedestrian


img_storage = []

def calculate_relative_direction(vehicle_vector, angle_offset=0):
    """
    Calculate a direction vector relative to the vehicle's direction
    
    :param vehicle_vector: The vehicle's forward direction vector
    :param angle_offset: Angle offset in degrees (0 = forward, 90 = right, -90 = left)
    :return: Normalized direction vector
    """
    # Convert angle offset to radians
    angle_rad = math.radians(angle_offset)
    
    # Rotate the vehicle vector
    rotated_x = (
        vehicle_vector.x * math.cos(angle_rad) - 
        vehicle_vector.y * math.sin(angle_rad)
    )
    rotated_y = (
        vehicle_vector.x * math.sin(angle_rad) + 
        vehicle_vector.y * math.cos(angle_rad)
    )
    
    # Create and normalize the new direction vector
    relative_direction = carla.Vector3D(x=rotated_x, y=rotated_y, z=0)
    return relative_direction.make_unit_vector()

def create_scenario(args, town="Town02", vehicle_spawn_point=31, num_pedestrians=5, 
                    min_dist=10, max_dist=70,  # Distance range in front
                    min_lateral=-3, max_lateral=3, pedestrian_target=[],
                    pedestrian_positions=[], pedestrian_ids=None, camera_active=True,
                    texture_to_id=None):
    client = carla.Client('localhost', 2000) # 2000
    client.set_timeout(25.0)
    client.load_world(town)
    synchronous_master = False

    world = client.get_world()
    #traffic_manager = client.get_trafficmanager(args.tm_port)
    bpLibrary = world.get_blueprint_library()

    settings = world.get_settings()
    if not args.asynch:
        #traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        else:
            synchronous_master = False
    else:
        print("You are currently in asynchronous mode. If this is a traffic simulation, \
                you could experience some issues. If it's not working correctly, switch to \
                synchronous mode by using traffic_manager.set_synchronous_mode(True)")
    print("synchron: ", settings.synchronous_mode)
    world.apply_settings(settings)
    
    # Finding actors
    bpLibrary = world.get_blueprint_library()

    ## Finding vehicle
    vehicleBP = bpLibrary.filter('vehicle.tesla.model3')[0]
    vehicleBP.set_attribute('color', "255,255,255")
    vehicleBP.set_attribute('role_name', 'ego')
    if not vehicleBP:
        raise ValueError("Couldn't find any vehicles with the specified filters")

    vehicle_spawn_points = world.get_map().get_spawn_points()

    vehicle = world.spawn_actor(vehicleBP, vehicle_spawn_points[vehicle_spawn_point])
    
    #vehicle.set_autopilot(True)
    vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))


    ### Spawn walker
    # Get vehicle's transform
    world.tick()
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_forward = vehicle_transform.get_forward_vector()
    vehicle_right = vehicle_transform.get_right_vector()
    vehicle_up = vehicle_transform.get_up_vector()
    # Sample random distance and lateral offset within the specified range

    dist = [e["x"] for e in pedestrian_positions]
    lateral = [e["y"] for e in pedestrian_positions]
    elevation = [e["z"] for e in pedestrian_positions]

    pedestrian_controllers = []
    pedestrian_walkers = []
    for pedestrian_i in range(num_pedestrians):
        walker = None
        if pedestrian_ids is None:
            walkerBP = bpLibrary.filter(texture_to_id.get("benign"))[0] # 0002 # pedestrian_ids
        else:
            walkerBP = bpLibrary.filter(texture_to_id.get(pedestrian_ids[pedestrian_i], texture_to_id["benign"]))[0] # 0002 # pedestrian_ids
        walkerControllerBP = bpLibrary.find('controller.ai.walker')
        if not walkerBP:
            raise ValueError("Couldn't find any walkers with the specified filters")
        while walker == None:
            if dist:
                forward_dist = dist[pedestrian_i]
            else:
                forward_dist = random.uniform(min_dist, max_dist)

            if lateral:
                lateral_offset = lateral[pedestrian_i]
            else:
                lateral_offset = random.uniform(min_lateral, max_lateral)

            if elevation:
                elevation_offset = elevation[pedestrian_i]
            else:
                elevation_offset = 0
            
            # Calculate spawn location in front of the vehicle
            spawn_location = vehicle_location + vehicle_forward * forward_dist + vehicle_right * lateral_offset + vehicle_up * elevation_offset

            spawn_transform = carla.Transform(spawn_location)
            print(f"trying random spawn point for {pedestrian_i}", spawn_transform)
            print("vehicle_location", vehicle_location)


            walker = world.try_spawn_actor(walkerBP, spawn_transform)

        world.tick()
        walkerController = world.spawn_actor(walkerControllerBP, carla.Transform(), walker)
        world.tick()

        pedestrian_controllers.append(walkerController)
        pedestrian_walkers.append(walker)
    
    # Retrieve the spectator object
    spectator = world.get_spectator()

    # Set the spectator with our transform
    spectator.set_transform(carla.Transform(carla.Location(x=10, y=113.387848, z=1.780676), carla.Rotation(pitch=-19.108484, yaw=0, roll=0)))

    world.tick()
    
    target_dist = [e["x"] for e in pedestrian_target]
    target_lateral = [e["y"] for e in pedestrian_target]
    target_elevation = [e["z"] for e in pedestrian_target]
    
    camera = None
    if camera_active:
        print("creating camera")
        camera_bp = bpLibrary.find('sensor.camera.rgb')
        # settings from carla_garage\team_code\config.py
        camera_bp.set_attribute('image_size_x', '2048') # 1024
        camera_bp.set_attribute('image_size_y', '1024') # 512
        camera_bp.set_attribute('fov', '90') # 110
        camera_bp.set_attribute('sensor_tick', '0.1')
        camera_bp.set_attribute('blur_amount', '0')
        camera_bp.set_attribute('motion_blur_intensity', '0')
        camera_bp.set_attribute('shutter_speed', '2000')
        camera_bp.set_attribute('fstop', '16')
        
        print("created camera")
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=-1.5, z=2), carla.Rotation(pitch=0, yaw=0, roll=0)), attach_to=vehicle,  attachment_type=carla.AttachmentType.Rigid)
        global img_storage
        global contains_pedestrian
        img_storage = []
        contains_pedestrian = False
        def camera_listener(image):
            img_storage.append(image)
        camera.listen(camera_listener)

    print("screate scenario returning")
    return world, bpLibrary, vehicle, pedestrian_controllers, pedestrian_walkers, camera
