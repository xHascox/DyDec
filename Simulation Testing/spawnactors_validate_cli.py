import carla
import argparse
import time
import random
import create_scenario
import json
from tqdm.auto import trange
import sys
import os
from datetime import datetime
import csv
import cv2

# Add the directory containing PCLA.py to the Python path and machine specific walker IDs

if 'MACHINE_NAME' in os.environ:
    MACHINE_NAME = os.environ['MACHINE_NAME']
else:
    MACHINE_NAME = "Vortex"  # TODO adjust accordingly
    print(f"Using default MACHINE_NAME: {MACHINE_NAME}\nYou can set it in the environment variable MACHINE_NAME")
SAVE_IMAGES = 4 # NOTE must evaluate True or False, but if int is MAX_STEPS divided by SAVE_IMAGES to take the representative image

if MACHINE_NAME == "hasco":
    texture_to_id = {"stop": "walker.pedestrian.0045", 
                    "adv": "walker.pedestrian.0015", 
                    "disguise": "walker.pedestrian.0024",
                    "benign": "walker.pedestrian.0002",
                    "adv_left": "walker.pedestrian.0006",
                    "adv_right": "walker.pedestrian.0001",}
    SAVE_IMAGES = 15
    PCLA_PATH = "/home/hasco/PCLA"
elif MACHINE_NAME == "Vortex":
    texture_to_id = {"stop": "walker.pedestrian.0027", 
                    "adv": "walker.pedestrian.0003", 
                    "disguise": "walker.pedestrian.0046",
                    "benign": "walker.pedestrian.0029",
                    "adv_left": "walker.pedestrian.0038",
                    "adv_right": "walker.pedestrian.0045",}
    PCLA_PATH = "/home/vortex/PCLA"



sys.path.append(os.path.dirname(os.path.abspath(PCLA_PATH)))

from PCLA import PCLA

print(PCLA)
#print(PCLA.__path__)

print("ok")
import inspect
import math

def get_distance(loc1, loc2):
    """Calculate Euclidean distance between two carla.Location objects."""
    return math.sqrt(
        (loc1.x - loc2.x) ** 2 +
        (loc1.y - loc2.y) ** 2 +
        (loc1.z - loc2.z) ** 2
    )

DEBUG = False

# Load the JSON data from a file
def load_pedestrian_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        return None

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '-wp','--walker-point',
        default=60,
        type=int,
        help='Spawn Point for walker')
    argparser.add_argument(
        '-s','--scenario',
        default=None,
        type=int,
        help='Which Scenario to run, 0 for first (default)')
    argparser.add_argument(
        '-a','--agent',
        default=None,
        type=str,
        help="""Which Agent to run, if_if, neat_neat, 
        tfpp_l6_#, tfpp_lav_, tfpp_aim_, tfpp_wp_ (#=seed, 0-2)""")
    argparser.add_argument(
        '-w','--weather',
        default=None,
        type=str,
        help='Which weather to use, ClearNoon, Specific, etc. (default: None)')
    argparser.add_argument(
        '-sw','--save_weather',
        default=None,
        type=str,
        help='save the weather to file if set to "SAVE"')
    argparser.add_argument(
        '-l','--logfile',
        default=None,
        type=str,
        help='Main Log FIle Path')
    argparser.add_argument(
        '-d','--datafile',
        type=str,
        help='Filepath to JSON file containing all scenarios')
    
    args = argparser.parse_args()

    # Example usage
    file_path = args.datafile
    scenarios = load_pedestrian_data(file_path)

    if args.scenario is not None:
        SKIP_SCENARIOS = args.scenario
    else:
        SKIP_SCENARIOS = 0 # NOTE: MAINLY FOR DEBUGGING
    MAX_STEPS = 80#100

    agent = args.agent

    route = "./route.xml"
    os.environ.pop("ROUTES", None)
    os.environ.pop("UNCERTAINTY_THRESHOLD", None)
    os.environ.pop("STOP_CONTROL", None)
    os.environ.pop("DIRECT", None)
    if agent == "if_if":
        os.environ["ROUTES"] = route
    elif agent.startswith("tfpp_l6"):
        os.environ["UNCERTAINTY_THRESHOLD"] = "033"
    elif agent.startswith("tfpp_lav"):
        os.environ["STOP_CONTROL"] = "1"
    elif agent.startswith("tfpp_aim"):
        os.environ["DIRECT"] = "0"
    elif agent.startswith("tfpp_wp"):
        os.environ["DIRECT"] = "0"

    client = carla.Client('localhost', 2000) # 2000
    client.set_timeout(65.0)

    terminated_successfully = False

    for scenario in scenarios["scenarios"][SKIP_SCENARIOS:]:
        town = scenario["world"]
        description = scenario["description"]
        num_pedestrians = scenario["num_pedestrians"]
        vehicle_spawnpoint = scenario["vehicle_spawnpoint"]
        vehicle_route = scenario["vehicle_route"]
        pedestrian_positions = scenario["pedestrian_positions"]
        pedestrian_direction = scenario["pedestrian_direction"] 
        pedestrian_ids = scenario["pedestrian_ids"] if "pedestrian_ids" in scenario else None
        change_weather = scenario["change_weather"] if "change_weather" in scenario else None
        save_weather = scenario["save_weather"] if "save_weather" in scenario else args.save_weather
        try:
            camera_active = SAVE_IMAGES
            world, bpLibrary, vehicle, walkerControllers, walkers, camera = create_scenario.create_scenario(args, town=town, num_pedestrians=num_pedestrians, vehicle_spawn_point=vehicle_spawnpoint, pedestrian_positions=pedestrian_positions, pedestrian_ids=pedestrian_ids, camera_active=camera_active, texture_to_id=texture_to_id)

            CHANGE_WEATHER = change_weather
            weather_directory = "weather_presets"
            print("CHANGE_WEATHER", CHANGE_WEATHER)
            weather = world.get_weather()
            if args.weather:
                CHANGE_WEATHER = args.weather
            print("CHANGE_WEATHER", CHANGE_WEATHER)
            if CHANGE_WEATHER == "ClearNoon":
                world.set_weather(carla.WeatherParameters.ClearNoon)
            elif CHANGE_WEATHER == "ClearNoon_180":
                pass
                weather.sun_azimuth_angle = (weather.sun_azimuth_angle + 180) % 360
                world.set_weather(weather)
            elif CHANGE_WEATHER == "ClearNoon_90":
                pass
                weather.sun_azimuth_angle = (weather.sun_azimuth_angle + 90) % 360
                world.set_weather(weather)
            elif CHANGE_WEATHER == "Town07_stop":
                world.set_weather(carla.WeatherParameters.ClearNoon)
                weather = world.get_weather()
                weather.sun_altitude_angle = 30
                weather.sun_azimuth_angle = 15
                #weather.fog_density = 0
                #weather.cloudiness = 0
                #weather.fog_distance = 10
                #weather.precipitation = 0
                #weather.wetness = 0
                world.set_weather(weather)
            else:
                try:
                    load_path = os.path.join(weather_directory, f'{CHANGE_WEATHER}_weather.json')
                    with open(load_path, 'r') as f:
                        weather_data = json.load(f)
                    
                    # Create new weather parameters and apply saved values
                    weather = carla.WeatherParameters()
                    weather.cloudiness = weather_data['cloudiness']
                    weather.precipitation = weather_data['precipitation']
                    weather.precipitation_deposits = weather_data['precipitation_deposits']
                    weather.wind_intensity = weather_data['wind_intensity']
                    weather.sun_azimuth_angle = weather_data['sun_azimuth_angle']
                    weather.sun_altitude_angle = weather_data['sun_altitude_angle']
                    weather.fog_density = weather_data['fog_density']
                    weather.fog_distance = weather_data['fog_distance']
                    weather.wetness = weather_data['wetness']
                    
                    # Apply the weather to the world
                    world.set_weather(weather)
                    print("Loaded and applied weather from 'saved_weather.json'")
                except FileNotFoundError:
                    print(f"No saved weather file found {load_path}!")
                except Exception as e:
                    print(f"Error loading weather: {e}")
            
            if save_weather == "SAVE":
                # Save current weather parameters to a file
                current_map = os.path.basename(world.get_map().name)
                weather_data = {
                    'map_name': current_map,
                    'sun_altitude_angle': weather.sun_altitude_angle,
                    'sun_azimuth_angle': weather.sun_azimuth_angle,
                    'fog_density': weather.fog_density,
                    'fog_distance': weather.fog_distance,
                    'cloudiness': weather.cloudiness,
                    'precipitation': weather.precipitation,
                    'precipitation_deposits': weather.precipitation_deposits,
                    'wetness': weather.wetness,
                    'wind_intensity': weather.wind_intensity,
                }       
                
                
                os.makedirs(weather_directory, exist_ok=True)
                save_path = os.path.join(weather_directory, f'{current_map}_weather.json')
                print(f"Saving weather settings to {save_path}")
                with open(save_path, 'w') as f:
                    json.dump(weather_data, f, indent=4)
                print(f"Weather settings saved to {save_path}")

            print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.')

            spectator = world.get_spectator()

            step = 0
            
            
            vehicle_spawn_points = world.get_map().get_spawn_points()
            
            waypoints = PCLA.location_to_waypoint(client, vehicle_spawn_points[vehicle_route[0]].location, vehicle_spawn_points[vehicle_route[1]].location)
            # Pass the waypoints to PCLA to make it usable in PCLA

            import time
            PCLA.route_maker(waypoints, savePath=route)
            
            pcla = PCLA.PCLA(agent, vehicle, route, client)
        
            print('Spawned the vehicle with model =', agent,', press Ctrl+C to exit.\n')
            print("scenario", description)    
                
            initial_location = vehicle.get_location()
            initial_time = world.get_snapshot().timestamp.elapsed_seconds

            speed_sum = 0
            statistics = {"brake": [], "throttle": [], "steer": [], "speed": []}
            for step in trange(MAX_STEPS):
                # Walker controll
                for i, walker_controller in enumerate(walkerControllers):
                    
                    vehicle_transform = vehicle.get_transform()
                    vehicle_location = vehicle_transform.location
                    vehicle_forward = vehicle_transform.get_forward_vector()

                    angle_offset = pedestrian_direction[i]["angle"] if pedestrian_direction else 0
                    pedestrian_speed = pedestrian_direction[i]["speed"] if pedestrian_direction else 1
                    direction = create_scenario.calculate_relative_direction(vehicle_forward, angle_offset=angle_offset)
                    walker_control = carla.WalkerControl(
                            direction=direction,
                            speed=pedestrian_speed,
                            jump=False  # Set to True if you want the walker to jump
                        )
                    walkers[i].apply_control(walker_control)

                if camera:
                    camera_world_transform = camera.get_transform()
                    camera_world_transform.rotation.yaw = camera_world_transform.rotation.yaw + 45.0
                    spectator.set_transform(camera_world_transform) # new_yaw = camera_world_transform.rotation.yaw + 45.0
                else:
                    spectator.set_transform(carla.Transform(vehicle.get_transform().location + carla.Location(z=1.5), vehicle.get_transform().rotation))

                ego_action = pcla.get_action()
                
                statistics["brake"].append(ego_action.brake)
                statistics["throttle"].append(ego_action.throttle)
                statistics["steer"].append(ego_action.steer)
                statistics["speed"].append(vehicle.get_velocity().length())
                vehicle_speed = vehicle.get_velocity().length()
                speed_sum += vehicle_speed
                vehicle.apply_control(ego_action)

                world.tick()
            
            final_location = vehicle.get_location()
            final_time = world.get_snapshot().timestamp.elapsed_seconds
            # Calculate distance and average speed
            distance = get_distance(initial_location, final_location)
            time_elapsed = final_time - initial_time
            avg_speed = distance / time_elapsed if time_elapsed > 0 else 0

            print(f"{agent} >>> {description}")
            print(f"average speed (distance/time): {avg_speed:.2f} m/s")

            print(f"average speed: {speed_sum / step}")

            # Prepare CSV entry
            timeseries_dir = 'timeseries'
            img_dir = 'scenario_imgs'
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            filename_timeseries = f'{timeseries_dir}/{timestamp}.csv'
            filename_scenarioimg = f'{img_dir}/{timestamp}.png'
            entry = [timestamp, agent, description, f"{avg_speed:.2f}", args.weather, filename_timeseries, filename_scenarioimg]
            
            os.makedirs(img_dir, exist_ok=True)

            
            os.makedirs(timeseries_dir, exist_ok=True)
            with open(filename_timeseries, 'w', newline='') as log_file:
                writer = csv.writer(log_file)
                # Write header
                writer.writerow(statistics.keys())
                # Write each step's throttle value
                for values in zip(*statistics.values()):
                    writer.writerow(values)

            # Append to the CSV log file
            main_logfile = 'logs/scenario_log.csv'
            if args.logfile is not None:
                main_logfile = args.logfile
            with open(main_logfile, 'a', newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerow(entry)

            terminated_successfully = True

        except Exception as e:
            print(e)
        
        finally:
            try:
                pcla.cleanup()
            except Exception as e:
                print(f"!!!pcla cleanup failed: {e}")
                pcla.agent_instance.destroy()
                pcla = None

            try:
                if int(SAVE_IMAGES)==SAVE_IMAGES:
                    img = create_scenario.img_storage[MAX_STEPS//SAVE_IMAGES]
                    img.save_to_disk(filename_scenarioimg)
                    img = create_scenario.img_storage[-1]
                    img.save_to_disk(filename_scenarioimg+ "_last")
                elif SAVE_IMAGES:
                    print(f"saving {len(create_scenario.img_storage)} images")
                    import concurrent.futures
                    def worker_save_image(image, town):
                        image.save_to_disk(f'output/dsv5/{town}_{image.frame:06d}.png') #description and town
                    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                        futures = []
                        for image in create_scenario.img_storage:
                            futures.append(
                                executor.submit(worker_save_image, image, town)
                            )
                        
                        # Wait for all saving operations to complete
                        concurrent.futures.wait(futures)
            except Exception as e:
                print(f"image saving failed: {e}")

            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

            #Destroy walker
            try:
                for walkerController in walkerControllers:
                    walkerController.stop()
                for walker in walkers:
                    walker.destroy()
            except Exception as e:
                print(f"!!!walker controller stop failed: {e}")
            
            
            time.sleep(0.5)
        
        if args.scenario is not None:
            break
        #break # TODO REMOVE IF NO GPU VRAM MEMORY ERRORS
    if not terminated_successfully:
        sys.exit(2)
    sys.exit(0)

if __name__ == '__main__':
    
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.')
