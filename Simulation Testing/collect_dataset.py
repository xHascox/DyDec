import carla
import argparse
import time
import random
import create_scenario
import json
from tqdm.auto import trange
import os

DEBUG = False
DATASET_PATH = "./dataset/dsv5/"
os.makedirs(DATASET_PATH, exist_ok=True)

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
        '-w','--walker-point',
        default=60,
        type=int,
        help='Spawn Point for walker')

    args = argparser.parse_args()

    file_path = "scenario definitions/dataset_scenarios.json"
    scenarios = load_pedestrian_data(file_path)

    SAVE_IMAGES = True # NOTE: Just for debugging, must be True to collect the dataset

    for scenario in scenarios["scenarios"]:
        town = scenario["world"]
        #description = scenario["description"]
        num_pedestrians = scenario["num_pedestrians"]
        vehicle_spawnpoint = scenario["vehicle_spawnpoint"]
        pedestrian_positions = scenario["pedestrian_positions"]
        pedestrian_direction = scenario["pedestrian_direction"] 

        try:
            world, bpLibrary, vehicle, walkerControllers, walkers, camera = create_scenario.create_scenario(args, town=town, num_pedestrians=num_pedestrians, vehicle_spawn_point=vehicle_spawnpoint, pedestrian_positions=pedestrian_positions)
            
            weather = world.get_weather()
            weather.sun_altitude_angle = 50
            weather.fog_density = 0
            weather.cloudiness = 0
            #weather.fog_distance = 10
            weather.precipitation = 0
            weather.wetness = 0
            world.set_weather(weather)
            world.set_weather(carla.WeatherParameters.ClearNoon)

            print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.')

            spectator = world.get_spectator()

            step = 0
            MAX_STEPS = 150
            
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
                            jump=False
                        )
                    walkers[i].apply_control(walker_control)
                
                camera_world_transform = camera.get_transform()
                
                spectator.set_transform(camera_world_transform)
                world.tick()


        except Exception as e:
            print(e)
        
        finally: # NOTE: images are captured during the scenario and saved in memory, as writing to disk can be too slow
            if SAVE_IMAGES:
                print(f"saving {len(create_scenario.img_storage)} images")
                import concurrent.futures
                def worker_save_image(image, town):
                    image.save_to_disk(f'{DATASET_PATH}{town}_{image.frame:06d}.png') #description and town
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    futures = []
                    for image in create_scenario.img_storage:
                        futures.append(
                            executor.submit(worker_save_image, image, town)
                        )
                    
                    # Wait for all saving operations to complete
                    concurrent.futures.wait(futures)

            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

            #Destroy vehicle
            print('\ndestroying the vehicle')
            vehicle.destroy()
            camera.destroy()

            #Destroy walker
            for walkerController in walkerControllers:
                walkerController.stop()
            print('\ndestroying the walker')
            for walker in walkers:
                walker.destroy()
            time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.')
