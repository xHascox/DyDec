import carla

def draw_all(world):
    spawn_points = world.get_map().get_spawn_points()
    # Draw the spawn point locations as numbers in the map
    for i, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(i), life_time=400)

def draw_route(world, name, route, life_time=600):
    spawn_points = world.get_map().get_spawn_points()
    # Create route : from the chosen spawn points
    route_locs = []
    for ind in route:
        route_locs.append(spawn_points[ind].location)
    # Now let's print them in the map so we can see our routes
    world.debug.draw_string(route_locs[0], f'Starting point {name}', life_time= life_time, color=carla.Color(255,0,0))
    world.debug.draw_string(route_locs[-1], f'Destination {name}', life_time= life_time, color=carla.Color(255,0,0))

    for loc in route_locs[1:-1]:
        world.debug.draw_string(loc, 'waypoint', life_time= life_time, color=carla.Color(0,200,0))


# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000) # 2000
client.set_timeout(25.0)
client.load_world("Town07_stop") # Town02_stop_f
synchronous_master = False

world = client.get_world()
bpLibrary = world.get_blueprint_library()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

draw_all(world)
"""
route_L_indices = [50, 55]
draw_route(world, 'L', route_L_indices)
route_S_indices = [230, 226]
draw_route(world, 'S', route_S_indices)
route_R_indices = [185, 22]
draw_route(world, 'R', route_R_indices)
"""

while True:
    world.tick()
