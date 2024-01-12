def simulate_takeoff(host,port,takeoff_distance=3,start_flight = True, disarmed = False):
	from udacidrone import Drone
	from udacidrone.connection import MavlinkConnection

	conn = MavlinkConnection(f'tcp:{host}:{port}',threaded= True)
	drone = Drone(conn)
	if start_flight:
		drone.start()
		drone.take_control()
		drone.arm()
	drone.takeoff(takeoff_distance)
	if disarmed:
		drone.disarm()
		drone.release_control()

def simulate_position_takeover(host,port,north_pos,east_pos,altitude,angle_in_rad,start_flight=True,disarmed=False):
	from udacidrone import Drone
	from udacidrone.connection import MavlinkConnection

	conn = MavlinkConnection(f'tcp:{host}:{port}',threaded= True)
	drone = Drone(conn)
	if start_flight:
		drone.start()
		drone.take_control()
		drone.arm()
	drone.cmd_position(north_pos,east_pos,altitude,angle_in_rad)
	drone.land()
	if disarmed:
		drone.disarm()
		drone.release_control()
	
if __name__=="__main__":
	host             = "localhost"
	port             = "5067"
	takeoff_distance = 30
	simulate_takeoff(host,port,takeoff_distance)
	
	north_pos    = 2
	east_pos     = 1
	altitude     = 0.5
	angle_in_rad = 0
	simulate_position_takeover(host,port,north_pos,east_pos,altitude,angle_in_rad,start_flight = False,disarmed = True)