import itertools
import random
import numpy as np

from vehicle import Driver


class VehicleState:#######################TODO: MAYBE MAKE TWO OF THESE AND ONE QUANTIZES. USE INHERITANCE.
    LIDAR_THRESHOLDS = (0.25, 1, 5)
    SPEED_THRESHOLDS = (0.25, 0.75, 1.25)
    def __init__(self, lidar_vals, speed, crashed, checkpoint_count):
        self.enumerated_lidar_vals = [self.enumerate_val(v, self.LIDAR_THRESHOLDS) for v in lidar_vals]
        self.enumerated_speed = self.quantize_val(speed, self.SPEED_THRESHOLDS)
        self.crashed = crashed
        self.checkpoint_count = checkpoint_count

    def enumerate_val(self, val, thresholds):###todo: maybe get a new name.
        for i, thresh in enumerate(thresholds):
            if val > thresh:
                return i
        return i + 1

    def minimized_state(self):
        return ((self.enumerated_lidar_vals, self.enumerated_speed))###todo: what should be here?

    @classmethod
    def generate_all_minimized_states(self, lidar_laser_count):
        for lt in itertools.product(self.LIDAR_THRESHOLDS, repeat=lidar_laser_count):
            for st in self.SPEED_THRESHOLDS:
                yield (lt, st)


class VehicleAction:
    ALLOWED_SPEEDS = [0.1, 0.5, 1.5]
    ALLOWED_ANGLES = [-0.4, -0.2, 0, 0.2, 0.4]
    def __init__(self, speed, angle):
        self.speed = speed
        self.angle = angle

    @classmethod
    def generate_all_action_pairs(cls):
        for a in cls.ALLOWED_ANGLES:
            for s in cls.ALLOWED_SPEEDS:
                yield (a, s)


def generate_random_q(lidar_laser_count):#############TODO: WHAT RANDOM VALUES SHOULD BE USED?
    q = {}
    for state in VehicleState.generate_all_minimized_states(lidar_laser_count):
        q[state] = {action: random.random() for action in VehicleAction.generate_all_action_pairs()}
    return state

def update_q(q_matrix, trajectory, step_size):####TODO: CHECK ME WELL.
    # for i in range(len(trajectory)-1):
    #Iterate backwards through the trajectory and calculate the reward for each state.
    #Go backwards so the reward is attributed to the first instance of the state.
    state_rewards = {}
    reward_sum = 0
    # for i in range(len(trajector)-2, -1, -1):
    i = len(trajectory) - 2  # Start index at the last reward
    assert type(trajectory[i]) in (float, int), "something is wrong with trajectory. Expected number."
    while i >= 0:
        assert i >= 2, "trajectory index went out of range"
        reward = trajectory[i]
        reward_sum += reward
        action = trajectory[i-1]
        state = trajectory[i-2]
        if state not in state_rewards:
            state_rewards[state] = {}
        state_rewards[state][action] = reward_sum

    for state in state_rewards.values():
        for action in state_rewards[state].values():
            q_matrix[action][state] += step_size * (state_rewards[state][action] - q_matrix[action][state])


class CheckpointData:
    def __init__(self, dims, pose, color_field):

        #Calculate 3 corners of a rectangle such that the vector from the first to the second point is
        # perpendicular to the vector from the second point to the third. z values don't matter because we'll throw them out.
        shape_coords = np.array((
            (-dims[0]/2, dims[1]/2, 0, 1),
            (dims[0]/2, dims[1]/2, 0, 1),
            (dims[0]/2, -dims[1]/2, 0, 1),
        ))

        shape_coords = [pose @ coord for coord in shape_coords]####todo: do i need to set all z values to 0? probably doesn't matter.
        #Throw out the z coord and the trailing 1.
        shape_coords = [coord[0:2] for coord in shape_coords]
        self.rect_coords = shape_coords
        self.color_field = color_field

    def set_cp_color(self, rgb):
        self.color_field.setSFColor(list(rgb))
    
    def _make_vec2(self, p1, p2):
        return np.array([p2[i] - p1[i] for i in range(2)])

    def contains_point(self, point):
        #Made with help from this post. https://stackoverflow.com/a/2763387
        AB = self._make_vec2(self.rect_coords[0], self.rect_coords[1])
        BC = self._make_vec2(self.rect_coords[1], self.rect_coords[2])
        AM = self._make_vec2(self.rect_coords[0], point)
        BM = self._make_vec2(self.rect_coords[1], point)
        return 0 <= AB @ AM <= AB @ AB and 0 <= BC @ BM <= BC @ BC



CAR_DEF = "the_car"
CHECKPOINT_GROUP_DEF = "CHECKPOINTS"

class VehicleManager:
    def __init__(self):
        self.checkpoint_count = 0
        self.crashed = False
        self.checkpoints = None

        #Connect to the car.
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())

        self.car_node = self.driver.getFromDef(CAR_DEF)


        #Set up the lidar sensor.
        self.lidar_sensor = self.driver.getDevice('lidar')
        self.lidar_sensor.enable(self.timestep)

        #Set up the gps sensor.
        self.gps_sensor = self.driver.getDevice('gps')
        self.gps_sensor.enable(self.timestep)

        self.trans_field = self.car_node.getField("translation")
        self.starting_trans = self.trans_field.getSFVec3f()

        self.rotation_field = self.car_node.getField("rotation")
        self.starting_rotation = self.rotation_field.getSFRotation()

        #Enable contact tracking
        self.car_node.enableContactPointsTracking(self.timestep)

        self.load_checkpoint_info()


    def load_checkpoint_info(self):
        checkpoint_group = self.driver.getFromDef(CHECKPOINT_GROUP_DEF)
        checkpoint_group_values = checkpoint_group.getField("children")
        checkpoint_count = checkpoint_group_values.getCount()
        self.checkpoints = []
        for i in range(len(checkpoint_count)):##########TODO: MAKE SURE THE CHECKPOINTS COME THROUGH IN THE RIGHT ORDER!!!
            cp_solid = checkpoint_group_values.getMFNode(0)
            cp_pose = cp_solid.getPose()
            cp_pose = np.array(cp_pose).reshape((4,4))
            cp_shape = cp_solid.getField("children").getMFNode(0)  # NOTE: THIS ASSUMES THAT SHAPE IS THE FIRST CHILD OF THE SOLID!
            dims = cp_shape.getField("geometry").getSFNode().getField("size").getSFVec3f()
            appearance_obj = cp_shape.getField("appearance") # NOTE: THIS ASSUMES THE APPEARANCE IS A PBRAppearance object.
            color_field = appearance_obj.getSFNode().getField("baseColor")
            cp_data = CheckpointData(dims, cp_pose, color_field)
            self.checkpoints.append(cp_data)

    def get_state(self):
        #####todo: check to see if we are in contact with a checkpoint?
        lidar_value = self.lidar_sensor.getValue()
        speed = self.driver.getCruisingSpeed()
        angle = self.driver.getSteeringAngle()
        if self.check_for_collisions():#############################################TODO: FIGURE OUT HOW TO SEE A CHECKPOINT!!! MAYBE GPS
            print("crashed!!!")
            self.crashed = True

        #Check to see if we entered the next checkpoint.
        next_cp = self.checkpoints[self.checkpoint_count]####todo: will this run off the edge after hitting the last checkpoint???
        gps_coord = self.gps_sensor.getValues()
        if next_cp.contains_point(gps_coord):
            next_cp.set_cp_color((0, 1, 0))
            self.checkpoint_count += 1
        return VehicleState(lidar_value, speed, self.crashed, self.checkpoint_count)

    def execute_action(self, speed_angle):#########TODO: MAX SPEED OF 1.8???
        ##############################todo: be able to do breaks?? maybe driver does it for me.
        ##todo: max angle of -0.4, 0.4??
        speed, angle = speed_angle
        self.driver.setSteeringAngle(angle)
        self.driver.setCruisingSpeed(speed)

    def reset_car(self):
        self.trans_field.setSFVec3f(self.starting_trans)
        self.rotation_field.setSFRotation(self.starting_rotation)
        self.driver.resetPhysics()
        #Reset checkpoint colors.
        for checkpoint in self.checkpoints:
            checkpoint.set_cp_color((1, 1, 0))
        #############################################TODO: USE SUPERVISOR API TO RESET THE COORDINATES. PROBABLY SHOULD GET THE COORDINATES AT INITIALIZATION. RESET PHYSICS AND EVERYTHING TOO!!!
        ###################################todo: need to check for collisions with walls and checkpoints!!!!!!!

    def check_for_collisions(self):
        collisions = self.car_node.getContactPoints()
        return len(collisions) > 0
    
    def step(self):
        return self.driver.step()

###todo: what should this be?
def calculate_reward(state, action):
    return state.checkpoint_count - int(state.crashed)


LIDAR_LASER_COUNT = 3###TODO: GET THIS FROM ENVIRONMENT!

q_matrix = generate_random_q(LIDAR_LASER_COUNT)
epsilon = 0.9################TODO: SHOULD THIS BE CLOSE TO 1 OR 0?

car = VehicleManager()

trajectory = []#########todo: do we need to actually store this? I think so.

race_time = 0

MAX_RACE_TIME = 10

race_counter = 1




# while car.step() != -1:

#     current_state = car.get_state()
#     trajectory.append(current_state)##########################TODO: IS THE ORDER OF THINGS IN THE TRAJECTORY CORRECT?
#     action_values = q_matrix[current_state.minimized_state]
#     #Choose an action
#     if random.random() < epsilon:
#         #Use the best action.
#         next_action = max(action_values.values(), key=lambda x: q_matrix[x])
#     else:
#         #Take a random action.
#         next_action = random.choice(action_values.values())

#     car.execute_action(next_action)
#     trajectory.append(next_action)

#     reward = calculate_reward(current_state, next_action)###TODO: DISCOUNTED REWARD?
#     trajectory.append(reward)

#     race_time += car.timestep / 1000

#     race_over = False
#     if race_time >= MAX_RACE_TIME:
#         print("timeout!")
#         race_over = True
#     if current_state.crashed:
#         print("crashed!")
#         race_over = True

#     if race_over:
#         print("TODO: RESTART THE RACE!!!")
#         step_size = 1 / race_counter
#         update_q(q_matrix, trajectory, step_size)
#         race_counter += 1
#         trajectory = []
