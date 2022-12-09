import itertools
import json
import math
import numpy as np
import random
from vehicle import Driver



class AsdfAlgorithm:#################TODO: RENAME ME!
    def __init__(self, epsilon=0.1, discount_factor=0.999):
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def get_next_action(self, q_matrix, current_state):
        state = current_state.minimized_state
        if random.random() < self.epsilon:
            # print("random action!")
            return get_random_action()
        else:
            # print("best known action")
            return max(q_matrix[state].keys(), key=lambda a: q_matrix[state][a])


    def update_q(self, q_matrix, trajectory, step_size):####TODO: CHECK ME WELL.       ######################TODO: MAKE THIS WORK WITH AN AVERAGED TARGET OVER MULTIPLE EPISODES WITH THE SAME Q VALUES(?) RESPONSE FROM DR. TRIPATHY ABOUT THIS?
        #Iterate backwards through the trajectory and calculate the reward for each state.
        #Go backwards so the reward is attributed to the first instance of the state.
        state_rewards = {}
        reward_sum = 0
        for trajectory_point in reversed(trajectory):
            reward_sum = (reward_sum * self.discount_factor) + trajectory_point.reward#############TODO: DISCOUNT FACTOR I THINK!!!
            state = trajectory_point.state.minimized_state##############TODO: MINIMIZED STATE IS UGLY!!! PROBABLY MAKE A MORE MINIMAL OBJECT THAT GOES IN THE STATE OBJECT.
            if state not in state_rewards:
                state_rewards[state] = {}
            state_rewards[state][trajectory_point.action] = reward_sum

        for state in state_rewards.keys():
            for action in state_rewards[state].keys():
                print("updating q", state, action, state_rewards[state][action])
                try:
                    q_matrix[state][action] += step_size * (state_rewards[state][action] - q_matrix[state][action])
                except KeyError:
                    import pdb; pdb.set_trace()






class VehicleState:
    LIDAR_THRESHOLDS = (0.25, 1, 5)
    def __init__(self, lidar_vals, crossing_checkpoint, crashed, checkpoint_count, crossed_finishline, quantize_values=True):####TODO: SHOULD CHECKPOINT COUNT BE HERE? SHOULD IT BE A BOOLEAN VALUE TO SAY WHETHER IT IS CROSSING A CHECKPOINT?
        print("lidar", lidar_vals)
        if quantize_values:
            self.lidar_vals = tuple(self.enumerate_val(v, self.LIDAR_THRESHOLDS) for v in lidar_vals)
        else:
            self.lidar_vals = tuple(lidar_vals)
        self.crashed = crashed
        self.crossing_checkpoint = crossing_checkpoint
        self.checkpoint_count = checkpoint_count
        self.crossed_finishline = crossed_finishline


    def enumerate_val(self, val, thresholds):###todo: maybe get a new name.
        return min(thresholds, key=lambda thresh: abs(val-thresh))

    @property
    def minimized_state(self):
        return self.lidar_vals

    # def __hash__(self):
        # return hash(self.minimized_state)

    def __str__(self):
        return str(f"lidar: {self.lidar_vals}   crashed: {self.crashed}   checkpoint_count: {self.checkpoint_count}")

    @classmethod
    def generate_all_minimized_states(self, lidar_laser_count):
        for lt in itertools.product(self.LIDAR_THRESHOLDS, repeat=lidar_laser_count):
            yield lt



ALLOWED_ACTIONS = [-0.4, -0.2, 0, 0.2, 0.4]###TODO: IS 0.4 ENOUGH? GO TO MAX.
def get_random_action():
    return random.choice(ALLOWED_ACTIONS)


def generate_random_q(lidar_laser_count):
    q = {}
    for state in VehicleState.generate_all_minimized_states(lidar_laser_count):
        q[state] = {action: random.random() for action in ALLOWED_ACTIONS}
    return q



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


class TrajectoryTriplet:
    def __init__(self, state: VehicleState, action: float, reward: float):
        self.state = state
        self.action = action
        self.reward = reward


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
        self.car_node.enableContactPointsTracking(self.timestep)########TODO: ADD ME BACK!

        self.load_checkpoint_info()


    def load_checkpoint_info(self):
        checkpoint_group = self.driver.getFromDef(CHECKPOINT_GROUP_DEF)
        checkpoint_group_values = checkpoint_group.getField("children")
        checkpoint_count = checkpoint_group_values.getCount()
        self.checkpoints = []
        for i in range(checkpoint_count):##########TODO: MAKE SURE THE CHECKPOINTS COME THROUGH IN THE RIGHT ORDER!!!
            cp_solid = checkpoint_group_values.getMFNode(i)
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
        lidar_value = self.lidar_sensor.getRangeImage()
        if self.check_for_collisions():#############################################TODO: FIGURE OUT HOW TO SEE A CHECKPOINT!!! MAYBE GPS
            print("crashed!!!")
            self.crashed = True

        #Check to see if we entered the next checkpoint.
        crossing_checkpoint = False
        if self.checkpoint_count < len(self.checkpoints):
            next_cp = self.checkpoints[self.checkpoint_count]####todo: will this run off the edge after hitting the last checkpoint???
            gps_coord = self.gps_sensor.getValues()
            if next_cp.contains_point(gps_coord):
                print("just hit checkpoint", self.checkpoint_count)
                next_cp.set_cp_color((0, 1, 0))
                self.checkpoint_count += 1
                crossing_checkpoint = True
        else:
            print("final checkpoint reached but race didn't finish?")
        
        crossed_finishline = self.checkpoint_count == len(self.checkpoints)
        if crossed_finishline:
            print("asdfasdf", self.checkpoint_count, len(self.checkpoints))
        return VehicleState(lidar_value, crossing_checkpoint, self.crashed, self.checkpoint_count, crossed_finishline)

    def execute_action(self, angle):
        ##todo: max angle of -0.4, 0.4??
        self.driver.setSteeringAngle(angle)
        self.driver.setCruisingSpeed(1.5)#########TODO: MAX SPEED OF 1.8???

    def reset_car(self):
        self.trans_field.setSFVec3f(self.starting_trans)
        self.rotation_field.setSFRotation(self.starting_rotation)
        self.car_node.resetPhysics()
        #Reset checkpoint colors.
        for checkpoint in self.checkpoints:
            checkpoint.set_cp_color((1, 0, 0))
        self.checkpoint_count = 0
        self.crashed = False
        car.step()
        print("stepped!")##################TODO: REMOEV ME????

    def check_for_collisions(self):
        collisions = self.car_node.getContactPoints()
        return len(collisions) > 0
    
    def step(self):
        return self.driver.step()

def calculate_reward(state, action, crash_penalty=3):
    return int(state.crossing_checkpoint) - (crash_penalty * int(state.crashed))


def save_info(path, iteration_count, car, total_reward):
    print("saving")
    with open(path, "a") as fout:
        fout.write(json.dumps({"iteration_count": iteration_count,
                                "checkpoint_count": car.checkpoint_count,
                                "total_reward": total_reward}))


def run(car):

    LIDAR_LASER_COUNT = 4###TODO: GET THIS FROM ENVIRONMENT!

    q_matrix = generate_random_q(LIDAR_LASER_COUNT)
    epsilon = 0.1

    race_time = 0

    MAX_EPISODE_TIME = 10

    race_counter = 1
    discount_factor = 0.99

    output_file_path = "c:\\temp\\rlRacerOut.txt"###todo: maybe need to pass this in via command line.

    algo = AsdfAlgorithm(epsilon, discount_factor)
    trajectory = []
    episode_time = 0

    while car.step() != -1:

        current_state = car.get_state()
        #Choose an action
        next_action = algo.get_next_action(q_matrix, current_state)
        # print("picked action", next_action)
        car.execute_action(next_action)

        reward = calculate_reward(current_state, next_action)

        trajectory.append(TrajectoryTriplet(current_state, next_action, reward))

        episode_time += car.timestep / 1000


        race_over = False
        if episode_time >= MAX_EPISODE_TIME:##todo: change to episode time?
            print("timeout!")
            race_over = True
        if current_state.crashed:
            print("crashed!")
            race_over = True
        if current_state.crossed_finishline:
            print("crossed finishline!!")
            race_over = True

        if race_over:
            total_race_reward = sum(t.reward for t in trajectory)######################                         TODO: MAKE IT SO WE CAN RUN THE SAME Q VALUES MULTIPLE TIMES AND AVERAGE THEM TOGETHER.
            step_size = 1 / race_counter###todo: what should this be?
            algo.update_q(q_matrix, trajectory, step_size)
            trajectory = []
            episode_time = 0
            save_info(output_file_path, race_counter, car, total_race_reward)  # NOTE: Saved reward is that of the whole race and not a single episode.
            car.reset_car()
            race_counter += 1
            print("race counter:", race_counter)
            total_race_reward = 0

print("hi")

car = VehicleManager()
run(car)