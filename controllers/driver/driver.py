import itertools
import json
import pickle
import numpy as np
import random
from vehicle import Driver



MAX_CHECKPOINT_TIME = 10



class MainAlgorithm:
    def __init__(self, epsilon, discount_factor, include_checkpoint_in_state):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.include_checkpoint_in_state = include_checkpoint_in_state

    def get_next_action(self, q_matrix, current_state):
        state = current_state.minimized_state(self.include_checkpoint_in_state)
        if random.random() < self.epsilon:
            # print("random action!")
            return get_random_action()
        else:
            # print("best known action")
            try:
                return max(q_matrix[state].keys(), key=lambda a: q_matrix[state][a])
            except Exception as e:
                import pdb; pdb.set_trace()


    def update_q(self, q_matrix, trajectory, step_size, single_state_calculate=False):####TODO: CHECK ME WELL.       ######################TODO: MAKE THIS WORK WITH AN AVERAGED TARGET OVER MULTIPLE EPISODES WITH THE SAME Q VALUES(?)
        #Iterate backwards through the trajectory and calculate the reward for each state.
        #Go backwards so the reward is attributed to the first instance of the state.
        state_rewards = {}
        reward_sum = 0
        for trajectory_point in reversed(trajectory):
            reward_sum = (reward_sum * self.discount_factor) + trajectory_point.reward#############TODO: DISCOUNT FACTOR I THINK!!!
            state = trajectory_point.state.minimized_state(self.include_checkpoint_in_state)
            if state not in state_rewards:
                state_rewards[state] = {}
            
            if single_state_calculate:
                state_rewards[state][trajectory_point.action] = reward_sum
            else:
                if trajectory_point.action not in state_rewards[state]:
                    state_rewards[state][trajectory_point.action] = []
                state_rewards[state][trajectory_point.action].append(reward_sum)


        if not single_state_calculate:
            for state in state_rewards.keys():
                for action in state_rewards[state].keys():
                    rewards = state_rewards[state][action]
                    state_rewards[state][action] = sum(rewards) / len(rewards)##########################TODO: EITHER DOCUMENT WELL OR REMOVE!!!!

        for state in state_rewards.keys():
            for action in state_rewards[state].keys():
                # print("updating q", state, action, state_rewards[state][action])
                try:
                    q_matrix[state][action] += step_size * (state_rewards[state][action] - q_matrix[state][action])
                except KeyError:
                    import pdb; pdb.set_trace()






class VehicleState:
    LIDAR_THRESHOLDS = (0.25, 1, 5)
    def __init__(self, lidar_vals, crossing_checkpoint, crashed, checkpoint_count, crossed_finishline):
        self.lidar_vals = tuple(self.quantize_val(v, self.LIDAR_THRESHOLDS) for v in lidar_vals)
        self.crashed = crashed
        self.crossing_checkpoint = crossing_checkpoint
        self.checkpoint_count = checkpoint_count
        self.crossed_finishline = crossed_finishline

    def quantize_val(self, val, thresholds):
        return min(thresholds, key=lambda thresh: abs(val-thresh))

    def minimized_state(self, include_checkpoint):
        if include_checkpoint:
            return self.format_minimized_state(self.lidar_vals, self.checkpoint_count)
        else:
            return self.format_minimized_state(self.lidar_vals)

    @classmethod
    def format_minimized_state(cls, lidar_vals, checkpoint_count=None):
        if checkpoint_count is not None:
            return (lidar_vals, checkpoint_count)
        else:
            return (lidar_vals, )

    @classmethod
    def state_includes_cp(cls, state):
        if len(state) == 1:
            return False
        elif len(state) == 2:
            return True
        else:
            raise ValueError(f"State with length of {len(state)}?")

    def __str__(self):
        return str(f"lidar: {self.lidar_vals}   crashed: {self.crashed}   checkpoint_count: {self.checkpoint_count}")

    @classmethod
    def generate_all_minimized_states(self, lidar_laser_count, checkpoint_count=None):
        """
        if checkpoint_count is None, checkpoint isn't included in the states.
        """
        lidar_iterator = itertools.product(self.LIDAR_THRESHOLDS, repeat=lidar_laser_count)
        if checkpoint_count is None:
            for lt in lidar_iterator:
                yield self.format_minimized_state(lt)
        else:
            for lt in lidar_iterator:
                for i in range(checkpoint_count + 1):
                    yield self.format_minimized_state(lt, i)



ALLOWED_ACTIONS = [-0.4, -0.2, 0, 0.2, 0.4]###TODO: IS 0.4 ENOUGH? GO TO MAX.
def get_random_action():
    return random.choice(ALLOWED_ACTIONS)




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



class VehicleManager:
    
    CAR_DEF = "the_car"
    CHECKPOINT_GROUP_DEF = "CHECKPOINTS"

    def __init__(self):
        self.checkpoint_count = 0
        self.crashed = False
        self.checkpoints = None

        #Connect to the car.
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())

        self.car_node = self.driver.getFromDef(self.CAR_DEF)


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
        checkpoint_group = self.driver.getFromDef(self.CHECKPOINT_GROUP_DEF)
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
        if self.check_for_collisions():
            print("crashed!!!")
            self.crashed = True

        #Check to see if we entered the next checkpoint.
        crossing_checkpoint = False
        if self.checkpoint_count < len(self.checkpoints):
            next_cp = self.checkpoints[self.checkpoint_count]####todo: will this run off the edge after hitting the last checkpoint???
            gps_coord = self.gps_sensor.getValues()
            if next_cp.contains_point(gps_coord):
                next_cp.set_cp_color((0, 1, 0))
                self.checkpoint_count += 1
                crossing_checkpoint = True
        else:
            print("final checkpoint reached but race didn't finish?")
        
        crossed_finishline = self.checkpoint_count == len(self.checkpoints)
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

    def check_for_collisions(self):
        collisions = self.car_node.getContactPoints()
        return len(collisions) > 0
    
    def step(self):
        return self.driver.step()






#############################################################################
##############################File IO Functions##############################
#############################################################################

def initialize_file(path):
    #Just open the file handle to clear the file if it is already populated.
    with open(path, "w") as fout:
        pass


def save_info(path, iteration_count, race_time, missing_checkpoints):
    with open(path, "a") as fout:
        fout.write(json.dumps({"iteration_count": iteration_count,
                                "race_time": race_time,
                                "missing_checkpoints": missing_checkpoints}) + "\n")


def save_q(path, q):
    with open(path, "wb") as fout:
        pickle.dump(q, fout)
        

def load_q(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)







def calculate_reward(state, action, crash_penalty=5, finishline_bonus=3):
    """
    Reward is 1 for crossing a checkpoint with a finishline_bonus multiplier if the checkpoint is the finish line.
    Reward is taken if the car crashed.
    """
    return int(state.crossing_checkpoint) * (1 + finishline_bonus * int(state.crossed_finishline)) - (crash_penalty * int(state.crashed))



def generate_random_q(lidar_laser_count, checkpoint_count=None):
    q = {}
    for state in VehicleState.generate_all_minimized_states(lidar_laser_count, checkpoint_count):
        q[state] = {action: random.random() for action in ALLOWED_ACTIONS}
    return q


def upgrade_q(q, checkpoint_count):################TODO: RANDOMNESS HERE????
    """
    Takes a q matrix that was made without checkpoints being part of states and returns a new q matrix with checkpoints in the states.
    """
    new_q = {}
    for state in new_q.keys():
        for action in new_q[state].keys():
            for i in range(checkpoint_count + 1):
                if i == 0:
                    new_q[new_state] = {}
                new_state = VehicleState.format_minimized_state(state[0], i)
                new_q[new_state][action] = q[state][action]









def run(car, output_base_path, include_cp_in_state, epsilon, input_q_path=None):

    lidar_laser_count = car.lidar_sensor.getHorizontalResolution()

    if include_cp_in_state:
        checkpoint_count = len(car.checkpoints)
    else:
        checkpoint_count = None


    if input_q_path:
        q_matrix = load_q(input_q_path)
        q_includes_checkpoint = VehicleState.state_includes_cp(list(q_matrix.keys())[0])
        if q_includes_checkpoint and not include_cp_in_state:
            raise ValueError("Q matrix downgrading now supported!")
        elif include_cp_in_state and not q_includes_checkpoint:
            #Need to upgrade.
            q_matrix = upgrade_q(q_matrix, checkpoint_count)
    else:
        q_matrix = generate_random_q(lidar_laser_count, checkpoint_count)


    race_counter = 1
    discount_factor = 0.99

    algo = MainAlgorithm(epsilon, discount_factor, include_cp_in_state)
    trajectory = []
    race_time = 0
    timeout_time = MAX_CHECKPOINT_TIME

    results_output_path = output_base_path + "_results.txt"

    initialize_file(output_file_path)

    while car.step() != -1:

        current_state = car.get_state()
        #Choose an action
        next_action = algo.get_next_action(q_matrix, current_state)
        car.execute_action(next_action)

        reward = calculate_reward(current_state, next_action)

        trajectory.append(TrajectoryTriplet(current_state, next_action, reward))

        race_time += car.timestep / 1000

        #If we crossed a checkpoint, give us more time before we timeout.
        if current_state.crossing_checkpoint:
            timeout_time = race_time + MAX_CHECKPOINT_TIME

        race_over = False
        if race_time >= timeout_time:##todo: change to episode time?
            print("timeout!")
            race_over = True
        if current_state.crashed:
            print("crashed!")
            race_over = True
        if current_state.crossed_finishline:
            print("crossed finishline!!")
            race_over = True

        if race_over:
            # total_race_reward = sum(t.reward for t in trajectory)######################                         TODO: MAKE IT SO WE CAN RUN THE SAME Q VALUES MULTIPLE TIMES AND AVERAGE THEM TOGETHER.
            # step_size = 1 / race_counter###todo: what should this be?
            step_size = 0.025
            missing_checkpoints = len(car.checkpoints) - car.checkpoint_count
            algo.update_q(q_matrix, trajectory, step_size)
            save_info(results_output_path, race_counter, race_time, missing_checkpoints)  # NOTE: Saved reward is that of the whole race and not a single episode.
            save_q(output_base_path + "_q.pickle", q_matrix)
            trajectory = []
            race_time = 0
            timeout_time = MAX_CHECKPOINT_TIME
            car.reset_car()
            race_counter += 1
            print("race counter:", race_counter)


if __name__ == "__main__":
    print("hi")

    car = VehicleManager()
    output_file_path = "c:\\temp\\rlRacerOut"

    include_cp_in_state = True
    epsilon = 0.1
    q_path = "c:\\temp\\rlRacerOut_q.txt"
    run(car,  output_file_path, include_cp_in_state, epsilon, input_q_path=q_path)












#MAJOR TODOS:
#Look over update algorithm. Not sure why it learns quickly but gets stuck. MAYBE PICK A DIFFERENT ALGORITHM
#Look over function to save. Probably should put race time in there or something. Maybe discounted reward. Make it format things better. Probably should clear the file the first time.
#*************Get to where we can switch to new state scheme which includes the number of checkpoints

#####################TODO: REWARD FOR FINISH LINE SHOULD BE HIGHER

###########TODO: MAKE SURE WE CAN SAVE Q! PROBABLY NOT BECAUSE STATES ARE OBJECTS INSTEAD OF JUST NUMBERS!! MAKE THEM JUST NUMBERS PROBABLY!