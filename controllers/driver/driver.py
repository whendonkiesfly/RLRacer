import itertools
import json
import pickle
import numpy as np
import random
from vehicle import Driver






#############################################################################
##########################State/Action Management############################
#############################################################################

class VehicleState:
    """
    Stores all relevant car and race related stateful information necessary.
    Also has the ability to generated the reduced "minimized state" which only
    has the information required by the RL algorithm.
    """

    LIDAR_ENUMS = (0.25, 1, 5)  # Enumerated LIDAR values.

    def __init__(self, lidar_vals, crossing_checkpoint, crashed, checkpoint_count, crossed_finishline, timed_out):
        self.lidar_vals = tuple(self.quantize_val(v, self.LIDAR_ENUMS) for v in lidar_vals)
        self.crashed = crashed
        self.crossing_checkpoint = crossing_checkpoint
        self.checkpoint_count = checkpoint_count
        self.crossed_finishline = crossed_finishline
        self.timed_out = timed_out

    def quantize_val(self, val, enums):
        """
        Returns the value in enums that is nearest to val
        """
        return min(enums, key=lambda enum_val: abs(val-enum_val))

    def minimized_state(self, include_checkpoint):
        """
        Returns the minimized state which can be used by the RL algorithm.
        """
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
        """
        Returns True if the minimized state value includes the checkpoint in the state and False otherwise.
        """
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
        Generator function for getting all possible minimized states given the lidar
          laser count and potentially the checkpoint count.
        If checkpoint_count is None, checkpoint isn't included in the states.
        """
        lidar_iterator = itertools.product(self.LIDAR_ENUMS, repeat=lidar_laser_count)
        if checkpoint_count is None:
            #Checkpoint count not stored in the state.
            for lt in lidar_iterator:
                yield self.format_minimized_state(lt)
        else:
            #Checkpoint count stored in the state.
            for lt in lidar_iterator:
                for i in range(checkpoint_count + 1):
                    yield self.format_minimized_state(lt, i)


ALLOWED_ACTIONS = [-0.6, -0.2, 0, 0.2, 0.6]
def get_random_action():
    """
    Returns a random aciton from ALLOWED_ACTIONS
    """
    return random.choice(ALLOWED_ACTIONS)






#############################################################################
#######################Environment Management Classes########################
#############################################################################

class CheckpointData:
    """
    This class stores information about a checkpoint and can determine whether a point is within the x, y bounds of the checkpoint.
    """
    def __init__(self, dims, pose, color_field):
        #Calculate 3 corners of a rectangle such that the vector from the first to the second point is
        # perpendicular to the vector from the second point to the third. z values don't matter because we'll throw them out.
        shape_coords = np.array((
            (-dims[0]/2, dims[1]/2, 0, 1),
            (dims[0]/2, dims[1]/2, 0, 1),
            (dims[0]/2, -dims[1]/2, 0, 1),
        ))

        shape_coords = [pose @ coord for coord in shape_coords]
        #Throw out the z coord and the trailing 1.
        shape_coords = [coord[0:2] for coord in shape_coords]
        self.rect_coords = shape_coords
        self.color_field = color_field

    def set_cp_color(self, rgb):
        """
        Sets the color of the checkpoint.
        rgb should be in format (r, g, b) where each color channel is between 0 and 1.
        """
        self.color_field.setSFColor(list(rgb))
    
    def _make_vec2(self, p1, p2):
        """
        Converts two points to a vector.
        """
        return np.array([p2[i] - p1[i] for i in range(2)])

    def contains_point(self, point):
        #Made with help from this post. https://stackoverflow.com/a/2763387
        AB = self._make_vec2(self.rect_coords[0], self.rect_coords[1])
        BC = self._make_vec2(self.rect_coords[1], self.rect_coords[2])
        AM = self._make_vec2(self.rect_coords[0], point)
        BM = self._make_vec2(self.rect_coords[1], point)
        return 0 <= AB @ AM <= AB @ AB and 0 <= BC @ BM <= BC @ BC


class VehicleManager:
    """
    Class for interacting with a Webots vehicle and environment.
    """
    
    CAR_DEF = "the_car"  # def value of the vehicle in Webots
    CHECKPOINT_GROUP_DEF = "CHECKPOINTS"  # def value of a group containing ordered checkpoints in Webots.

    def __init__(self):
        self.checkpoint_count = 0
        self.crashed = False
        self.checkpoints = None

        #Connect to the car.
        print("Connecting...")
        self.driver = Driver()
        print("Connected.")
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

        self._load_checkpoint_info()


    def _load_checkpoint_info(self):
        """
        Retrieves all checkpoints from the checkpoint group.
        """
        checkpoint_group = self.driver.getFromDef(self.CHECKPOINT_GROUP_DEF)
        checkpoint_group_values = checkpoint_group.getField("children")
        checkpoint_count = checkpoint_group_values.getCount()
        self.checkpoints = []
        for i in range(checkpoint_count):
            cp_solid = checkpoint_group_values.getMFNode(i)
            cp_pose = cp_solid.getPose()
            cp_pose = np.array(cp_pose).reshape((4,4))
            cp_shape = cp_solid.getField("children").getMFNode(0)  # NOTE: THIS ASSUMES THAT SHAPE IS THE FIRST CHILD OF THE SOLID!
            dims = cp_shape.getField("geometry").getSFNode().getField("size").getSFVec3f()
            appearance_obj = cp_shape.getField("appearance") # NOTE: THIS ASSUMES THE APPEARANCE IS A PBRAppearance object.
            color_field = appearance_obj.getSFNode().getField("baseColor")
            cp_data = CheckpointData(dims, cp_pose, color_field)
            self.checkpoints.append(cp_data)

    def get_state(self, timed_out):
        """
        Returns a VehicleState object for the current state of the vehicle and race.
        """
        lidar_value = self.lidar_sensor.getRangeImage()
        if self.check_for_collisions():
            self.crashed = True

        #Check to see if we entered the next checkpoint.
        crossing_checkpoint = False
        if self.checkpoint_count < len(self.checkpoints):
            next_cp = self.checkpoints[self.checkpoint_count]
            gps_coord = self.gps_sensor.getValues()
            if next_cp.contains_point(gps_coord):
                next_cp.set_cp_color((0, 1, 0))
                self.checkpoint_count += 1
                crossing_checkpoint = True
        # else:
        #     print("Getting state post-race")
        
        crossed_finishline = self.checkpoint_count == len(self.checkpoints)
        return VehicleState(lidar_value, crossing_checkpoint, self.crashed, self.checkpoint_count, crossed_finishline, timed_out)

    def execute_action(self, angle):
        """
        Takes a steering angle and updates the vehicle with that angle and full speed.
        """
        self.driver.setSteeringAngle(angle)
        self.driver.setCruisingSpeed(1.5)

    def reset_car(self):
        """
        Resets the race to the original conditions.
        """
        self.trans_field.setSFVec3f(self.starting_trans)
        self.rotation_field.setSFRotation(self.starting_rotation)
        self.car_node.resetPhysics()
        #Reset checkpoint colors.
        for checkpoint in self.checkpoints:
            checkpoint.set_cp_color((1, 0, 0))
        self.checkpoint_count = 0
        self.crashed = False

    def check_for_collisions(self):
        """
        Returns True if the vehicle is colliding with anything in the environment and False otherwise.
        """
        collisions = self.car_node.getContactPoints()
        return len(collisions) > 0
    
    def step(self):
        """
        Takes one simulator step.
        """
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
        

def load_q(path, include_cp_in_state, checkpoint_count=None):
    with open(path, "rb") as fin:
        q_matrix = pickle.load(fin)

    q_includes_checkpoint = VehicleState.state_includes_cp(list(q_matrix.keys())[0])

    if q_includes_checkpoint and not include_cp_in_state:
        raise ValueError("Q matrix downgrading now supported!")
    elif include_cp_in_state and not q_includes_checkpoint:
        #Need to upgrade.
        if checkpoint_count is None:
            raise ValueError("Checkpoint count must be included when using checkpoints in the state.")
        print("upgrading")
        q_matrix = upgrade_q(q_matrix, checkpoint_count)

    return q_matrix






#############################################################################
###################################Helpers###################################
#############################################################################

def calculate_reward(state, action, crash_penalty=5, finishline_bonus=3, timeout_penalty=5):
    """
    Reward is 1 for crossing a checkpoint with a finishline_bonus multiplier if the checkpoint is the finish line.
    Reward is taken if the car crashed.
    """
    return int(state.crossing_checkpoint) * (1 + finishline_bonus * int(state.crossed_finishline)) - \
                                (crash_penalty * int(state.crashed)) - (timeout_penalty * int(state.timed_out))



def pull_action_from_q(q_matrix, state, epsilon, include_checkpoint_in_state):
    """
    Uses epsilon greedy selection of a state from a q matrix given the current state.
    """
    #If this isn't a minimized state, convert to that.
    if type(state) is VehicleState:
        state = state.minimized_state(include_checkpoint_in_state)
    if random.random() < epsilon:
        #Pick a random state.
        action = get_random_action()
    else:
        #Use greedy selection of the next action.
        action = max(q_matrix[state].keys(), key=lambda a: q_matrix[state][a])
    return action


def generate_random_q(lidar_laser_count, checkpoint_count=None):
    """
    Creates a random q matrix given the lidar laser count and checkpoint count.
    If checkpoint_count is None, checkpoints are not included in the states.
    """
    q = {}
    for state in VehicleState.generate_all_minimized_states(lidar_laser_count, checkpoint_count):
        q[state] = {action: random.random() for action in ALLOWED_ACTIONS}
    return q


def upgrade_q(q, checkpoint_count):
    """
    Takes a q matrix that was made without checkpoints being part of states and returns a new q matrix with checkpoints in the states.
    """
    new_q = {}
    for state in q.keys():
        for i in range(checkpoint_count + 1):
            new_state = VehicleState.format_minimized_state(state[0], i)
            new_q[new_state] = {}
            for action in q[state].keys():
                new_q[new_state][action] = q[state][action]
    return new_q



class CycleInfo:
    """
    Stores all information an RL algorithm may need in its get_next_action step
    """
    def __init__(self, state, trajectory, fully_greedy):
        self.state = state
        self.trajectory = trajectory
        self.fully_greedy = fully_greedy


class TrajectoryTriplet:
    """
    Stores a state, the action taken at that state, and the received reward.
    """
    def __init__(self, state: VehicleState, action: float, reward: float):
        self.state = state
        self.action = action
        self.reward = reward






#############################################################################
################################RL Algorithms################################
#############################################################################

class SarsaAlgorithm:
    """
    Implements the SARSA algorithm.
    """

    def __init__(self, step_size, epsilon, discount_factor, include_checkpoint_in_state, initial_q):
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.include_checkpoint_in_state = include_checkpoint_in_state
        self.initial_q = initial_q
        self.q_matrix = initial_q


    def get_next_action(self, cycle_info:CycleInfo):

        epsilon = self.epsilon if not cycle_info.fully_greedy else 0

        if not cycle_info.trajectory:  # If we have no trajectory yet (first cycle), just pull an action from Q.
            action = pull_action_from_q(self.q_matrix, cycle_info.state, epsilon, self.include_checkpoint_in_state)
            return action
        else:
            #This isn't the first state, we need to update Q using info from the last cycle.
            #Gather info.
            sar = cycle_info.trajectory[-1]
            current_S = cycle_info.state.minimized_state(self.include_checkpoint_in_state)
            previous_S = sar.state.minimized_state(self.include_checkpoint_in_state)
            previous_A = sar.action
            previous_R = sar.reward

            #Choose A' from S' using Q
            next_action = pull_action_from_q(self.q_matrix, current_S, epsilon, self.include_checkpoint_in_state)

            #Update q.
            self.q_matrix[previous_S][previous_A] += self.step_size * (previous_R + self.q_matrix[current_S][next_action] - self.q_matrix[previous_S][previous_A])

            return next_action

    def update_q(self, final_state, trajectory):
        cycle_info = CycleInfo(final_state, trajectory, False)
        self.get_next_action(cycle_info)

    
class MCAlgorithm:
    """
    Algorithm similar to first visit MC except that all visits are used for updating q.
    """
    def __init__(self, step_size, epsilon, discount_factor, include_checkpoint_in_state, initial_q, first_visit):
        """
        If first_visit is True, this runs the first-visit MC algorithm.
        """
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.include_checkpoint_in_state = include_checkpoint_in_state
        self.q_matrix = initial_q
        self.first_visit = first_visit

    def get_next_action(self, cycle_info:CycleInfo):
        state = cycle_info.state
        epsilon = self.epsilon if not cycle_info.fully_greedy else 0
        return pull_action_from_q(self.q_matrix, cycle_info.state, epsilon, self.include_checkpoint_in_state)


    def update_q(self, final_state, trajectory):
        #Iterate backwards through the trajectory and calculate the reward for each state.
        #Go backwards so the reward is attributed to the first instance of the state.
        state_rewards = {}
        reward_sum = 0

        #Iterate backwards over the trajectory tracking the reward sum.
        for trajectory_point in reversed(trajectory):
            reward_sum = (reward_sum * self.discount_factor) + trajectory_point.reward
            state = trajectory_point.state.minimized_state(self.include_checkpoint_in_state)
            if state not in state_rewards:
                state_rewards[state] = {}
            
            if self.first_visit:
                #We are doing the first visit algorithm. Update the state, action pair reward value with the new one.
                state_rewards[state][trajectory_point.action] = reward_sum
            else:
                #We are not doing the first visit algorithm. Maintain a list of all rewards for each state, action pair.
                if trajectory_point.action not in state_rewards[state]:
                    state_rewards[state][trajectory_point.action] = []
                state_rewards[state][trajectory_point.action].append(reward_sum)


        #If we aren't doing the first visit algorithm, average all of the samples together.
        if not self.first_visit:
            for state in state_rewards.keys():
                for action in state_rewards[state].keys():
                    rewards = state_rewards[state][action]
                    state_rewards[state][action] = sum(rewards) / len(rewards)

        #Update the q matrix using the state rewards.
        for state in state_rewards.keys():
            for action in state_rewards[state].keys():
                # print("updating q", state, action, state_rewards[state][action])
                self.q_matrix[state][action] += self.step_size * (state_rewards[state][action] - self.q_matrix[state][action])








def run(car, algo, output_base_path, termination_value=1000, max_checkpoint_time=10):
    """
    Runs the main control loop for the RL algorithm.
    Finishes after termination_value races have been completed.
    """

    trajectory = []
    race_time = 0
    timeout_time = max_checkpoint_time
    race_counter = 0

    results_output_path = output_base_path + "_results.txt"

    initialize_file(output_file_path)

    while car.step() != -1:
        #Gather cycle information
        timed_out = (race_time >= timeout_time)
        current_state = car.get_state(timed_out)

        #Choose an action
        fully_greedy = (race_counter == (termination_value - 1))  # Fully greedy (epsilon=0) if we are on the last cycle
        cycle_info = CycleInfo(current_state, trajectory, fully_greedy)
        next_action = algo.get_next_action(cycle_info)

        #Execute the action
        car.execute_action(next_action)

        #Calculate the reward and update the trajectory.
        reward = calculate_reward(current_state, next_action)
        trajectory.append(TrajectoryTriplet(current_state, next_action, reward))

        #Update the race time.
        race_time += car.timestep / 1000

        #If we crossed a checkpoint, give us more time before we timeout.
        if current_state.crossing_checkpoint:
            timeout_time = race_time + max_checkpoint_time

        #Check to see if the race is over.
        race_over = False
        if current_state.timed_out:
            print("timeout!")
            race_over = True
        if current_state.crashed:
            print("crashed!")
            race_over = True
        if current_state.crossed_finishline:
            print("crossed finishline!!")
            race_over = True

        if race_over:
            #Finalize things for this race.
            race_counter += 1
            final_state = car.get_state(timed_out)
            algo.update_q(final_state, trajectory)
            missing_checkpoints = len(car.checkpoints) - car.checkpoint_count
            save_info(results_output_path, race_counter, race_time, missing_checkpoints)  # NOTE: Saved reward is that of the whole race and not a single episode.
            save_q(output_base_path + "_q.pickle", algo.q_matrix)

            #Get ready for the next race.
            trajectory = []
            race_time = 0
            timeout_time = max_checkpoint_time
            car.reset_car()
            print("race counter:", race_counter)
            #Check to see if we are done.
            if race_counter == termination_value:
                #We hit our termination count.
                return

    print("stopped without standard termination")


if __name__ == "__main__":

    car = VehicleManager()

    epsilon = 0.1
    discount_factor = 0.9999
    mc_step_size = 0.025
    sarsa_step_size = 0.25



    cp_count = len(car.checkpoints)
    lidar_laser_count = car.lidar_sensor.getHorizontalResolution()





    include_cp_in_state = False
    output_file_path = "c:\\temp\\rlRacerOut_all_visit1"
    q_matrix = generate_random_q(lidar_laser_count, None)
    first_visit = False
    algo = MCAlgorithm(mc_step_size, epsilon, discount_factor, include_cp_in_state, q_matrix, first_visit)
    run(car, algo, output_file_path, 20)


    # print("LOADING NEW!!!!")

    # q_path = output_file_path + "_q.pickle"
    # include_cp_in_state = True
    # output_file_path = "c:\\temp\\rlRacerOut_all_visit2"
    # q_matrix = load_q(q_path, include_cp_in_state, cp_count)
    # first_visit = False
    # algo = MCAlgorithm(mc_step_size, epsilon, discount_factor, include_cp_in_state, q_matrix, first_visit)
    # run(car, algo, output_file_path, 10)


    # include_cp_in_state = False
    # output_file_path = "c:\\temp\\rlRacerOut"
    # q_matrix = generate_random_q(lidar_laser_count, None)
    # algo = SarsaAlgorithm(mc_step_size, epsilon, discount_factor, include_cp_in_state, q_matrix)
    # run(car, algo, output_file_path)


    print("All tests complete.")





#Set main to run over different parameters. Also sometimes with same parameters.
#new tracks. Mirror first and make a new one
##does the MC algorithm fit with something we learned in class??????