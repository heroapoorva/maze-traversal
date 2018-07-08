import os
import time
import math
import random

class AgentRealistic:

    def __init__(self, alpha, gamma, epsilon):

        self.map_dimensions = 0
        self.training = True
        self.q_table = {}
        self.prev_a = None
        self.prev_s = None

        self.alpha = alpha # 0.1
        self.gamma = gamma # 0.9
        self.epsilon = epsilon # 0.01

    def state_string_to_point(self, state_string):
        current_x = int(state_string.split("_")[1])
        current_y = int(state_string.split("_")[2])
        return current_x, current_y

    def state_to_string(self, x, y):
        return "S_%s_%s" % (str(x), str(y))

    def is_valid_position(self, x, y):
        if 0 <= x < self.map_dimensions and 0 <= y < self.map_dimensions:
            return True
        return False

    def find_state(self, basic_grid, state):
        for x in range(0, self.map_dimensions):
            for y in range(0, self.map_dimensions):
                if basic_grid[x][y] == state:
                    return self.state_to_string(x, y)
        return "NOT FOUND"

    def is_passable(self, basic_grid, x, y):
        if not self.is_valid_position(x, y):
            return False
        block_name = basic_grid[x][y]
        if block_name == u'glowstone' or block_name == u'redstone_block' or block_name == u'emerald_block':
            return True
        return False

    def find_passable_neighbors(self, basic_grid, x, y):
        neighbors = {}

        current_x = x - 1
        current_y = y
        if self.is_passable(basic_grid, current_x, current_y):
            neighbors[self.state_to_string(current_x, current_y)] = 1

        current_x = x + 1
        current_y = y
        if self.is_passable(basic_grid, current_x, current_y):
            neighbors[self.state_to_string(current_x, current_y)] = 1

        current_x = x
        current_y = y - 1
        if self.is_passable(basic_grid, current_x, current_y):
            neighbors[self.state_to_string(current_x, current_y)] = 1

        current_x = x
        current_y = y + 1
        if self.is_passable(basic_grid, current_x, current_y):
            neighbors[self.state_to_string(current_x, current_y)] = 1

        return neighbors

    def calculate_action(self, current_state, next_state):
        current_x = int(current_state.split("_")[1])
        current_y = int(current_state.split("_")[2])
        next_x = int(next_state.split("_")[1])
        next_y = int(next_state.split("_")[2])

        if next_y < current_y:
            return "movenorth 1"
        elif next_y > current_y:
            return "movesouth 1"
        elif next_x < current_x:
            return "movewest 1"
        elif next_x > current_x:
            return "moveeast 1"

    def set_map_dimensions(self, grid):
        self.map_dimensions = int(math.sqrt(len(grid)))

    def read_basic_grid(self, grid):
        basic_grid = [""] * self.map_dimensions
        idx = 0
        y_iter = range(0, self.map_dimensions)
        # y_iter.reverse()

        for x in range(0, self.map_dimensions):
            basic_grid[x] = [""] * self.map_dimensions

        for y in y_iter:
            for x in range(0, self.map_dimensions):
                basic_grid[x][y] = grid[idx]
                idx += 1

        return basic_grid

    def get_possible_actions_from_grid(self, grid, xpos, ypos):

        self.set_map_dimensions(grid)
        basic_grid = self.read_basic_grid(grid)

        passable_neighbors = self.find_passable_neighbors(basic_grid, xpos, ypos)
        current_state = self.state_to_string(xpos, ypos)

        possible_actions = []
        for neighbor in passable_neighbors:
            action = self.calculate_action(current_state, neighbor)
            possible_actions.append(action)

        return possible_actions

    def get_next_action(self, grid, xpos, zpos, current_reward):
        current_r = current_reward
        possible_actions = self.get_possible_actions_from_grid(grid, xpos, zpos)

        current_s = "%d:%d" % (int(xpos), int(zpos))
        if not self.q_table.has_key(current_s):
            self.q_table[current_s] = ([0] * len(possible_actions))

        if self.training and self.prev_s is not None and self.prev_a is not None:
            old_q = self.q_table[self.prev_s][self.prev_a]
            self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * \
                                                             (current_r + self.gamma * max(
                                                                 self.q_table[current_s]) - old_q)

        rnd = random.random()
        action_index = None
        next_action = ""
        if rnd < self.epsilon:
            action_index = random.randint(0, len(possible_actions) - 1)
            log ( "Random action: %s" % possible_actions[action_index] )
        else:
            m = max(self.q_table[current_s])
            log ( "Current values: %s" % ",".join(str(x) for x in self.q_table[current_s])  )
            l = list()
            for x in range(0, len(possible_actions)):
                if x == len(self.q_table[current_s]):
                    self.q_table[current_s].append(0)
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l) - 1)
            action_index = l[y]
            log( "Taking q action: %s" % possible_actions[action_index] )

        self.prev_s = current_s
        self.prev_a = action_index

        next_action = possible_actions[action_index]

        log("Actual Action: %s" % next_action)
        return next_action


logging = False


def log(msg):
    if logging:
        print msg

def emulate_action(action, current_x, current_y):

    if action == "movenorth 1":
        return current_x, current_y - 1

    if action == "movesouth 1":
        return current_x, current_y + 1

    if action == "movewest 1":
        return current_x - 1, current_y

    if action == "moveeast 1":
        return current_x + 1, current_y

    return -1, -1


def run_training(alpha, gamma, epsilon):

    avg_all_mission_rewards = []
    best_all_mission_rewards = []

    for mission_seed in range(100):

        agent = AgentRealistic(alpha, gamma, epsilon)
        script_dir = os.path.join(__file__, os.pardir)
        grids_dir = os.path.realpath(os.path.join(script_dir, "grids"))
        grid_name = os.path.join(grids_dir, "grid_%s.json" % str(mission_seed))

        grid = []
        with open(grid_name, "r") as grid_file:
            grid_str = grid_file.read().replace("[", "").replace("]", "").replace("u'", "").replace("'", "").replace(" ", "")
            grid_split = grid_str.split(",")
            grid = grid_split

        num_times_to_run = 60
        mission_rewards = []
        best_reward = 0.0
        for run_num in range(num_times_to_run):
            agent.set_map_dimensions(grid)
            basic_grid = agent.read_basic_grid(grid)

            initial_state = agent.find_state(basic_grid, u'emerald_block')
            goal_state = agent.find_state(basic_grid, u'redstone_block')

            current_x, current_y = agent.state_string_to_point(initial_state)
            goal_x, goal_y = agent.state_string_to_point(goal_state)

            current_reward = -6.0
            total_reward = 0.0
            goal_found = False

            num_actions = 300
            for _ in range(num_actions):
                next_action = agent.get_next_action(grid, current_x, current_y, current_reward)
                current_x, current_y = emulate_action(next_action, current_x, current_y)

                if current_reward == 1000.0:
                    total_reward += current_reward
                    goal_found = True
                    break

                if current_x == goal_x and current_y == goal_y:
                    current_reward = 1000.0
                else:
                    total_reward += current_reward

                log( "X: %d, Y: %d" % (current_x, current_y) )
                log( "Reward so far: %f" % total_reward )

            if total_reward > best_reward:
                best_reward = total_reward
            mission_rewards.append(total_reward)
            #print( "Scenario Seed: %s, Run #%s, Total Reward: %s, Goal Found: %s" % (str(mission_seed), str(run_num), str(total_reward), str(goal_found)) )

        mission_reward_average = sum(mission_rewards) / float(len(mission_rewards))
        #print( "Scenario Seed: %s, Average Reward: %s, Best Reward %s" % (str(mission_seed), str(mission_reward_average), str(best_reward)) )

        avg_all_mission_rewards.append(mission_reward_average)
        best_all_mission_rewards.append(best_reward)

    total_average = sum(avg_all_mission_rewards) / float(len(avg_all_mission_rewards))
    best_reward_average = sum(best_all_mission_rewards) / float(len(best_all_mission_rewards))

    return total_average, best_reward_average


def float_range(start, end, increment):
    i = start
    while i < end:
        yield i
        i += increment


def main():

    start_time = time.time()

    best_alpha = 0.0
    best_gamma = 0.0
    best_epsilon = 0.0
    best_average = 0.0

    #for alpha in float_range(1.0, 1.01, 0.05):
    #   for gamma in float_range(0.51, 0.511, 0.01):
    #       for epsilon in float_range(0.01, 0.011, 0.01):
    for alpha in float_range(0.0, 1.0, 0.1):
        for gamma in float_range(0.0, 1.0, 0.1):
            for epsilon in float_range(0.01, 0.1, 0.01):

                _, total_average = run_training(alpha, gamma, epsilon)

                if total_average > best_average:
                    best_average = total_average
                    best_alpha = alpha
                    best_gamma = gamma
                    best_epsilon = epsilon

                print "Alpha %s, Gamma %s, Epsilon %s. Total Reward Average: %s" % (str(alpha), str(gamma), str(epsilon), str(total_average))
                print "Best Alpha: %s | Best Gamma: %s | Best Epsilon: %s | Best Total Reward Average: %s" % (str(best_alpha), str(best_gamma), str(best_epsilon), str(best_average))

    elapsed_time = time.time() - start_time
    print "Elapsed Time: %s" % str(elapsed_time)


if __name__ == '__main__':
    main()
