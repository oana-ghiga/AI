######1########
# representation of the matrix
# ex of a matrix, we pick 0 to show the empty space where another cell can move
# matrix = [[8,6,7],[2,5,4],[0,3,1]]
# for i in range(3):
#     for j in range(3):
#         print(matrix[i][j],end=" ")
#     print()
import heapq
import math


# Representation of a State 0 can be anywhere for the final state as long as the rest of the numbers are in order so
# we have 9 possible final states

def is_final_state(state):
    return state in [[1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 0, 5, 6, 7, 8], [1, 2, 3, 0, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 0, 2, 3, 4, 5, 6, 7, 8], [1, 2, 0, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 0, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 0, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 0, 8]]


# print(is_final_state([1, 2, 3, 4, 5, 6, 7, 8, 0]))
def initialize_puzzle(initial_state):
    return initial_state


# print(initialize_puzzle([1, 2, 3, 4, 5, 6, 7, 8, 0]))

####2####

# Transitions
# see if the cell was moved before and check for a 0 around it
def can_move(empty_index, target_index, last_moved_index):
    adjacent_indices = [[1, 3], [0, 2, 4], [1, 5], [0, 4, 6], [1, 3, 5, 7], [2, 4, 8], [3, 7], [4, 6, 8], [5, 7]]
    return target_index in adjacent_indices[empty_index] and target_index != last_moved_index


# For Tile at Index 0 (Top Left Corner):
# Can Move To: Index 1 (Right) and Index 3 (Down)
#
# For Tile at Index 1 (Top Middle):
# Can Move To: Index 0 (Left), Index 2 (Right), and Index 4 (Down)
#
# For Tile at Index 2 (Top Right Corner):
# Can Move To: Index 1 (Left) and Index 5 (Down)
#
# For Tile at Index 3 (Middle Left):
# Can Move To: Index 0 (Up), Index 4 (Right), and Index 6 (Down)
#
# For Tile at Index 4 (Center):
# Can Move To: Index 1 (Up), Index 3 (Left), Index 5 (Right), and Index 7 (Down)
#
# For Tile at Index 5 (Middle Right):
# Can Move To: Index 2 (Up), Index 4 (Left), and Index 8 (Down)
#
# For Tile at Index 6 (Bottom Left Corner):
# Can Move To: Index 3 (Up) and Index 7 (Right)
#
# For Tile at Index 7 (Bottom Middle):
# Can Move To: Index 4 (Up), Index 6 (Left), and Index 8 (Right)
#
# For Tile at Index 8 (Bottom Right Corner):
# Can Move To: Index 5 (Up) and Index 7 (Left)

def move(state, empty_index, target_index):
    state[empty_index], state[target_index] = state[target_index], state[empty_index]
    return state, target_index


####3####
# Iterative Deepening Depth-First Search (IDDFS)....stackoverflow comes to aid
def depth_limited_search(state, depth, max_depth, last_moved_index, state_dict=None):
    if depth > max_depth:
        return None, [], -1, state_dict
    if is_final_state(state):
        return state, [], depth, state_dict
    empty_index = state.index(0)
    for target_index in range(9):
        if can_move(empty_index, target_index, last_moved_index):
            # chaning the last moved index to the current target index
            last_moved_index = target_index
            new_state, _ = move(state.copy(), empty_index, target_index)
            if tuple(new_state) not in state_dict:
                state_dict[tuple(new_state)] = tuple(state)
                result, moves_local, found_depth, state_dict = depth_limited_search(new_state, depth + 1, max_depth,
                                                                                    target_index, state_dict)
                if result is not None:
                    moves_local.insert(0, target_index)
                    return result, moves_local, found_depth, state_dict
    return None, [], -1, state_dict


def iddfs(initial_state):
    max_depth = 0
    last_moved_index = None
    while True:
        state_dict = {tuple(initial_state): None}
        result, moves_local, found_depth, state_dict = depth_limited_search(initial_state, 0, max_depth,
                                                                            last_moved_index,
                                                                            state_dict)  # dfs with depth limit so we go
        # through all of them
        if result is not None:
            return result, moves_local, found_depth
        max_depth += 1


def hamming_distance(state):
    return len([i for i in range(len(state)) if state[i] != 0 and state[i] != i + 1])


def diagonal_distance(state):
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    distance = 0
    for i, val in enumerate(state):
        if val != 0:
            dx = abs(i // 3 - goal_state.index(val) // 3)
            dy = abs(i % 3 - goal_state.index(val) % 3)
            distance += math.sqrt(2) * min(dx, dy) + abs(dx - dy)
    return distance

# def diagonal_distance(state): goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0] return sum(math.sqrt(2) * min(abs(i // 3 -
# goal_state.index(val) // 3), abs(i % 3 - goal_state.index(val) % 3)) + abs(abs(i // 3 - goal_state.index(val) // 3)
# - abs(i % 3 - goal_state.index(val) % 3)) for i, val in enumerate(state) if val != 0)
#

def manhattan_distance(state):
    def distance(i):
        return 0 if state[i] == 0 else abs(((state[i] - 1) / 3) - (i / 3)) + abs(((state[i] - 1) % 3) - (i % 3))

    return sum(distance(i) for i in range(len(state)))


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements.append((item, priority))
        self.elements.sort(key=lambda x: x[1])

    def get(self):
        if not self.empty():
            return self.elements.pop(0)


def greedy_search(initial_state, heuristic):
    state_dict = {tuple(initial_state): None}
    priority_queue = PriorityQueue()
    priority_queue.put(initial_state, heuristic(initial_state))
    last_moved_index = None

    while not priority_queue.empty():
        state, _ = priority_queue.get()
        if is_final_state(state):
            return state

        empty_index = state.index(0)
        for target_index in range(9):
            if can_move(empty_index, target_index, last_moved_index):
                last_moved_index = target_index
                new_state, _ = move(state.copy(), empty_index, target_index)
                if tuple(new_state) not in state_dict:
                    state_dict[tuple(new_state)] = tuple(state)
                    priority_queue.put(new_state, heuristic(new_state))


####4####
# Testing

def print_iddfs():
    instances = [[8, 6, 7, 2, 5, 4, 0, 3, 1], [2, 5, 3, 1, 0, 6, 4, 7, 8], [2, 7, 5, 0, 8, 4, 3, 1, 6]]
    print("Solving with moves:")
    for instance in instances:
        solution, moves, depth = iddfs(instance)
        if solution is not None:
            print(f"Initial State: {instance}")
            print(f"Solution: {solution}")
            print(f"Moves: {moves}")
            print(f"Depth: {depth}")
            print("--------------")
        else:
            print(f"No solution found for Initial State: {instance}")


def print_greedy():
    instances = [[8, 6, 7, 2, 5, 4, 0, 3, 1], [2, 5, 3, 1, 0, 6, 4, 7, 8], [2, 7, 5, 0, 8, 4, 3, 1, 6]]
    print("Solving with moves:")
    for instance in instances:
        solution = greedy_search(instance, manhattan_distance)
        if solution is not None:
            print(f"Initial State: {instance}")
            print(f"Solution: {solution}")
            print("--------------")
        else:
            print(f"No solution found for Initial State: {instance}")


print_greedy();
