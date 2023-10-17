######1########
# representation of the matrix
# ex of a matrix, we pick 0 to show the empty space where another cell can move
# matrix = [[8,6,7],[2,5,4],[0,3,1]]
# for i in range(3):
#     for j in range(3):
#         print(matrix[i][j],end=" ")
#     print()

# Representation of a State
def is_final_state(state):
    return state == [1, 2, 3, 4, 5, 6, 7, 8, 0]


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


def move(state, empty_index, target_index):
    state[empty_index], state[target_index] = state[target_index], state[empty_index]
    return state, target_index


####3####
# Iterative Deepening Depth-First Search (IDDFS) ...stackoverflow comes to aid
def depth_limited_search(state, depth, max_depth, last_moved_index, state_dict=None):
    if depth > max_depth:
        return None, [], -1, state_dict
    if is_final_state(state):
        return state, [], depth, state_dict
    empty_index = state.index(0)
    for target_index in range(9):
        if can_move(empty_index, target_index, last_moved_index):
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
                                                                            last_moved_index, state_dict)  # dfs with depth limit so we go
        # thru all of them
        if result is not None:
            return result, moves_local, found_depth
        max_depth += 1


####4####
# Testing
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
