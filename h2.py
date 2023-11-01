puzzle_with_flag = [
    [(8, 1), (4, 1), (0, 1), (0, 1), (5, 1), (0, 1), (0, 0), (0, 1), (0, 1)],
    [(3, 1), (0, 1), (0, 1), (6, 1), (0, 1), (8, 1), (0, 1), (4, 1), (0, 1)],
    [(0, 1), (0, 1), (0, 0), (4, 1), (0, 1), (9, 1), (0, 1), (0, 1), (0, 0)],
    [(0, 1), (2, 1), (3, 1), (0, 1), (0, 0), (0, 1), (9, 1), (8, 1), (0, 1)],
    [(1, 1), (0, 1), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 1), (4, 1)],
    [(0, 1), (9, 1), (8, 1), (0, 1), (0, 0), (0, 1), (1, 1), (6, 1), (0, 1)],
    [(0, 0), (0, 1), (0, 1), (5, 1), (0, 1), (3, 1), (0, 0), (0, 1), (0, 1)],
    [(0, 1), (3, 1), (0, 1), (1, 1), (0, 1), (6, 1), (0, 1), (0, 1), (7, 1)],
    [(0, 1), (0, 1), (0, 0), (0, 1), (2, 1), (0, 1), (0, 1), (1, 1), (3, 1)]
]


def print_puzzle(puzzle):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print(23 * "-")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            val, flag = puzzle[i][j]
            print(f"{val}", end=" ") if flag % 2 else print(f"\033[94m{val}\033[0m", end=" ")  # blue if even
        print()


print_puzzle(puzzle_with_flag)

# aici variabila var e declarata ca un tuplu de pozitii pentru valoarea cell(i,j) in loc de valoarea acesteia
# valoarea val poate foarte usor sa fie accesata folosind val, flag = puzzle[var[0]][var[1]], unde var are forma (i,j) si val[0] = i

variables = [(i, j) for i in range(9) for j in range(9)]


# defining domains for each variable: Xij is even, 2 4 6 8 -> blue
# Xij is odd, 1 2 3 4 5 6 7 8  -> white

def define_domains(puzzle):
    domains = {}
    for var in variables:
        val, flag = puzzle[var[0]][var[1]]
        if flag:  # accessing dict as list
            domains[var] = tuple(range(1, 10)) if val == 0 else (val, flag)
        else:  # vezi Observatia din enunt, in loc de domeniu 2 4 6 8, se poate face constrangere de la domeniu range(1,10) cat val din range sa fie para BUT IM LAZY SO I WONT DO IT
            domains[var] = tuple(range(2, 10, 2)) if val == 0 else (val, flag)
    return domains


def add_constraint(var):
    constraints[var] = []
    for i in range(9):
        if i != var[0]:
            constraints[var].append((i, var[1]))
        if i != var[1]:
            constraints[var].append((var[0], i))
    for i in range((var[0] // 3) * 3, (var[0] // 3) * 3 + 3):
        for j in range((var[1] // 3) * 3, (var[1] // 3) * 3 + 3):
            if (i, j) != var:
                constraints[var].append((i, j))
    if var[0] % 2 == 0 and var[1] % 2 == 0:
        constraints[var].append("grey")


constraints = {}
for i in range(9):
    for j in range(9):
        add_constraint((i, j))

domains = define_domains(puzzle_with_flag)

for var in variables:
    print(var, domains[var])

print_puzzle(puzzle_with_flag)

isComplete = lambda puzzle: all(puzzle[i][j][0] != 0 for i in range(9) for j in
                                range(9))  # puzzle is complete if all the cells have a value different from 0

next_unassigned_variable = lambda puzzle: min((i, j) for i in range(9) for j in range(9) if puzzle[i][j][
    0] == 0)  # next_unassigned_variable returns the first cell with value 0


def backtracking_with_forward_checking(puzzle, domains, constraints):
    if isComplete(puzzle):
        return puzzle

    var = next_unassigned_variable(puzzle)
    original_domain = domains.copy()
    original_flag = puzzle[var[0]][var[1]][1]
    for val in domains[var]:
        if isConsistent(puzzle, var, val, constraints):

            puzzle[var[0]][var[1]] = (val, original_flag)

            # Create a new copy of the domains dictionary
            new_domains = domains.copy()

            # Update domains for all related variables
            new_domains = update_domains(puzzle, new_domains, constraints, var, val)

            result = backtracking_with_forward_checking(puzzle, new_domains, constraints)

            if result is not None:
                return result

            # If result is None, backtrack
            puzzle[var[0]][var[1]] = (0, original_flag)

        # Reset domains to the original dictionary for the next iteration
        domains = original_domain.copy()

    return None


def isConsistent(puzzle, var, val, constraints):
    row, col = var

    # Check row and column constraints
    for i in range(9):
        if puzzle[row][i][0] == val or puzzle[i][col][0] == val:
            print(f"Row/Column constraint violated at {var} for value {val}")
            return False

    # Check box constraints
    box_start_row, box_start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(box_start_row, box_start_row + 3):
        for j in range(box_start_col, box_start_col + 3):
            if puzzle[i][j][0] == val:
                print(f"Box constraint violated at {var} for value {val}")
                return False

    # Check grey constraint
    if constraints[var] == "grey" and val % 2 != 0:
        print(f"Grey constraint violated at {var} for value {val}")
        return False

    return True


def update_domains(puzzle, domains, constraints, var,
                   val):  # update_domains returns the new domains for the variables that are constrained by cell(i,j)
    for item in constraints[var]:
        if item == "grey":
            # Handle "grey" constraint
            if var[0] % 2 == 0 and var[1] % 2 == 0 and val % 2 != 0:
                continue  # Skip this iteration for "grey" constraint
        else:
            i, j = item
            if puzzle[i][j][0] == 0:
                if val in domains[(i, j)]:
                    domains[(i, j)] = tuple(x for x in domains[(i, j)] if x != val)  # the value of cell(i,j) is removed from the domain of the variables that are constrained by cell(i,j)
    return domains


# 3 backtracking with forward checking and mrv heuristic
def next_unassigned_variable_MRV(puzzle, domains):
    unassigned_vars = [var for var in variables if puzzle[var[0]][var[1]][0] == 0]
    return min(unassigned_vars, key=lambda var: len(domains[var]))



def update_domains_FC(domains, var, value):
    new_domains = domains.copy()

    for item in constraints[var]:
        if item == "grey":
            if var[0] % 2 == 0 and var[1] % 2 == 0 and value % 2 != 0:
                continue
        else:
            i, j = item
            if new_domains[(i, j)] and value in new_domains[(i, j)]:
                new_domains[(i, j)] = tuple(x for x in new_domains[(i, j)] if x != value)

    return new_domains


def BKT_with_FC_MRV(assignment, domains):
    if isComplete(assignment):
        return assignment

    var = next_unassigned_variable_MRV(assignment, domains)

    for value in domains[var]:
        if isConsistent(assignment, var, value, constraints):
            new_assignment = [row[:] for row in assignment]
            original_flag = assignment[var[0]][var[1]][1]
            new_assignment[var[0]][var[1]] = (value, original_flag)
            new_domains = update_domains_FC(domains, var, value)

            empty_domain = False
            for var in variables:
                if assignment[var[0]][var[1]][0] == 0 and not new_domains[var]:
                    empty_domain = True
                    break

            if not empty_domain:
                result = BKT_with_FC_MRV(new_assignment, new_domains)
                if result is not None:
                    return result

    return None


solution_fc = backtracking_with_forward_checking(puzzle_with_flag, domains, constraints)

if solution_fc is not None:
    print("2.Solution found:")
    print_puzzle(solution_fc)
else:
    print("No solution found.")


solution_mrv = BKT_with_FC_MRV(puzzle_with_flag, domains)

if solution_mrv is not None:
    print("3.Solution found with MRV:")
    print_puzzle(solution_mrv)
else:
    print("No solution found with MRV.")
