from colorama import Fore, Style
from itertools import combinations

class NumberScrabble:
    def __init__(self):
        self.state = [[2, 7, 6], [9, 5, 1], [4, 3, 8]]
        self.player_picked = set()
        self.computer_picked = set()

    def is_final(self):
        # Check if any player's or computer's 3 numbers add up to 15
        return (self.check_winning_combination(self.player_picked) or
                self.check_winning_combination(self.computer_picked) or
                len(self.player_picked) + len(self.computer_picked) == 9)

    def check_winning_combination(self, picked_numbers):
        # Check all combinations of 3 picked numbers
        for combination in combinations(picked_numbers, 3):
            if sum(combination) == 15:
                return True
        return False

    def transition(self, number, player):
        if 1 <= number <= 9 and number not in self.player_picked and number not in self.computer_picked:
            if player == 'player':
                self.player_picked.add(number)
            elif player == 'computer':
                self.computer_picked.add(number)
            return True
        return False

    def print_board(self):
        for row in range(3):
            for col in range(3):
                number = self.state[row][col]
                if number in self.computer_picked:
                    print(Fore.RED + str(number).zfill(2), end=" ")
                elif number in self.player_picked:
                    print(Fore.BLUE + str(number).zfill(2), end=" ")
                else:
                    print(Fore.WHITE + str(number).zfill(2), end=" ")
            print(Style.RESET_ALL)

    # def computer_move(self):
    #     available_numbers = set(range(1, 10)) - self.player_picked - self.computer_picked
    #     if not available_numbers:
    #         return
    #     best_move = None
    #     best_score = float('-inf')
    #     for number in available_numbers:
    #         if self.transition(number, 'computer'):
    #             score = self.heuristic(self.computer_picked)  # Calculate heuristic score after the move
    #             self.transition(number, 'computer')  # Revert the move
    #             if score > best_score:
    #                 best_score = score
    #                 best_move = number
    #                 break  # Stop after selecting one move
    #     if best_move:
    #         self.transition(best_move, 'computer')
    #         print("Computer moves:", best_move)

    def minimax(self, depth, player):
        if self.is_final():
            if self.check_winning_combination(self.computer_picked):
                return [None, 1]  # computer takes the chicken dinner
            elif self.check_winning_combination(self.player_picked):
                return [None, -1]  # eu winner chicken dinner
            else:
                return [None, 0]  # remiza

        if player == 'computer':
            best = [-1, float('-inf')]
        else:
            best = [-1, float('inf')]

        if depth < 0:
            return [None, 0]
        elif depth == 0:
            return [None, self.heuristic(self.computer_picked)]

        for number in set(range(1, 10)) - self.player_picked - self.computer_picked:
            # simulating moves
            if player == 'computer':
                self.computer_picked.add(number)
            else:
                self.player_picked.add(number)

            score = self.minimax(depth - 1, 'player' if player == 'computer' else 'computer')[1]
            # minimax -> [best move, best score]
            if player == 'computer':
                self.computer_picked.remove(number)
            else:
                self.player_picked.remove(number)

            if player == 'computer' and score > best[1]:
                best = [number, score]
            elif player == 'player' and score < best[1]:
                best = [number, score]

        return best

    def computer_move(self):
        if len(self.player_picked) + len(self.computer_picked) < 9:
            # minimax -> [best move, best score]
            move = self.minimax(2, 'computer')[0]
            if move:
                self.computer_picked.add(move)
                print("Computer moves:", move)

    def player_move(self):
        while True:
            try:
                number = int(input("Enter a number (1-9): "))
                if self.transition(number, 'player'):
                    print("You picked the number", number)
                    break
                else:
                    print("Invalid number. Please pick an available number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def heuristic(self, player_picked):
        magic_square = [[2, 7, 6], [9, 5, 1], [4, 3, 8]]
        final_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        opponent_picked = self.computer_picked if player_picked == self.player_picked else self.player_picked
        avl_nums = final_set - (player_picked | opponent_picked)
        score = 0

        for win_box in magic_square:
            avl_nums_in_win_box = len(set(win_box) - avl_nums)
            if avl_nums_in_win_box == 0:
                continue

            if avl_nums_in_win_box == 1:
                if len(set(win_box) - player_picked) == 2:
                    score += 5
                elif len(set(win_box) - player_picked) == 1:
                    score += 2

        return score

    def play(self):
        while not self.is_final():
            self.print_board()
            self.player_move()
            if self.is_final():
                break
            self.print_board()
            self.computer_move()
        self.print_board()

        if self.check_winning_combination(self.player_picked):
            print("You win!")
        elif self.check_winning_combination(self.computer_picked):
            print("Computer wins!")
        else:
            print("It's a draw!")

        print("Game Over!")

game = NumberScrabble()
game.play()
