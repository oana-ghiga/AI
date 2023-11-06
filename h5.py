from colorama import Fore, Style

class NumberScrabble:
    def __init__(self):
        self.state = [[2, 7, 6], [9, 5, 1], [4, 3, 8]]
        self.player_picked = set()
        self.computer_picked = set()

    def is_final(self):
        return len(self.player_picked) + len(self.computer_picked) == 9

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

    def computer_move(self):
        available_numbers = set(range(1, 10)) - self.player_picked - self.computer_picked
        if not available_numbers:
            return
        best_move = None
        best_score = float('-inf')
        for number in available_numbers:
            if self.transition(number, 'computer'):
                score = self.heuristic(self.computer_picked)  # Calculate heuristic score after the move
                self.transition(number, 'computer')  # Revert the move
                if score > best_score:
                    best_score = score
                    best_move = number
                    break  # Stop after selecting one move
        if best_move:
            self.transition(best_move, 'computer')
            print("Computer moves:", best_move)

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
        if self.heuristic(self.computer_picked) > self.heuristic(self.player_picked):
            print("Computer wins!")
        elif self.heuristic(self.computer_picked) < self.heuristic(self.player_picked):
            print("You win!")
        else:
            print("It's a draw!")
        print("Game Over!")

game = NumberScrabble()
game.play()
