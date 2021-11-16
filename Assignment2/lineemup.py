# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time

class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3

    def __init__(self, n=3, b=0, bboard=[], s=3, d1=0, d2=0, t=0.0):
        self.n = n # board size
        self.b = b #number of blocks
        self.bboard = bboard
        self.s = s #winning size
        self.d1 = d1  # Max depth adversial search value
        self.d2 = d2  # Max depth adversial search value
        self.t = t  # Max value of time t
        self.initialize_game()

    def initialize_game(self):
        self.current_state = [['.' for x in range(self.n)] for y in range(self.n)]

        if self.bboard == []:
            for i in range(self.b):
                X = random.randint(0, self.n - 1)
                Y = random.randint(0, self.n - 1)
                while self.current_state[X][Y] == '#':
                    X = random.randint(0, self.n - 1)
                    Y = random.randint(0, self.n - 1)
                self.current_state[X][Y] = '#'
                self.bboard.append([X, Y])
        else:
            for element in self.bboard:
                self.current_state[element[0]][element[1]] = '#'

        # Player X always plays first
        self.player_turn = 'X'

    def draw_board(self):
        print()
        for y in range(self.n):
            for x in range(self.n):
                print(F'{self.current_state[x][y]}', end="")
            print()
        print()

    def is_valid(self, px, py):
        if px not in range(self.n) or py not in range(self.n):
            return False
        elif self.current_state[px][py] != '.':
            return False
        else:
            return True

    def is_end(self):

        # Vertical win
        for i in range(self.n):
            lineWin = 0
            for j in range(self.n - 1):
                if self.current_state[j][i] == '.' or self.current_state[j][i] == '*' \
                        or self.current_state[j][i] != self.current_state[j + 1][i]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.line_size - 1:
                    return self.current_state[j][i]

        # Horizontal win
        for j in range(self.n):
            lineWin = 0
            for i in range(self.n - 1):
                if self.current_state[j][i] == '.' or self.current_state[j][i] == '*' \
                        or self.current_state[j][i] != self.current_state[j][i + 1]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.s - 1:
                    return self.current_state[j][i]

        # Main diagonal win
        # Takes into account off-diagonals (Top half left to right)
        for j in range((self.n + 1) - self.s):
            lineWin = 0
            for i in range(self.n - 1 - j):
                if self.current_state[i][i + j] == '.' or self.current_state[i][i + j] == '*' \
                        or self.current_state[i][i + j] != self.current_state[i + 1][i + j + 1]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.s - 1:
                    return self.current_state[i][i + j]

        # Second diagonal win
        # Takes into account off-diagonals (Top half right to left)
        for j in range((self.n + 1) - self.s):
            lineWin = 0
            for i in range(self.n - 1 - j):
                if self.current_state[i][self.n - 1 - i - j] == '.' \
                        or self.current_state[i][self.n - 1 - i - j] == '*' \
                        or self.current_state[i][self.n - 1 - i - j] \
                        != self.current_state[i + 1][self.n - 1 - (i + 1) - j]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.s - 1:
                    return self.current_state[i][self.n - 1 - i - j]

        # Need to account for off-diagonals for board size > line size
        if self.n > self.s:

            # Off-diagonal left side
            # Excludes main diagonal
            for j in range(self.n - self.s):
                lineWin = 0
                for i in range(self.n - 2 - j):
                    if self.current_state[i + j + 1][i] == '.' or self.current_state[i + j + 1][i] == '*' \
                            or self.current_state[i + j + 1][i] != self.current_state[i + j + 2][i + 1]:
                        lineWin = 0
                    else:
                        lineWin += 1

                    if lineWin == self.s - 1:
                        return self.current_state[i + j + 1][i]

            # Off-diagonal right side
            # Excludes second diagonal
            for j in range(self.n - self.s):
                lineWin = 0
                for i in range(self.n - 2 - j):
                    if self.current_state[i + j + 1][self.n - 1 - i] == '.' \
                            or self.current_state[i + j + 1][self.n - 1 - i] == '*' \
                            or self.current_state[i + j + 1][self.n - 1 - i] != self.current_state[i + j + 2][
                        self.n - 1 - (i + 1)]:
                        lineWin = 0
                    else:
                        lineWin += 1

                    if lineWin == self.s - 1:
                        return self.current_state[i + j + 1][self.n - 1 - i]

        # Is whole board full?
        for i in range(self.n):
            for j in range(self.n):
                # There's an empty field, we continue the game
                if self.current_state[i][j] == '.':
                    return None
        # It's a tie!
        return '.'

    def check_end(self):
        self.result = self.is_end()
        # Printing the appropriate message if the game has ended
        if self.result is not None:
            if self.result == 'X':

                print('The winner is X!')
                self.winning_e = self.heuristics[0]
            elif self.result == 'O':

                print('The winner is O!')
                self.winning_e = self.heuristics[1]
            elif self.result == '.':

                print("It's a tie!")
                self.winning_e = -1
            self.initialize_game()
        return self.result

    def input_move(self):
        while True:
            print(F'Player {self.player_turn}, enter your move:')
            px = int(input('enter the x coordinate: '))
            py = int(input('enter the y coordinate: '))

            if px in self.colLabels:
                px = self.colLabels[px]
            else:
                print('The move is not valid! Try again.')
                continue

            if self.is_valid(px, py):
                return px, py
            else:
                print('The move is not valid! Try again.')

    def switch_player(self):
        if self.player_turn == 'X':
            self.player_turn = 'O'
        elif self.player_turn == 'O':
            self.player_turn = 'X'
        return self.player_turn

    def minimax(self, max=False, depth=0):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -inf - win for 'X'
        # 0  - a tie
        # inf  - loss for 'X'
        # We're initially setting it to large positive and negative numbers as worse than the worst case:

        value = 10000000
        if max: value = -10000000
        x = None
        y = None
        result = self.is_end()
        if round(time.time() - self.currentTime, 7) >= (9.5 / 10) * self.max_AI_time:
            self.number_of_evaluated_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1



        elif result == 'X':
            return -1000000, x, y
        elif result == 'O':
            return 1000000, x, y
        elif result == '.':
            return 0, x, y
        elif self.max_depth[0] == depth + 1:
            self.number_of_evaluated_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1


        elif self.max_depth[1] == depth + 1:
            self.number_of_evaluated_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1


        for i in range(self.n):
            for j in range(self.n):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.minimax(max=False, depth=depth + 1)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.minimax(max=True, depth=depth + 1)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'

        return value, x, y

    def alphabeta(self, alpha=-2, beta=2, max=False, depth=0):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -inf - win for 'X'
        # 0  - a tie
        # inf  - loss for 'X'
        # We're initially setting it to inf++  or -inf-- as worse than the worst case:

        value = 10000000
        if max: value = -10000000
        x = None
        y = None
        result = self.is_end()
        if round(time.time() - self.currentTime, 7) >= (9.5 / 10) * self.max_AI_time:
            self.number_of_evaluated_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1



        elif result == 'X':
            return -1000000, x, y
        elif result == 'O':
            return 1000000, x, y
        elif result == '.':
            return 0, x, y
        elif self.max_depth[0] == depth + 1:
            self.number_of_evaluated_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1



        elif self.max_depth[1] == depth + 1:
            self.number_of_evaluated_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1


        for i in range(self.n):
            for j in range(self.n):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.alphabeta(alpha, beta, max=False, depth=depth + 1)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.alphabeta(alpha, beta, max=True, depth=depth + 1)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'
                    if max:
                        if value >= beta:
                            return value, x, y
                        if value > alpha:
                            alpha = value
                    else:
                        if value <= alpha:
                            return value, x, y
                        if value < beta:
                            beta = value
        return value, x, y

    def play(self, algo=None, player_x=None, player_o=None):
        if algo is None:
            algo = self.ALPHABETA
        if player_x is None:
            player_x = self.HUMAN
        if player_o is None:
            player_o = self.HUMAN

        self.turn_number = 1
        self.total_time = 0
        self.total_evaluated_nodes = 0
        self.total_average_depths = 0
        self.total_depth_dict = {}


        while True:
            self.currentTime = 0
            self.draw_board(f=f)
            if self.check_end(f=f):
                return
            start = time.time()
            self.currentTime = time.time()

            self.number_of_evaluated_nodes = 0
            self.depth_dict = {}


            if algo == self.MINIMAX:
                if self.player_turn == 'X':
                    (_, x, y) = self.minimax(max=False)
                else:
                    (_, x, y) = self.minimax(max=True)
            else:
                if self.player_turn == 'X':
                    (m, x, y) = self.alphabeta(max=False)
                else:
                    (m, x, y) = self.alphabeta(max=True)
            end = time.time()

            self.total_time += round(end - start, 7)
            self.total_evaluated_nodes += self.number_of_evaluated_nodes

            for key in self.depth_dict:
                if str(int(key) + (self.turn_number - 1)) in self.total_depth_dict:
                    self.total_depth_dict[str(int(key) + (self.turn_number - 1))] += self.depth_dict[key]
                else:
                    self.total_depth_dict[str(int(key) + (self.turn_number - 1))] = self.depth_dict[key]

            print(F'Evaluation time: {round(end - start, 7)}s\n')
            print(F'Number of evaluated nodes: {self.number_of_evaluated_nodes}\n')
            print(F'Evaluations by depth: {self.depth_dict}\n')

            if self.number_of_evaluated_nodes > 0:
                sum = 0
                for key in self.depth_dict:
                    sum += int(key) * self.depth_dict[key]
                print(F'Average evaluation depth: {sum / self.number_of_evaluated_nodes}\n')
                self.total_average_depths += sum / self.number_of_evaluated_nodes



            if (self.player_turn == 'X' and player_x == self.HUMAN) or \
                    (self.player_turn == 'O' and player_o == self.HUMAN):
                if self.recommend:
                    print(F'Evaluation time: {round(end - start, 7)}s')
                    print(F'Recommended move: '
                          F'x = {list(self.colLabels.keys())[list(self.colLabels.values()).index(x)]}, '
                          F'y = {y}')
                (x, y) = self.input_move()
                print(F'Player {self.player_turn} plays: '
                        F'x = {x} '
                        F'y = {y}\n')
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(F'Player {self.player_turn} under AI control plays: '
                        F'x = {list(self.colLabels.keys())[list(self.colLabels.values()).index(x)]}, '
                        F'y = {y}\n')
                print(F'Player {self.player_turn} under AI control plays: '
                      F'x = {list(self.colLabels.keys())[list(self.colLabels.values()).index(x)]}, '
                      F'y = {y}')
            self.current_state[x][y] = self.player_turn
            self.switch_player()
            self.turn_number += 1





def main():
    {}


if __name__ == "__main__":
    main()

