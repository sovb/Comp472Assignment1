import math
import complexHeuristic
import simpleHeuristic

def alphabeta(self, alpha=-math.inf, beta=math.inf, max=False):
    # Minimizing for 'X' and maximizing for 'O'
    # Possible values are:
    # -1 - win for 'X'
    # 0  - a tie
    # 1  - loss for 'X'
    # We're initially setting it to 2 or -2 as worse than the worst case:
    value = math.inf
    if max:
        value = -math.inf
    x = None
    y = None
    maxDepth = 0
    heuristic = None
    if self.player_turn == 'X':
        maxDepth = self.d1
        heuristic = self.player1Heuristic
    else:
        maxDepth = self.d2
        heuristic = self.player2Heuristic
    currentDepth = 0
    result = self.is_end(self)
    if result == 'X':
        return (-1, x, y)
    elif result == 'O':
        return (1, x, y)
    elif result == '.':
        return (0, x, y)
    elif currentDepth == maxDepth:
        #call heuristics - this is our last level of nodes
        currentDepth -= 1
        if heuristic == 'e1':
            # call simple heuristic
            simpleHeuristic.scoreThisCell(self, x, y, self.n, self.s)
        else:
            # call complex heuristic
            complexHeuristic.scoreThisCell(self, x, y, self.n, self.s)
    for i in range(0, self.n):
        for j in range(0, self.n):
            if self.current_state[i][j] == '.':
                if max:
                    self.current_state[i][j] = 'O'
                    currentDepth += 1
                    (v, _, _) = self.alphabeta(alpha, beta, max=False)
                    if v > value:
                        value = v
                        x = i
                        y = j
                else:
                    self.current_state[i][j] = 'X'
                    currentDepth += 1
                    (v, _, _) = self.alphabeta(alpha, beta, max=True)
                    if v < value:
                        value = v
                        x = i
                        y = j
                self.current_state[i][j] = '.'
                if max:
                    if value >= beta:
                        return (value, x, y)
                    if value > alpha:
                        alpha = value
                else:
                    if value <= alpha:
                        return (value, x, y)
                    if value < beta:
                        beta = value
    return (value, x, y)
