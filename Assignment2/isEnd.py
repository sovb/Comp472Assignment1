
def is_end(self):
    result = None
    n = self.n
    s = self.s
    player = self.player_turn
    unplayedCells = 0
    streakX = 0
    streakY = 0
    winFoundX = False
    winFoundY = False

    # checking if all empty-  if all empty return None
    for row in self.current_state:
        for item in row:
            if item == '.' or item =='*':
                unplayedCells += 1
    if unplayedCells == n*n:
        return None

    # checking for X

    for i in range(n):
        streakX = 0
        for j in range(n):
            if self.current_state[j][i] == 'X':
                streakX += 1
            else:
                streakX = 0
        if streakX >= s:
            winFoundX = True
        if winFoundX is True:
            break

    # Horizontal win
    if winFoundX is False:
        for i in range(n):
            streakX = 0
            for j in range(n):


                if self.current_state[i][j] == 'X':
                    streakX += 1

                else:
                    streakX = 0
            if streakX >= s:
                winFoundX = True
            if winFoundX is True:
                break

    # diagonal check
    if winFoundX is False:
        for i in range(n):
            streakX = 0
            if self.current_state[i][i] == 'X':
                streakX += 1
            else:
                streakX = 0
            if streakX >= s:
                winFoundX = True
            if winFoundX is True:
                break

    # diagonal check 2
    if winFoundX is False:
        for i in range(n):
            streakX = 0
            if self.current_state[i][n-1-i] == 'X':
                streakX += 1
            else:
                streakX = 0
            if streakX >= s:
                winFoundX = True
            if winFoundX is True:
                break

    # checking for Y

    for i in range(n):
        streakY = 0
        for j in range(n):
            if self.current_state[j][i] == 'Y' :
                streakY += 1
            else:

                streakY = 0
        if streakY >= s:
            winFoundY = True
        if winFoundY is True:
            break


    if winFoundY is False:
        for i in range(n):
            streakY = 0
            for j in range(n):
                if self.current_state[i][j] == 'Y':
                    streakY += 1
                else:
                    streakY = 0
            if streakY >= s:
                winFoundY = True
            if winFoundY is True:
                break

    # checking diagonal 1
    if winFoundY is False:
        for i in range(n):
            streakY = 0
            if self.current_state[i][i] == 'Y':
                streakY += 1
            else:
                streakY = 0
            if streakY >= s:
                winFoundY = True
            if winFoundY is True:
                break

    # diagonal check  2
    if winFoundY is False:
        for i in range(n):
            streakY = 0
            if self.current_state[i][n-1-i] == 'Y':
                streakY += 1
            else:
                streakY = 0
            if streakY >= s:
                winFoundY = True
            if winFoundY is True:
                break


    if winFoundX == True and winFoundY == True:
        result = '.'
    elif winFoundX == True and winFoundY == False:
        result = 'X'
    elif winFoundX == False and winFoundY == True:
        result = 'Y'

    # returning to  final result
    return result


