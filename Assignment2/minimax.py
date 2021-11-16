def minimax(self,max=False):
		# Minimizing for 'X' and maximizing for 'O'
		# Possible values are:
		# -1 - win for 'X'
		# 0  - a tie
		# 1  - loss for 'X'
		# We're initially setting it to 2 or -2 as worse than the worst case:
		value = 2
		if max:value = -2
		x = None
		y = None
		#checking if the current move resulted in a draw or win or loss
		result = self.is_end()
		if result == 'X':
			return (-1, x, y)
		elif result == 'O':
			return (1, x, y)
		elif result == '.':
			return (0, x, y)
	  #implementing Heuristics(?)
		if self.d1 ==0:
			return (self.simpleHeuristic,x,y)
	  else:
			return (self.sophisticatedHeuristic,x,y)
	  if self.d2 ==0:
			return (self.simpleHeuristic,x,y)
	  else:
			(self.sophisticatedHeuristic,x,y)
	 
	 #
		 for i in range(0,self.n):
			for j in range(0,self.n):
				if (current_state[i][j] == '.'):
					if max:
						self.current_state[i][j]= 'O'
						(v,_,_)=self.minimax(self.d2-1, max= False) #Min when max=false
			      if v> value:
							value=v;
							x=i;
							y=i;
					else:
						self.current_state[i][j]= 'X'
						(v,_,_)=self.minimax(self.d1-1, max= False)
			      if v> value:
							value=v;
							x=i;
							y=i;
			    self.current_state[i][j]='.'
		return (value, x, y)			
