def minimax(self,max=False):

		value = 2
		if max:value = -2
		x = None
		y = None
		#checking  the current move resulted in a draw / win / loss
		result = self.is_end()
		if result == 'X':
			return (-1, x, y)
		elif result == 'O':
			return (1, x, y)
		elif result == '.':
			return (0, x, y)
	  # implement Heuristics(?)
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
						(v,_,_)=self.minimax(self.d2-1, max= False)
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
