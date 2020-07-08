# search2Agents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say


The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

import time

import search2
import util
from game import Actions
from game import Agent
from game import Directions


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class Search2Agent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    #def __init__(self, fn='greedySearch', prob='PositionSearchProblem2', heuristic='nullHeuristic'):
    def __init__(self, fn='depthFirstSearch', prob='FoodSearchProblem2', heuristic='nullHeuristic'):
        if fn not in dir(search2):
            raise AttributeError(fn + ' is not a search function in search2.py.')
        func = getattr(search2, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[Search2Agent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search2):
                heur = getattr(search2, heuristic)
            else:
                raise AttributeError( heuristic + ' is not a function in searchAgents.py or search.py.')
            if (fn == 'hcs' or fn == 'astar' or fn == 'gbfs'):
                print('[Search2Agent] using function %s and [R16] heuristic %s' % (fn, heuristic))
            else:
                print('[Search2Agent] using function %s ' % (fn))

            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem2'):
            raise AttributeError(prob + ' is not a search problem type in Search2Agents.py.')
        self.searchType = globals()[prob]
        print('[Search2Agent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('[R16] Path found with total cost g(x) of '+str(totalCost)+ ' in '+str(time.time() - starttime)+'s')
        if '_expanded' in dir(problem): print('[R13] Search nodes expanded: '+ str(problem._expanded))
        if '_visitedlist' in dir(problem): print('[R13] Nodes visited: ' + str(problem._visitedlist))
        if '_path' in dir(problem): print('[R13] Solution states: ' + str(len(problem._path)) + ' - ' + str(problem._path))
        if '_actions' in dir(problem): print('[R14] Solution actions: ' + str(problem._actions))


    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem2(search2.SearchProblem2):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.goal=(1,1)
        self.goals=[]
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start

        n=0
        try:
            for j in range(1, 40):
                n=j
                x=gameState.hasWall(1, j)
        except:
            n=n
        m=0
        try:
            for i in range(1, 40):
                m=i
                x=gameState.hasWall(i, 1)
        except:
            m=m
        print('maze dimension: ',m,'x',n)

        for i in range(1,m):
            for j in range(1,n):
                if (gameState.hasFood(i,j)):
                    if(gameState.getNumFood()==1):
                        self.goal=(i,j)
                    else:
                        x=(i,j)
                        self.goals.append(x)

        #print('goals',self.getFoodPositions())
        self.costFn = costFn
        self.visualize = visualize
        #x=getFoodPosition(gameState)
        #print("food positions: " )
        print("[R12] Initial position of pacman is "+str(gameState.getPacmanPosition()))
        print("[R10] Number of foods is "+str(gameState.getNumFood()))
        if(gameState.getNumFood()>1):
            print("[R10] Final goal positions are ", self.goals)
        else:
            print("[R10] Final goal position is "+str(self.goals))
        print("[R11] Ghost Positions is/are "+str(gameState.getGhostPositions()))
        print("[R15] has the game food? "+str(gameState.hasFood(*goal)))
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE


    def getStartState(self):
        return self.startState,self.goals

    def isGoalState(self, state):
        isGoal=False
        if (len(self.goals)>1):
            if(state in self.goals):
                isGoal=True
        else:
            isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getFoodPositions(self):
        return self.goals

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            print('search2Agents state',state)
            state1,goals = state
            x,y = state1
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )


        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state1 not in self._visited:
            self._visited[state1] = True
            self._visitedlist.append(state1)

        return successors

    def getFoodPosition(self, gameState):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        #if actions == None: return self.getStartState()
        x,y= self.getStartState()
        cost = 0
        for (x,y) in gameState.getFood():
            # Check figure out the next state and see whether its' legal
            if gameState.hasFood(x,y):
                return (x,y)
        return (x,y)

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        initstat,goals1= self.getStartState()
        x,y=initstat
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


class CornersProblem(search2.SearchProblem2):
    """
    This search problem finds paths through all four corners of a layout.
    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print
                'Warning: no food in corner ' + str(corner)
        self._expanded = 0  # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem

        self.startingGameState = startingGameState

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        """ A state space can be the start coordinates and a list to hold visited corners"""
        return (self.startingPosition, [])
        # util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """

        """ Check to see if a state is a corner, and if so are the other corners visited"""
        xy = state[0]
        visitedCorners = state[1]
        if xy in self.corners:
            if not xy in visitedCorners:
                visitedCorners.append(xy)
            return len(visitedCorners) == 4
        return False

        # util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        successors = []
        x, y = state[0]
        visitedCorners = state[1]
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]


            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                # Initialize a list of Visited corners for a successor using the visited corner list in state space.
                successorVisitedCorners = list(visitedCorners)
                next_node = (nextx, nexty)
                # Add node to the Visited corner list if it is a corner and not already in the list
                if next_node in self.corners:
                    if not next_node in successorVisitedCorners:
                        successorVisitedCorners.append(next_node)
                # Create a new state according to the state space and append it to the successor list.
                successor = ((next_node, successorVisitedCorners), action, 1)
                successors.append(successor)

        self._expanded += 1  # DO NOT CHANGE

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)



class FoodSearchProblem2:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.
    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information
        # new
        self.goal=(1,1)
        self.goals=[]
        #self.walls = gameState.getWalls()
        #self.startState = gameState.getPacmanPosition()

        n=0
        try:
            for j in range(1, 40):
                n=j
                x=startingGameState.hasWall(1, j)
        except:
            n=n
        m=0
        try:
            for i in range(1, 40):
                m=i
                x=startingGameState.hasWall(i, 1)
        except:
            m=m
        print('[R9] [R20] maze dimension: ',m,'x',n)

        alreadyFound=False
        for i in range(1,m):
            for j in range(1,n):
                if (startingGameState.hasFood(i,j)):
                    if(startingGameState.getNumFood()==1 and alreadyFound):
                        self.goal=(i,j)
                        alreadyFound=True
                        break
                    else:
                        x=(i,j)
                        self.goals.append(x)

        #print('goals',self.getFoodPositions())

        #x=getFoodPosition(gameState)
        #print("food positions: " )
        print("[R12] Initial position of pacman is "+str(startingGameState.getPacmanPosition()))
        print("[R10] [R15] Number of foods is "+str(startingGameState.getNumFood()))
        if(startingGameState.getNumFood()>1):
            print("[R10] [R15] Final goal (Food) positions are ", self.goals)
        else:
            print("[R10] [R15] Final goal (Food) position is "+str(self.goals))
        print("[R11] Ghost Positions is/are "+str(startingGameState.getGhostPositions()))



    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getSuccessors2(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            print('search2agent state=',state)
            x,y = state.copy()
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

    def getFoodPositions(self):
        return self.goals




class StayEastSearchAgent2(Search2Agent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search2.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem2(state, costFn, (1, 1), None, False)

class StayWestSearchAgent2(Search2Agent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search2.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem2(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position[0]
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position[0]
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5


