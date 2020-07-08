# search2.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random
import heapq

class SearchProblem2:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def getFoodPositions(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    Frontier = util.Stack()
    Visited = []
    Frontier.push((problem.getStartState(), []))
    Visited.append(problem.getStartState())
    prepath = util.Stack()
    path=[]
    Visited = []

    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()
        prepath.push((state[0], []))

        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in Visited:
                if problem.isGoalState(n_state):
                    visited = []
                    for i in Visited:
                        visited.append(i[0])
                    while not prepath.isEmpty():
                        x = prepath.pop()
                        path.append(x[0])

                    print('[R13] Nodes visited:', visited)
                    m=list(reversed(path))
                    m.append(n_state[0])
                    print('[R13] Solution states:', m)
                    print('[R14] Solution actions:', actions+[n_direction])
                    # print 'Find Goal'
                    return actions + [n_direction]
                else:
                    Frontier.push((n_state, actions + [n_direction]))
                    Visited.append(n_state)

    util.raiseNotDefined()


#util.raiseNotDefined()

    #print('[R13] Nodes visited:',visited)
    #print('[R13] Solution states:', states)
    #print('[R14] Solution actions:',path)
    #return path

    # util.raiseNotDefined()
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    """Search the shallowest nodes in the search tree first."""

    Frontier = util.Queue()
    Visited = []
    Frontier.push((problem.getStartState(), []))
    prepath = util.Stack()
    path=[]
    # print 'Start',problem.getStartState()
    # Visited.append( problem.getStartState() )

    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()
        prepath.push((state[0], []))

        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in Visited:
                if problem.isGoalState(n_state):
                    # print 'Find Goal'
                    visited = []
                    for i in Visited:
                        visited.append(i[0])
                    while not prepath.isEmpty():
                        x = prepath.pop()
                        path.append(x[0])

                    print('[R13] Nodes visited:', visited)
                    m=list(reversed(path))
                    m.append(n_state[0])
                    print('[R13] Solution states:', m)
                    print('[R14] Solution actions:', actions+[n_direction])
                    return actions + [n_direction]
                Frontier.push((n_state, actions + [n_direction]))
                Visited.append(n_state)

    util.raiseNotDefined()

    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    priority_queue = util.PriorityQueue()
    trace = {}
    seen = []

    start_state = problem.getStartState()
    prev_cost = 0
    trace[start_state] = [None, None, prev_cost]

    priority_queue.update(start_state, 0)
    seen.append(start_state)

    while not priority_queue.isEmpty():
        
        # arrive at state
        curr_state = priority_queue.pop()

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]

            # avoid traveling back to previous states
            if next_state not in seen:
                prev_cost = trace[curr_state][2]
                seen.append(next_state)
                priority_queue.update(next_state, next_cost + prev_cost)
                
            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > next_cost + prev_cost:
                    trace[next_state][2] = next_cost + prev_cost
                    trace[next_state][1] = next_action
                    trace[next_state][0] = curr_state
            else:
                trace[next_state] = [curr_state, next_action, next_cost + prev_cost]

    # back track
    problem, actions = backTrackInformed(problem, start_state, curr_state, trace)

    return actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    def _update(Frontier, item, priority):
        for index, (p, c, i) in enumerate(Frontier.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del Frontier.heap[index]
                Frontier.heap.append((priority, c, item))
                heapq.heapify(Frontier.heap)
                break
        else:
            Frontier.push(item, priority)

    Frontier = util.PriorityQueue()
    Visited = []
    Frontier.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem))
    Visited.append(problem.getStartState())
    prepath = util.Stack()
    path=[]

    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()
        prepath.push((state[0], []))
        # print state
        if problem.isGoalState(state):
            # print 'Find Goal'
            visited = []
            for i in Visited:
                visited.append(i[0])
            while not prepath.isEmpty():
                x = prepath.pop()
                path.append(x[0])

            print('[R13] Nodes visited:', visited)
            m = list(reversed(path))
            print('[R13] Solution states:', m)
            print('[R14] Solution actions:', actions )
            # print 'Find Goal'
            return actions

        if state not in Visited:
            Visited.append(state)

        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in Visited:
                _update(Frontier, (n_state, actions + [n_direction]), \
                        problem.getCostOfActions(actions + [n_direction]) + heuristic(n_state, problem))

    util.raiseNotDefined()


# util.raiseNotDefined()


def backTrackInformed(problem, start_state, curr_state, trace):
    actions = []
    states = []
    backtrack_state = curr_state # the goal state
    states.append(curr_state)
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
        states.append(prev_state)
    actions = list(reversed(actions))
    problem._actions = actions
    problem._path = list(reversed(states))

    return problem, actions

def backTrackUninformed(problem, start_state, curr_state, trace):
    actions = []
    states = []
    backtrack_state = curr_state # the goal state
    states.append(curr_state)
    while backtrack_state != start_state:
        #print('backtrack_state: ',backtrack_state,'trace[backtrac_state]: ',trace[backtrack_state])
        prev_state,action,_ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
        states.append(prev_state)
    actions = list(reversed(actions))
    problem._actions = actions
    problem._path = list(reversed(states))
    #print('problem=',problem._path)

    return problem, actions

def backTrackStackUninformed(problem, traceAction, traceState):
    problem._actions = traceAction.list
    problem._path = traceState.list
    return problem

def greedyBestFirstSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    def _update(Frontier, item, priority):
        for index, (p, c, i) in enumerate(Frontier.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del Frontier.heap[index]
                Frontier.heap.append((priority, c, item))
                heapq.heapify(Frontier.heap)
                break
        else:
            Frontier.push(item, priority)

    Frontier = util.PriorityQueue()
    Visited = []
    prepath = util.Stack()
    path=[]
    Frontier.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem))
    Visited.append(problem.getStartState())

    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()
        prepath.push((state[0], []))
        # print state
        if problem.isGoalState(state):
            visited = []
            for i in Visited:
                visited.append(i[0])
            while not prepath.isEmpty():
                x = prepath.pop()
                path.append(x[0])

            print('[R13] Nodes visited:', visited)
            m = list(reversed(path))
            print('[R13] Solution states:', m)
            print('[R14] Solution actions:', actions)
            # print 'Find Goal'
            return actions

        if state not in Visited:
            Visited.append(state)

        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in Visited:
                _update(Frontier, (n_state, actions + [n_direction]), heuristic(n_state, problem))

    util.raiseNotDefined()


def hillClimbingSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    def _update(Frontier, item, priority):
        for index, (p, c, i) in enumerate(Frontier.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del Frontier.heap[index]
                Frontier.heap.append((priority, c, item))
                heapq.heapify(Frontier.heap)
                break
        else:
            Frontier.push(item, priority)
    prepath = util.Stack()
    Visited = []

    Frontier = util.PriorityQueue()
    Visited = []
    path=[]
    Frontier.push( (problem.getStartState(), []), 0 )
    Visited.append( problem.getStartState() )


    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()
        prepath.push((state[0], []))

        if problem.isGoalState(state):
            visited=[]
            for i in Visited:
                visited.append(i[0])
            while not prepath.isEmpty():
                x=prepath.pop()
                path.append(x[0])

            print('[R13] Nodes visited:',visited)
            print('[R13] Solution states:',list(reversed(path)))
            print('[R14] Solution actions:',actions)
            return actions

        if state not in Visited:
            Visited.append( state )

        if(len(Visited)>100):
            Visited.pop(0)

        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in Visited:

                _update( Frontier, (n_state, actions + [n_direction]), problem.getCostOfActions(actions+[n_direction])+ heuristic(n_state, problem) )


    util.raiseNotDefined()


# Abbreviations
astar = aStarSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
gbfs = greedyBestFirstSearch
hcs = hillClimbingSearch
ucs = uniformCostSearch
