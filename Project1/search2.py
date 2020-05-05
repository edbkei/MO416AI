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

    stack = util.Stack()
    traceAction = util.Stack()
    traceState = util.Stack()

    traveled = []
    step_counter = 0

    start_state = problem.getStartState()
    stack.push((start_state, step_counter, 'START'))
    traceState.push(start_state)

    while not stack.isEmpty():
        
        # arrive at state
        curr_state, _, action = stack.pop()
        traveled.append(curr_state)
        
        # record action that get to that state
        if action != 'START':
            traceAction.push(action)
            traceState.push(curr_state)
            step_counter += 1

        # check if state is goal
        if problem.isGoalState(curr_state):
            # back track
            problem = backTrackStackUninformed(problem, traceAction, traceState)
            return traceAction.list

        # get possible next states
        valid_successors = 0
        successors = problem.getSuccessors(curr_state)

        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]

            # avoid traveling back to previous states
            if next_state not in traveled:
                valid_successors += 1
                stack.push((next_state, step_counter, next_action))

        # dead end, step backwards
        if valid_successors == 0:
            while step_counter != stack.list[-1][1]: # back until next awaiting state
                step_counter -= 1
                traceAction.pop()
                traceState.pop()
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    queue = util.Queue()
    trace = {}
    seen = []

    start_state = problem.getStartState()
    queue.push(start_state)
    seen.append(start_state)

    while not queue.isEmpty():
        
        # arrive at state
        curr_state = queue.pop()

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]

            # avoid traveling back to previous states
            if next_state not in seen:
                seen.append(next_state)
                queue.push(next_state)
                trace[next_state] = (curr_state, next_action)

    # back track
    problem, actions = backTrackUninformed(problem, start_state, curr_state, trace)

    return actions



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
    start_state = problem.getStartState()

    g = {}
    g[start_state] = 0
    def f(curr_node): return float(g[curr_node] + manhattanHeuristic(curr_node, problem))


    open_list = util.PriorityQueue()
    open_list.push(start_state, 0)
    open_seen = [start_state] # for 'in' operator, as PriorityQueueWithFunction records a tuple with priority
    close_list = []
    trace = {}
    trace[start_state] = [None, None, 0]

    while not open_list.isEmpty():

        # arrive at state
        curr_state = open_list.pop()
        open_seen.remove(curr_state)

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        #print(successors) # MO416 testinig
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]
            successor_cost = g[curr_state] + next_cost
           
            UPDATE = False
            if next_state in open_seen:
                if g[next_state] <= successor_cost:
                    pass
                else:
                    g[next_state] = successor_cost
                    open_list.update(item=next_state, priority=f(next_state))
            elif next_state in close_list:
                if g[next_state] <= successor_cost:
                    pass
                else: UPDATE = True
            else: UPDATE = True



            if UPDATE:
                g[next_state] = successor_cost
                open_list.update(item=next_state, priority=f(next_state))
                open_seen.append(next_state)

                if next_state in close_list:
                    close_list.remove(next_state)
                    open_seen.remove(next_state)

            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > successor_cost:
                    trace[next_state][0] = curr_state
                    trace[next_state][1] = next_action
                    trace[next_state][2] = successor_cost
            else:
                trace[next_state] = [curr_state, next_action, successor_cost]

        close_list.append(curr_state)

    # back track
    problem, actions = backTrackInformed(problem, start_state, curr_state, trace)

    return actions

def aStarEuclideanSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()

    g = {}
    g[start_state] = 0

    def f(curr_node):
        return float(g[curr_node] + euclideanHeuristic(curr_node, problem))

    open_list = util.PriorityQueue()
    open_list.push(start_state, 0)
    open_seen = [start_state]  # for 'in' operator, as PriorityQueueWithFunction records a tuple with priority
    close_list = []
    trace = {}
    trace[start_state] = [None, None, 0]

    while not open_list.isEmpty():

        # arrive at state
        curr_state = open_list.pop()
        open_seen.remove(curr_state)

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        # print(successors) # MO416 testinig

        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]
            successor_cost = g[curr_state] + next_cost

            UPDATE = False
            if next_state in open_seen:
                if g[next_state] <= successor_cost:
                    pass
                else:
                    g[next_state] = successor_cost
                    open_list.update(item=next_state, priority=f(next_state))
            elif next_state in close_list:
                if g[next_state] <= successor_cost:
                    pass
                else:
                    UPDATE = True
            else:
                UPDATE = True

            if UPDATE:
                g[next_state] = successor_cost
                open_list.update(item=next_state, priority=f(next_state))
                open_seen.append(next_state)

                if next_state in close_list:
                    close_list.remove(next_state)
                    open_seen.remove(next_state)

            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > successor_cost:
                    trace[next_state][0] = curr_state
                    trace[next_state][1] = next_action
                    trace[next_state][2] = successor_cost
            else:
                trace[next_state] = [curr_state, next_action, successor_cost]

        close_list.append(curr_state)

    # back track
    actions = []
    backtrack_state = curr_state  # the goal state
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions


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
        prev_state, action = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
        states.append(prev_state)
    actions = list(reversed(actions))
    problem._actions = actions
    problem._path = list(reversed(states))

    return problem, actions

def backTrackStackUninformed(problem, traceAction, traceState):
    problem._actions = traceAction.list
    problem._path = traceState.list
    return problem

def greedyBestFirstSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # https://www.mygreatlearning.com/blog/best-first-search-bfs/
    start_state = problem.getStartState()

    g = {}
    g[start_state] = 0
    #h = euclideanHeuristic(problem.getStartState, problem)
    def f(curr_node):
        return float(manhattanHeuristic(curr_node, problem))
        #return float(g[curr_node] + manhattanHeuristic(curr_node, problem))
    #def f(curr_node):
        #print("g="+str(g[curr_node])+",h="+str(euclideanHeuristic(curr_node, problem)))
        #return float(euclideanHeuristic(curr_node, problem))
        #return float(g[curr_node] + euclideanHeuristic(curr_node, problem))

    open_list = util.PriorityQueue()
    open_list.push(start_state, 0)
    open_seen = [start_state]  # for 'in' operator, as PriorityQueueWithFunction records a tuple with priority
    close_list = []
    trace = {}
    trace[start_state] = [None, None, 0]

    while not open_list.isEmpty():

        # arrive at state
        curr_state = open_list.pop()
        #h=euclideanHeuristic(curr_node, problem)
        #print('curr_state='+str(curr_state))
        #print(open_seen)
        open_seen.remove(curr_state)

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        #print(successors)  # MO416 testinig

        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]
            successor_cost = g[curr_state] + next_cost

            UPDATE = False
            #print("successor_cost="+str(successor_cost))
            if next_state in open_seen:
                h=euclideanHeuristic(next_state, problem)
                #print("h="+str(h))
                #if(h==0):pass
                if g[next_state] <= successor_cost:
                    pass
                else:
                    g[next_state] = successor_cost
                    open_list.update(item=next_state, priority=f(next_state))
            elif next_state in close_list:
                if g[next_state] <= successor_cost:
                    pass
                else:
                    UPDATE = True
            else:
                UPDATE = True

            if UPDATE:
                g[next_state] = successor_cost
                open_list.update(item=next_state, priority=f(next_state))
                open_seen.append(next_state)

                if next_state in close_list:
                    close_list.remove(next_state)
                    open_seen.remove(next_state)

            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > successor_cost:
                    trace[next_state][0] = curr_state
                    trace[next_state][1] = next_action
                    trace[next_state][2] = successor_cost
            else:
                trace[next_state] = [curr_state, next_action, successor_cost]

        close_list.append(curr_state)

    # back track
    problem, actions = backTrackInformed(problem, start_state, curr_state, trace)

    return actions

def greedyBestFirstEuclideanSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # https://www.mygreatlearning.com/blog/best-first-search-bfs/
    start_state = problem.getStartState()

    g = {}
    g[start_state] = 0
    #h = euclideanHeuristic(problem.getStartState, problem)
    def f(curr_node):
        return float(euclideanHeuristic(curr_node, problem))
        #return float(g[curr_node] + manhattanHeuristic(curr_node, problem))
    #def f(curr_node):
        #print("g="+str(g[curr_node])+",h="+str(euclideanHeuristic(curr_node, problem)))
        #return float(euclideanHeuristic(curr_node, problem))
        #return float(g[curr_node] + euclideanHeuristic(curr_node, problem))

    open_list = util.PriorityQueue()
    open_list.push(start_state, 0)
    open_seen = [start_state]  # for 'in' operator, as PriorityQueueWithFunction records a tuple with priority
    close_list = []
    trace = {}
    trace[start_state] = [None, None, 0]

    while not open_list.isEmpty():

        # arrive at state
        curr_state = open_list.pop()
        #h=euclideanHeuristic(curr_node, problem)
        #print('curr_state='+str(curr_state))
        #print(open_seen)
        if curr_state in open_seen:
           open_seen.remove(curr_state)

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        #print(successors)  # MO416 testinig

        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]
            successor_cost = g[curr_state] + next_cost

            UPDATE = False
            #print("successor_cost="+str(successor_cost))
            if next_state in open_seen:
                h=euclideanHeuristic(next_state, problem)
                #print("h="+str(h))
                #if(h==0):pass
                if g[next_state] <= successor_cost:
                    pass
                else:
                    g[next_state] = successor_cost
                    open_list.update(item=next_state, priority=f(next_state))
            elif next_state in close_list:
                if g[next_state] <= successor_cost:
                    pass
                else:
                    UPDATE = True
            else:
                UPDATE = True

            if UPDATE:
                g[next_state] = successor_cost
                open_list.update(item=next_state, priority=f(next_state))
                open_seen.append(next_state)

                if next_state in close_list:
                    close_list.remove(next_state)
                    open_seen.remove(next_state)

            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > successor_cost:
                    trace[next_state][0] = curr_state
                    trace[next_state][1] = next_action
                    trace[next_state][2] = successor_cost
            else:
                trace[next_state] = [curr_state, next_action, successor_cost]

        close_list.append(curr_state)

    # back track
    actions = []
    backtrack_state = curr_state  # the goal state
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions

def hillClimbingSearch(problem, heuristic=nullHeuristic):
    priority_queue = util.PriorityQueue()
    #print(priority_queue.count)
    trace = {}
    seen = []
    goal=(1,3)

    start_state = problem.getStartState()
    #print(start_state)
    prev_cost = 0
    trace[start_state] = [None, None, prev_cost]

    priority_queue.update(start_state, 0)
    seen.append(start_state)
    prevsuccessors2 = []
    prevsuccessors22 = []
    while not priority_queue.isEmpty():

        # arrive at state
        curr_state = priority_queue.pop()

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        #print(curr_state)
        #t=len(successors) # MO416
        #print(random.randint(0,t-1))
        #print(successors) # MO416
        #print(successors[0])
        t=len(successors) # MO416
        successors2 = []
        if(t==1):
            idx=0
            successors2=successors
        else:
            i=0
            while True:
                i=i+1
                idx = random.randint(0, t - 1)
                #print(prevsuccessors2)
                if(prevsuccessors2==[]):
                    successors2.append(successors[idx])
                    prevsuccessors2=successors2
                    break
                if (successors[idx][0] != prevsuccessors2[0]):
                    #print(successors)
                    #print(successors[idx][1])
                    #print(prevsuccessors2)
                    #print(prevsuccessors2[0][0])
                    if(not((successors[idx][1]=='East' and prevsuccessors2[0][1]=='West') \
                            or (successors[idx][1]=='West' and prevsuccessors2[0][1]=='East') \
                            or (successors[idx][1]=='South' and prevsuccessors2[0][1]=='North') \
                            or (successors[idx][1] == 'North' and prevsuccessors2[0][1] == 'South'))):
                                successors2.append(successors[idx])
                                prevsuccessors2=successors2
                                break
                if (i>5):
                    idx = random.randint(0, t - 1)
                    successors2.append(successors[idx])
                    prevsuccessors2 = successors2
                    break
                #successors2.append(successors[idx])
        #print(str(successors)+" is "+str(successors2))
        #print(prevsuccessors2)
        #print(successors)
        #print(successors2)


        for successor in successors2:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]

            if (prevsuccessors22 == []):
                #prevsuccessors22.append(successor)
                prevsuccessors22 = successor
                prev_cost = trace[curr_state][2]
                seen.append(next_state)
                #priority_queue.update(next_state, next_cost + prev_cost)
                priority_queue.update(next_state, 1)
            elif (not ((next_action == 'East' and prevsuccessors22[0][1] == 'West') \
                     or (next_action == 'West' and prevsuccessors22[0][1] == 'East') \
                     or (next_action == 'South' and prevsuccessors22[0][1] == 'North') \
                     or (next_action == 'North' and prevsuccessors22[0][1] == 'South'))):
                        #prevsuccessors22.append(successor)
                        prevsuccessors22 = successor
                        prev_cost = trace[curr_state][2]
                        seen.append(next_state)
                        priority_queue.update(next_state, 1)
            else:
                seen.remove(next_state)

            #if next_state not in seen:
            #    prev_cost = trace[curr_state][2]
             #   seen.append(next_state)
                #priority_queue.update(next_state, next_cost + prev_cost)
             #   priority_queue.update(next_state, 1)
            #else:
                #seen.remove(next_state)


            # update and allow tracing to the best state
            #if next_state in trace:
                #if trace[next_state][2] > next_cost + prev_cost:
                    #trace[next_state][2] = next_cost + prev_cost
                #    trace[next_state][2] = next_cost + prev_cost
                #    trace[next_state][1] = next_action
                #    trace[next_state][0] = curr_state
            #else:
            #

            #print("seen is "+str(seen))
            if not next_state in trace:
                #trace[next_state] = [curr_state, next_action, next_cost + prev_cost]
                trace[next_state] = [curr_state, next_action, 1]
            #print("trace is "+str(trace))
            if(curr_state==goal):break

    # back track
    #print(trace)
    problem, actions = backTrackInformed(problem, start_state, curr_state, trace)

    return actions

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return int(( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

# Abbreviations
astar = aStarSearch
astare = aStarEuclideanSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
gbfs = greedyBestFirstSearch
gbfes = greedyBestFirstEuclideanSearch
hcs = hillClimbingSearch
ucs = uniformCostSearch
