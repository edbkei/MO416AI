# MO416AI

# Objetive
1- Comply MO416AI/Prototype3 with Project objectives (MO416AI/Prototype4maT/project.pdf), using the maximum possible the algorithms given in https://github.com/aimacode/aima-python/blob/master/search.ipynb

2- Install Python3 and Pycharm. Tool requirement shall follow: https://github.com/aimacode/aima-python/blob/master/requirements.txt (install with conda install xxxx, pip install xxx)

3- Adapt output.txt to MO416AI/Prototype3/layout/*.lay. OK. MO416AI/Prototype3/layouot/layoutMO416b.lay included.

4- New layout shall be possible with parameter -l in command line python pacman.py. OK. -l layoutMO416b is understood.

5- Ghost shall be static (3 Ghosts). OK

6- Parameter fn=dfs, bfs ucs shall be possible for choosing search method. OK. 

7- Fix Python2 to Python3 problems. OK

8- Check tkint in game.py, moviment shall be very understood in python3. OK. Done

9- 2 minutes videos of project shall be prepared

10- Jupyter notebook shall be prepared (MO416AI/Prototype3/pacman.ipynb). ONGOING. mo416pacman.ipynb started.

11- Test new layout output.txt. OK. output.txt is not appropriated, new layout layoutMO416b.lay is the valid one.

12- Test shall be possible with: python pacman.py -l bigMaze -p SearchAgent -a fn=dfs in Pycharm. OK. 

13- Single search agent using Prototype3: python pacman.py -l layoutMO416b -p SearchAgent -a fn=dfs shall be possible. OK.

14- Find in the code the heuristics used. ONGOING. A* heuristic is possible, one more is needed.

15- Check if missing any requirement. Example, the algorithm used in prototype3 matches https://github.com/aimacode/aima-python/blob/master/search.ipynb? ONGOING.

16- Is it possible to detect ghost using bfs? OK. Yes.

Version information:

v0.0 - Updated 2020/04/21. Updated graphicsUtil.py, autograder.py, grading.py, searchTestClasses.py
       Tests cases 1-5 were possible, TC6 not possible, of the Problem 1: Depth First Search (DFS)
       
v0.1 - Updated 2020/04/21. New layoutMO416.lay developped and included in the folder layout. The following command is possible now:
       python pacman.py -l layoutMO416 -p SearchAgent -a fn=dfs
       
v0.2 - Updated 2020/04/24. All updates has tag MO416. updated pacman.py (to pause ghosts), layout layoutMO416.lay (fix bug, add new packman and ghost positions), search.py (print successor for trace reason), searchAgents.py (print information about the game, for trace reason). It is now possible the command with zoom.

python pacman.py -l layoutMO416 -p SearchAgent -a fn=dfs -z .6

python pacman.py -l mediumSearch -p SearchAgent -a fn=dfs -z .6

v0.3 - Updated 2020/04/25. New maze layout layoutMO416b.lay created at folder layout. It has the only goal to eat only one food at position (3,1). This position is also set in the script search Agents.py, line 148. Now, it is possible to choose search methods DFS and BFS, as below:

python pacman.py -l layoutMO416b -p SearchAgent -a fn=dfs -z .6

python pacman.py -l layoutMO416b -p SearchAgent -a fn=bfs -z .6

Note: the search path used is highlighted during pacman movement.

v0.4 - Updated 2020/04/25. Updated due to requirement itemizations. new mo416pacman.ipynb, pacman.py, searchAgents.py, search.py.

v0.5 - Updated 2020/04/26. New scripts search2.py and search2Agents.py. Parameter -p Search2Agent can now be specified.

       python pacman.py -l mediumSearch -p Search2Agent -a fn=dfs -z .6

v0.6 - Update 2020/04/26. Updated search2.py and search2Agents.py with -a fn=gbfs (Greedy Best First Search).

       python pacman.py -l mediumSearch -p Search2Agent -a fn=gbfs -z .6
      
       introduced euclideanHeuristic in search2.py.  This heuristic is still not good with gbfs.
       
       The algorithm introduced in search2.py is based on A*. The stop condition in A* is f(x) = g(x) + h(x), and in gbfs is f(x) = g(x)
       
       https://www.mygreatlearning.com/blog/best-first-search-bfs/
       
       gbfs with h(x) is still not good.
       
v0.7 - Updated 2020/04/27. Updated search2.py, search2Agents.py, mo416pacman.ipynb to fix gbfs and A* with heuristic Manhattan.

v0.8 - Update 2020/04/27. Updated search2.py and search2Agents.py with -a fn=hcs (Hill Climbing Search).

       python pacman.py -l mediumSearch -p Search2Agent -a fn=hcs -z .6
      
      In this version, hcs is similar to ucs, as it is considered uniform cost.
      
v0.9 - Update 2020/04/27. Updated search2Agents, searchAgents, mo416pacman.ipynb with requirement [22] Ghost positions.    

v0.10 - Updated 2020/04/28. Updated search2.py to comply with Hill Climbing Search, random method introduced when similar cost is encountered between more than 1 neighbours.


