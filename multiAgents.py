# Yanir Buznah 207631466

# multiAgents.py
# --------------
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
from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState: GameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isLose():
            return -100
        # starting value of the additional score
        score = 0
        new_ghost_positions = successorGameState.getGhostPositions()
        walls = currentGameState.getWalls()
        ghost_states = currentGameState.getGhostStates()

        # check if there is ghost close to the agent
        ghosts_in_row = 0
        ghosts_in_col = 0
        for i in new_ghost_positions:
            if i[0] == newPos[0] and abs(i[1] - newPos[1]) < 2:
                ghosts_in_col += 1
            if i[1] == newPos[1] and abs(i[0] - newPos[0]) < 2:
                ghosts_in_row += 1
        # return -10 if there are 2 ghosts on either side of the agent
        if ghosts_in_col > 1 or ghosts_in_row > 1:
            return -10

        unscared_ghost = 3
        scared_ghost = max(newScaredTimes) + 1
        for i in range(len(ghost_states)):
            # if the agent eat ghost, add 10000 to the score
            if ghost_states[i].scaredTimer > 1 and newScaredTimes[i] == 0:
                score += 10000
            else:
                # get the distance of the closet agent
                if newScaredTimes[i] == 0:
                    unscared_ghost = min(unscared_ghost,
                                         manhattanDistance(successorGameState.getGhostPosition(i + 1), newPos))
                else:
                    scared_ghost = min(scared_ghost,
                                       manhattanDistance(successorGameState.getGhostPosition(i + 1), newPos))
        # add to score 1000/ distance of the closest ghost, (The closer the ghost is the higher the score)
        score += 1000 / scared_ghost

        # if the closest ghost to close return -1
        if unscared_ghost <= 1:
            return -1

        # adds the distance from the nearest ghost, the farther the ghost, the higher the score
        score += unscared_ghost

        foods = newFood.asList()
        dis = 100000
        # add 1000 if the agent eat capsule
        if len(successorGameState.data.capsules) < len(currentGameState.data.capsules):
            score += 1000
        # add 100 if the agent eat normal food
        elif currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 100
        else:
            for food in foods:
                if not check_walls(newPos, food, walls):
                    # find the minimum distance from all the food so there is no wall between the agent and the food
                    dis = min(dis, manhattanDistance(food, newPos))
            # The closer the distance, the greater the score
            score += 100 / dis
        # Always prefer to move
        if action == 'Stop':
            score = -10
        # add the normal score to the calculation
        return scoreEvaluationFunction(successorGameState) + score


# check if there is wall between the food and the position of the agent
def check_walls(pos, food, walls):
    if pos[1] == food[1]:
        if pos[0] > food[0]:
            return walls[pos[0] - 1][pos[1]]
        else:
            return walls[pos[0] + 1][pos[1]]
    if pos[0] == food[0]:
        if pos[1] > food[1]:
            return walls[pos[0]][pos[1] - 1]
        else:
            return walls[pos[0]][pos[1] + 1]
    return False


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minmax(gameState: GameState, depth, agent):
            best_action = None
            # End of round
            if agent == 0:
                depth -= 1
            # Terminal state
            if depth < 0 or gameState.isLose() or gameState.isWin() or not gameState.getLegalActions(agent):
                best_evaluation = self.evaluationFunction(gameState)
            else:
                # Max agent
                if agent == 0:
                    best_evaluation = -float('inf')
                    for action in gameState.getLegalActions(0):
                        evaluation = minmax(gameState.generateSuccessor(0, action), depth, 1)
                        if best_evaluation < evaluation[0]:
                            best_evaluation = evaluation[0]
                            best_action = action
                # Min agent
                else:
                    best_evaluation = float('inf')
                    for action in gameState.getLegalActions(agent):
                        # Goes through all the ghosts in a cyclical way
                        evaluation = minmax(gameState.generateSuccessor(agent, action), depth,
                                            (agent + 1) % gameState.getNumAgents())
                        if best_evaluation > evaluation[0]:
                            best_evaluation = evaluation[0]
                            best_action = action

            return best_evaluation, best_action

        return minmax(gameState, self.depth, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alpha_beta_pruning(gameState: GameState, depth, agent, alpha, beta):
            best_action = None
            if agent == 0:
                depth -= 1
            if depth < 0 or gameState.isLose() or gameState.isWin() or not gameState.getLegalActions(agent):
                best_evaluation = self.evaluationFunction(gameState)
            else:
                if agent == 0:
                    best_evaluation = -float('inf')
                    for action in gameState.getLegalActions(0):
                        evaluation = alpha_beta_pruning(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)
                        alpha = max(alpha, evaluation[0])
                        if best_evaluation < evaluation[0]:
                            best_evaluation = evaluation[0]
                            best_action = action
                        if beta < alpha:
                            break

                else:
                    best_evaluation = float('inf')
                    for action in gameState.getLegalActions(agent):
                        evaluation = alpha_beta_pruning(gameState.generateSuccessor(agent, action), depth,
                                                        (agent + 1) % gameState.getNumAgents(), alpha, beta)
                        beta = min(beta, evaluation[0])
                        if best_evaluation > evaluation[0]:
                            best_evaluation = evaluation[0]
                            best_action = action
                        if beta < alpha:
                            break
            return best_evaluation, best_action

        return alpha_beta_pruning(gameState, self.depth, 0, -float('inf'), float('inf'))[1]
