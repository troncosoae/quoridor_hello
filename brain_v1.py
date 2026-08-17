

# The base class which can act as a player
class AIPlayer:
    pass


# Includes BFS logic to check through every possible action. 
#   If there's a winning path, execute it. 
#   If there's a losing path, avoid it. 
#   If neither. Then the chance of winning at each position as the 
#      average of winning or losing in each position. 
#        Example: Position Foo has 4 posible subsequent positions
#            A: rated -0.6
#            B: rated 0.9
#            C: rated 0.0
#             Then current position is their average = 0.1 
class BFSBrain(AIPlayer):
    pass


