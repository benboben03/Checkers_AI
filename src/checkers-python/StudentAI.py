from BoardClasses import Move
from BoardClasses import Board
from copy import deepcopy 
from math import log, sqrt
from time import time
import random

class StudentAI():

    def __init__(self,col,row,p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col,row,p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1:2,2:1}
        self.color = 2
        self.monte_carlo_tree = MonteCarloTreeSearch(TreeNode(self.board, self.color, None, None))
        self.time_adjustment_coefficient = (self.row * self.col)/2
        self.num_moves = 2
        self.timer = 125
        
    def get_move(self, move) -> Move:
        '''
        Determines the next move for the AI player using the Monte Carlo Tree Search algorithm.

        Parameters: move (Move): The move executed by the opponent, represented as a Move object.

        Returns:  Move: The chosen move for the AI player.
        '''
        #Initialize timer
        start = time()
        
        #If it is the first move of the game, the AI initializes its color and makes a random move.
        if len(move) != 0:
            self.play_move(move, {1: 2, 2: 1}[self.color])
     
        #If the opponent has made a move, it is executed on the internal game board.
        else:
            self.color = 1
            self.monte_carlo_tree.root_node = TreeNode(self.board, self.color , None, None)

            possible_moves = self.board.get_all_possible_moves(self.color)
            f = possible_moves[0][1]
            self.play_move(f, self.color)
            return f
        
        #If there is only one possible move, it is executed.
        possible_moves = self.board.get_all_possible_moves(self.color)
        if len(possible_moves) == 1 and len(possible_moves[0]) == 1:
            self.play_move(possible_moves[0][0], self.color)
            return possible_moves[0][0]
        
        #The time limit for the MCTS algorithm is calculated based on the remaining timer and the time divisor.
        time_limit = self.timer / self.time_adjustment_coefficient
        move_chosen = self.monte_carlo_tree.tree_search(time_limit)
        self.play_move(move_chosen, self.color)
        
        #After each move, the time divisor is adjusted to gradually reduce the allocated time for future moves.
        offset = 0.5 - (1/self.num_moves)
        self.time_adjustment_coefficient -= offset
        
        #Increment number of moves
        self.num_moves += 1
        
        #The chosen move is returned, and the internal state of the AI is updated accordingly.
        self.timer = self.timer - (time() - start)
        return move_chosen
    
    def play_move(self, move, color):
        self.board.make_move(move, color)

        for child_move, child_node in self.monte_carlo_tree.root_node.children.items():
            if child_node is not None and str(child_move) == str(move):
                self.monte_carlo_tree.root_node = child_node
                self.monte_carlo_tree.root_node.parent = None
                return

        # If the move is not in the existing children, create a new TreeNode
        self.monte_carlo_tree.root_node = TreeNode(self.board, {1: 2, 2: 1}[color], None, None)
        
class TreeNode():
    def __init__(self, board, color, move, parent):
        self.board = deepcopy(board)
        self.color = color
        self.parent = parent
        self.upper_confidence_bound = 0
        self.parent_win_counter = 0
        self.current_node_visit_counter = 1
      
        #First move
        if move is not None:
            self.board.make_move(move, {1: 2, 2: 1}[self.color])

        #Children are created in instances where game state is over
        self.children = dict()
        if self.board.is_win({1: 2, 2: 1}[self.color]) == 0:
            moves_list = self.board.get_all_possible_moves(self.color)
            for i in range(len(moves_list)):
                for j in range(len(moves_list[i])):
                    self.children[moves_list[i][j]] = None
 
    def backpropogation(self, win_for_parent) -> None:
        """
        Recursively updates statistics for this node and all parents based on the game outcome.

        Parameters: win_for_parent (int): The outcome of the game for the parent node. 
                            1 represents a win, -1 represents a loss, and 0 represents a tie.
                            Decimal values are based on heuristic evaluations.

        This method updates the visit count, winning statistics, and upper confidence bound (UCB) for the node.
        The UCB is calculated to balance exploration and exploitation in the Monte Carlo Tree Search algorithm.
        """
        self.current_node_visit_counter += 1
        
        if self.parent:
            self.parent.backpropogation(-win_for_parent)
                        
            if win_for_parent > 0:
                self.parent_win_counter += win_for_parent
            elif not win_for_parent:
                self.parent_win_counter += 0.5

            exploration_term = sqrt(2) * sqrt(log(self.parent.current_node_visit_counter) / self.current_node_visit_counter)
            winning_ratio = self.parent_win_counter / self.current_node_visit_counter
            self.upper_confidence_bound = winning_ratio + exploration_term
    
class MonteCarloTreeSearch():
    '''
    State Representation: Represent the current state of the checkers game. This includes the positions of pieces on the board, the current player's turn, and any additional relevant information.
   
    Selection and Expansion: Use MCTS to traverse the tree, selecting nodes based on UCB values. At each iteration, expand the tree by adding child nodes corresponding to possible moves in the game.
   
    Simulation: Perform simulations (rollouts) from the newly added nodes by making random moves until a terminal state (win, lose, or draw) is reached.
   
    Backpropagation: Update the statistics of nodes along the path from the leaf node to the root based on the outcome of the simulation.
   
    Decision: After a certain number of iterations or a specified time limit, select the move that corresponds to the child node with the highest visit count.
    '''
    def __init__(self, root):
        self.root_node = root
          
    def tree_search(self, time_limit) -> Move:
        '''
        Performs Monte Carlo Tree Search until time runs out.
        Returns the best move.
        '''
        break_time = time() + time_limit
                
        while time() < break_time:
            node = self.selection(self.root_node)
            
            temp_board = deepcopy(node.board)
            temp_color = node.color
            win_val = temp_board.is_win({1: 2, 2: 1}[temp_color])
            
            while not win_val:
                possible_moves = temp_board.get_all_possible_moves(temp_color)
                i = random.randint(0, len(possible_moves) - 1)
                j = random.randint(0, len(possible_moves[i]) - 1)
                temp_move = possible_moves[i][j]
                
                temp_board.make_move(temp_move, temp_color)
                win_val = temp_board.is_win(temp_color)
                temp_color = {1: 2, 2: 1}[temp_color]
    
            if win_val == {1: 2, 2: 1}[node.color]:
                win_for_parent = 1
            elif win_val == -1:
                win_for_parent = 0
            elif win_val == node.color:
                win_for_parent = -1
                
            node.backpropogation(win_for_parent)
            
        return self.the_chosen_one()

    def the_chosen_one(self) -> Move:
        '''
        Node with the highest visit count is selected
        '''
        best_child = max(self.root_node.children.items(), key=lambda x: x[1].current_node_visit_counter)
        return best_child[0]
    
    def selection(self, node) -> 'TreeNode':
        '''
        Recursively traverses the Monte Carlo Tree Search (MCTS) tree to locate a terminal node with the highest 
        Upper Confidence Bound (UCB) value. Subsequently, it expands a new unexplored node for further exploration 
        and exploitation in the search space.
        '''
        if len(node.children) == 0:
            return node
        if None not in node.children.values():
            sorted_children = sorted(node.children.values(), key=lambda x: x.upper_confidence_bound, reverse=True)
            return self.selection(sorted_children[0])
        for move, child in node.children.items():
            if child is None:
                node.children[move] = TreeNode(node.board, {1: 2, 2: 1}[node.color], move, node)
                return node.children[move]