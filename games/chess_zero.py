import chess
import gym
import gym_chess
import re
import string

import datetime
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (7, 8, 8)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(4672))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 50  # Maximum number of moves if game is not finished before
        self.num_simulations = 15  # Number of future moves self-simulated
        self.discount = 0.97  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Chess_zero()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):

        """
        Display the game observation.
        """
        self.env.render()
        # input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        # while True:
        #     try:
        #         row = int(
        #             input(
        #                 f"Enter the row (1, 2 or 3) to play for the player {self.to_play()}: "
        #             )
        #         )
        #         col = int(
        #             input(
        #                 f"Enter the column (1, 2 or 3) to play for the player {self.to_play()}: "
        #             )
        #         )
        #         choice = (row - 1) * 3 + (col - 1)
        #         if (
        #             choice in self.legal_actions()
        #             and 1 <= row
        #             and 1 <= col
        #             and row <= 3
        #             and col <= 3
        #         ):
        #             break
        #     except:
        #         pass
        #     print("Wrong input, try again")
        # return choice
        return self.env.human_to_action()

    def expert_agent(self):
        pass
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return self.env.action_to_string(action_number)
    
    def get_action_dict(self):
        
        return self.env.get_action_dict()


class Chess_zero:
    def __init__(self):
        self.board = chess.Board()
        self.player = 1
        self.env1 = gym.make('ChessAlphaZero-v0')
        self.env1.reset()
        self.board_obs = self.board_encode('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        # self.n_state = chess.Board()

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = chess.Board()
        self.player = 1
        self.env1.reset()
        return self.get_observation()

    def step(self, action):


        self.board.push(self.env1.decode(action))
        self.env1.step(action)

        done = self.have_winner() or len(self.legal_actions()) == 0

        # reward = 1 if self.have_winner() else 0
        if self.board.is_checkmate() and self.player == 1:
            reward = 1
        elif self.board.is_checkmate() and self.player == -1:
            reward = -1
        else:
            reward = 0

        # print("--------------------------{}--------------------".format(action))
        # print(self.env1.render())
        # print(type(self.board))
        # print(type(self.n_state))
        # if self.board.fen() == self.n_state.fen():
        #     print('TRUE')
        #     print(self.board.fen())
        #     print(self.n_state.fen())
        # print(type(self.env1.decode(action)))
        # print('------------------------{}-------------------------------'.format(self.env1.decode(action)))
        


        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        # print(self.board.copy())
        return self.board_encode(self.board.fen())

    def legal_actions(self):
        # legal = []
        # for i in range(9):
        #     row = i // 3
        #     col = i % 3
        #     if self.board[row, col] == 0:
        #         legal.append(i)
        # print(self.env1.legal_actions)
        return self.env1.legal_actions

    def have_winner(self):
        # Horizontal and vertical checks
        if self.board.is_checkmate() or self.board.is_stalemate():
            return True
        if self.board.is_insufficient_material() or self.board.can_claim_threefold_repetition():
            return True
        if self.board.can_claim_fifty_moves() or self.board.can_claim_draw():
            return True
        if self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves():
            return True
        return False
    
    def human_to_action(self):
        while True:
            ip_move = input('Enter a legal move')
            try:
                parsed_move  = self.board.parse_san(ip_move)
                break
            except ValueError:
                print('Illegal Move!')
        
        return self.env1.encode(parsed_move)

    # def expert_action(self):
    #     board = self.board
    #     action = numpy.random.choice(self.legal_actions())
    #     # Horizontal and vertical checks
    #     for i in range(3):
    #         if abs(sum(board[i, :])) == 2:
    #             ind = numpy.where(board[i, :] == 0)[0][0]
    #             action = numpy.ravel_multi_index(
    #                 (numpy.array([i]), numpy.array([ind])), (3, 3)
    #             )[0]
    #             if self.player * sum(board[i, :]) > 0:
    #                 return action

    #         if abs(sum(board[:, i])) == 2:
    #             ind = numpy.where(board[:, i] == 0)[0][0]
    #             action = numpy.ravel_multi_index(
    #                 (numpy.array([ind]), numpy.array([i])), (3, 3)
    #             )[0]
    #             if self.player * sum(board[:, i]) > 0:
    #                 return action

    #     # Diagonal checks
    #     diag = board.diagonal()
    #     anti_diag = numpy.fliplr(board).diagonal()
    #     if abs(sum(diag)) == 2:
    #         ind = numpy.where(diag == 0)[0][0]
    #         action = numpy.ravel_multi_index(
    #             (numpy.array([ind]), numpy.array([ind])), (3, 3)
    #         )[0]
    #         if self.player * sum(diag) > 0:
    #             return action

    #     if abs(sum(anti_diag)) == 2:
    #         ind = numpy.where(anti_diag == 0)[0][0]
    #         action = numpy.ravel_multi_index(
    #             (numpy.array([ind]), numpy.array([2 - ind])), (3, 3)
    #         )[0]
    #         if self.player * sum(anti_diag) > 0:
    #             return action

    #     return action

    def render(self):
        print(self.board)
    
    def action_to_string(self,action_number):
        return self.env1.decode(action_number)
    
    def get_action_dict(self):
        
        ac_dict ={}
        for ac in self.env1.legal_actions:
            ac_dict[self.env1.decode(ac)] = ac
        
        return ac_dict
    
    def board_encode(self,board_fen):
        pawn = numpy.zeros((8,8))
        knight = numpy.zeros((8,8))
        bishop = numpy.zeros((8,8))
        king = numpy.zeros((8,8))
        queen = numpy.zeros((8,8))
        rook = numpy.zeros((8,8))


        fen_arr = re.split('/',board_fen)
        # print(fen_arr)

        temp = fen_arr[7].split()
        fen_arr.pop()
        fen_arr.append(temp[0])
        # print(temp[1])

        if temp[1] == 'w':
            turn = numpy.ones((8,8))
        elif temp[1] == 'b':
            turn = numpy.full((8,8), -1, int)

        # print(turn)
        # print('--------------------------------------------------------')

        # print(fen_arr)

        # print(board)
        # print(type(board))

        i=0

        for str in fen_arr:
            c=0
            l=len(str)
            for char in str:
                if char in list(string.ascii_lowercase) or char in list(string.ascii_uppercase):
                    # print('Char is ',char)
                    # print('Index is ',c)
                    
                    #black pawn
                    if char =='p':
                        pawn[i,c] = -1
                    
                    #white pawn
                    if char == 'P':
                        pawn[i,c] = 1
                    
                    #black king
                    if char == 'k':
                        king[i,c] = -1
                    
                    #white king
                    if char == 'K':
                        king[i,c] = 1
                    
                    #black knight
                    if char == 'n':
                        knight[i,c] = -1
                    
                    #white knight
                    if char == 'N':
                        knight[i,c] = 1
                    
                    #black bishop
                    if char == 'b':
                        bishop[i,c] = -1
                    
                    #white bishop
                    if char == 'B':
                        bishop[i,c] = 1
                    
                    #black rook
                    if char == 'r':
                        rook[i,c] = -1
                    
                    #white rook
                    if char == 'R':
                        rook[i,c] = 1
                    
                    #black queen
                    if char == 'q':
                        queen[i,c] = -1
                    
                    #white queen
                    if char == 'Q':
                        queen[i,c] = 1
                    c+=1
                else:
                    c += int(char)
            i+=1


        board_arr = [pawn,knight,bishop,rook,king,queen,turn]
        board_arr = numpy.array(board_arr)

        # print(board_arr)
        # print(board_arr.size)
        # print(board_arr.shape)

        return board_arr
