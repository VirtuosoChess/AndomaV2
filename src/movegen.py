from typing import Dict, List, Any
import chess
import sys
import time
from eval import evaluate_board, move_value, check_end_game

debug_info: Dict[str, Any] = {}

piece_value = {
    chess.PAWN: 100,
    chess.ROOK: 500,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.QUEEN: 900,
    chess.KING: 20000
}

MATE_SCORE     = 1000000000
MATE_THRESHOLD =  999000000


def next_move(depth: int, board: chess.Board, debug=True) -> chess.Move:
    """
    What is the next best move?
    """
    debug_info.clear()
    debug_info["nodes"] = 0
    t0 = time.time()

    move = minimax_root(depth, board)

    debug_info["time"] = time.time() - t0
    if debug == True:
        print(f"info {debug_info}")
    return move


def get_ordered_moves(board: chess.Board) -> List[chess.Move]:

    boardscore = evaluate_board_score(board)
    pawnstruct = evaluate_pawn_structure(board)
    open_line = evaluate_open_lines(board)
    threats = evaluate_threats(board)
    mobility = evaluate_mobility(board)
    end_game = check_end_game(board)

    def orderer(move):
        return move_value(board, move, end_game, mobility, threats, open_line, pawnstruct, boardscore)

    in_order = sorted(
        board.legal_moves, key=orderer, reverse=(board.turn == chess.WHITE)
    )
    return list(in_order)


def evaluate_mobility(board: chess.Board, color: chess.Color) -> int:
    mobility = 0
    for piece in board.pieces(color, chess.PAWN, chess.QUEEN):
        mobility += len(list(board.legal_moves for board in board.push(piece.move)))
    return mobility

def evaluate_threats(board: chess.Board, color: chess.Color) -> int:
    threats = 0
    for piece in board.pieces(chess.Color.opposite(color), chess.PAWN, chess.QUEEN):
        for move in board.legal_moves:
            new_board = board.copy()
            new_board.push(move)
            if new_board.is_check() and new_board.king(color) not in board.pieces(color, chess.KING):
                threats += 1
                break
    return threats

def evaluate_open_lines(board: chess.Board, color: chess.Color) -> int:
    open_lines = 0
    for piece in board.pieces(color, chess.ROOK, chess.QUEEN):
        for move in board.legal_moves:
            if move.piece_type == chess.PAWN:
                open_lines += 1
                break
    return open_lines

def evaluate_pawn_structure(board: chess.Board, color: chess.Color) -> int:
    pawn_structure = 0
    for square in chess.SQUARES:
        if board.piece_at(square) == chess.Piece(chess.PAWN, color):
            if board.pieces(color, chess.PAWN).count(square) > 1:
                pawn_structure -= 1
            elif board.pieces(chess.Color.opposite(color), chess.PAWN).count(square) == 0:
                pawn_structure += 1
            elif board.pieces(chess.Color.opposite(color), chess.PAWN).count(square) == 1:
                for direction in [chess.NORTH, chess.SOUTH]:
                    if board.piece_at(square + direction) == chess.Piece(chess.PAWN, chess.Color.opposite(color)):
                        pawn_structure -= 1
                        break
                else:
                    pawn_structure += 1
    return pawn_structure

def evaluate_board_score(board: chess.Board) -> int:
    score = 0
    for piece_type in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
        for color in [chess.WHITE, chess.BLACK]:
            score += piece_value[piece_type] * (len(board.pieces(color, piece_type)) - len(board.pieces(chess.Color.opposite(color), piece_type)))

    if board.is_checkmate():
        score += MATE_SCORE if board.turn == chess.WHITE else -MATE_SCORE
    elif board.is_stalemate():
        score = 0
    elif board.is_insufficient_material():
        score = 0
    elif board.is_seventyfive_moves():
        score = 0
    elif board.is_fivefold_repetition():
        score = 0

    return score


MOBILITY_FACTOR = 10
THREATS_FACTOR = 20
OPEN_LINES_FACTOR = 15
PAWN_STRUCTURE_FACTOR = 5

def evaluate_position(board: chess.Board) -> int:
    color = board.turn
    return (
        evaluate_board_score(board)
        + MOBILITY_FACTOR * evaluate_mobility(board, color)
        - THREATS_FACTOR * evaluate_threats(board, color)
        + OPEN_LINES_FACTOR * evaluate_open_lines(board, color)
        + PAWN_STRUCTURE_FACTOR * evaluate_pawn_structure(board, color)
    )


def minimax_root(depth: int, board: chess.Board) -> chess.Move:

    maximize = board.turn == chess.WHITE
    best_move = -float("inf")
    if not maximize:
        best_move = float("inf")

    moves = get_ordered_moves(board)
    best_move_found = moves[0]

    for move in moves:
        board.push(move)

        if board.can_claim_draw():
            value = 0.0
        else:
            value = minimax(depth - 1, board, -float("inf"), float("inf"), not maximize)
        board.pop()
        if maximize and value >= best_move:
            best_move = value
            best_move_found = move
        elif not maximize and value <= best_move:
            best_move = value
            best_move_found = move

    return best_move_found


def minimax(
    depth: int,
    board: chess.Board,
    alpha: float,
    beta: float,
    is_maximising_player: bool,
) -> float:

    debug_info["nodes"] += 1

    if board.is_checkmate():
        
        return -MATE_SCORE if is_maximising_player else MATE_SCORE

    elif board.is_game_over():
        return 0

    if depth == 0:
        return evaluate_board(board)

    if is_maximising_player:
        best_move = -float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player)

            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = max(
                best_move,
                curr_move,
            )
            board.pop()
            alpha = max(alpha, best_move)
            if beta <= alpha:
                return best_move
        return best_move
    else:
        best_move = float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player)
            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = min(
                best_move,
                curr_move,
            )
            board.pop()
            beta = min(beta, best_move)
            if beta <= alpha:
                return best_move
        return best_move
