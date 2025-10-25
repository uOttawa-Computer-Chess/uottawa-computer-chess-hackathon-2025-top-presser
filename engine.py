# engine.py
import time
import math
import random
import chess
import logging

logger = logging.getLogger(__name__)


class Engine:
    MATE = 1_000_000

    def __init__(self):
        self.board = chess.Board()
        self.nodes = 0
        self.stop_flag = False
        
        # --- Transposition Table (TT) & Zobrist ---
        self.tt: dict[int, dict] = {}
        self.tt_hits = 0

        # Zobrist tables (use a local RNG so we don't disturb global random)
        rng = random.Random(0xC0FFEE)
        self._zob_pieces = [[rng.getrandbits(64) for _ in range(64)] for _ in range(12)]
        self._zob_side   = rng.getrandbits(64)
        self._zob_castle = [rng.getrandbits(64) for _ in range(4)]   # WK, WQ, BK, BQ
        self._zob_ep     = [rng.getrandbits(64) for _ in range(8)]   # file A..H

        # PSTs (white perspective; black mirrored)
        self._pst = {
            "p": [
                0, 5, 5, -10, -10, 5, 5, 0,
                0, 10, -5, 0, 0, -5, 10, 0,
                0, 10, 10, 20, 20, 10, 10, 0,
                5, 20, 20, 30, 30, 20, 20, 5,
                5, 15, 15, 25, 25, 15, 15, 5,
                0, 10, 10, 20, 20, 10, 10, 0,
                5, 5, 10, -20, -20, 10, 5, 5,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            "n": [
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20, 0, 0, 0, 0, -20, -40,
                -30, 0, 10, 15, 15, 10, 0, -30,
                -30, 5, 15, 20, 20, 15, 5, -30,
                -30, 0, 15, 20, 20, 15, 0, -30,
                -30, 5, 10, 15, 15, 10, 5, -30,
                -40, -20, 0, 5, 5, 0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50,
            ],
            "b": [
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 10, 10, 5, 0, -10,
                -10, 5, 5, 10, 10, 5, 5, -10,
                -10, 0, 10, 10, 10, 10, 0, -10,
                -10, 10, 10, 10, 10, 10, 10, -10,
                -10, 5, 0, 0, 0, 0, 5, -10,
                -20, -10, -10, -10, -10, -10, -10, -20,
            ],
            "r": [
                0, 0, 0, 5, 5, 0, 0, 0,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                5, 10, 10, 10, 10, 10, 10, 5,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            "q": [
                -20, -10, -10, -5, -5, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 5, 5, 5, 0, -10,
                -5, 0, 5, 5, 5, 5, 0, -5,
                0, 0, 5, 5, 5, 5, 0, -5,
                -10, 5, 5, 5, 5, 5, 0, -10,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -20, -10, -10, -5, -5, -10, -10, -20,
            ],
            "k": [0] * 64,
        }

    # ---------- evaluation & helpers ----------
    @staticmethod
    def _piece_val(p: chess.Piece) -> int:
        t = p.piece_type
        if t == chess.PAWN:   return 100
        if t == chess.KNIGHT: return 320
        if t == chess.BISHOP: return 330
        if t == chess.ROOK:   return 500
        if t == chess.QUEEN:  return 900
        return 0
    
    # --- endgame helpers ---

    @staticmethod
    def _chebyshev(a: int, b: int) -> int:
        ar, af = divmod(a, 8)
        br, bf = divmod(b, 8)
        return max(abs(ar - br), abs(af - bf))

    def _material_counts(self, board: chess.Board) -> tuple[int, int]:
        w = b = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p:
                continue
            val = self._piece_val(p)
            if p.color == chess.WHITE: w += val
            else: b += val
        return w, b

    def _is_passed_pawn(self, board: chess.Board, sq: int, color: bool) -> bool:
        r, f = divmod(sq, 8)
        step = 1 if color == chess.WHITE else -1
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        files = [max(0, f-1), f, min(7, f+1)]
        rr = range(r+step, 8) if color == chess.WHITE else range(r+step, -1, -1)
        for rrk in rr:
            for ff in files:
                idx = rrk * 8 + ff
                p = board.piece_at(idx)
                if p and p.color == enemy and p.piece_type == chess.PAWN:
                    return False
        return True

    def _phase(self, board: chess.Board) -> int:
        phase = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p:
                continue
            if p.piece_type == chess.QUEEN:  phase += 8
            elif p.piece_type == chess.ROOK: phase += 4
            elif p.piece_type in (chess.BISHOP, chess.KNIGHT): phase += 2
        return min(24, phase)

    def _mvv_lva(self, board: chess.Board, m: chess.Move) -> int:
        if not board.is_capture(m):
            return 0
        victim = board.piece_at(m.to_square)
        attacker = board.piece_at(m.from_square)
        if not victim or not attacker:
            return 0
        return self._piece_val(victim) * 1000 - self._piece_val(attacker)

    def _evaluate(self, board: chess.Board) -> int:
        # Middlegame-like base (your PSTs + material)
        mg = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p:
                continue
            base = self._piece_val(p)
            r, c = divmod(sq, 8)
            idx_w = r * 8 + c
            idx_b = (7 - r) * 8 + (7 - c)
            pst = self._pst[p.symbol().lower()][idx_w if p.color == chess.WHITE else idx_b]
            mg += base + pst if p.color == chess.WHITE else - (base + pst)

        # Endgame score
        eg = self._evaluate_endgame(board)

        # Phase (0..24): 24 = MG, 0 = EG
        ph = self._phase(board)
        return (mg * ph + eg * (24 - ph)) // 24

    def _evaluate_endgame(self, board: chess.Board) -> int:
        score = 0

        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        center = [chess.D4, chess.E4, chess.D5, chess.E5]

        # King centralization
        for c in center:
            score += (4 - self._chebyshev(wk, c)) * 5
            score -= (4 - self._chebyshev(bk, c)) * 5

        # Approach enemy king
        kd = self._chebyshev(wk, bk)
        score += (7 - kd) * 4      # white wants to get closer
        score -= (7 - kd) * 4      # black too (from its POV)

        # Passed pawns
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p or p.piece_type != chess.PAWN:
                continue
            color = p.color
            if self._is_passed_pawn(board, sq, color):
                rank = chess.square_rank(sq) if color == chess.WHITE else (7 - chess.square_rank(sq))
                base = 12 + rank * 10
                ksq = wk if color == chess.WHITE else bk
                support = max(0, 4 - self._chebyshev(ksq, sq)) * 3
                bonus = base + support
                score += bonus if color == chess.WHITE else -bonus

        # Bishop pair
        def bp(color: bool) -> int:
            cnt = 0
            for sq in chess.SQUARES:
                p = board.piece_at(sq)
                if p and p.color == color and p.piece_type == chess.BISHOP:
                    cnt += 1
            return 20 if cnt >= 2 else 0

        score += bp(chess.WHITE)
        score -= bp(chess.BLACK)

        # Rook on 7th (2nd for black), near enemy king
        def r7(color: bool) -> int:
            target_rank = 6 if color == chess.WHITE else 1
            bonus = 0
            ek = bk if color == chess.WHITE else wk
            for sq in chess.SQUARES:
                p = board.piece_at(sq)
                if p and p.color == color and p.piece_type == chess.ROOK:
                    if chess.square_rank(sq) == target_rank and self._chebyshev(sq, ek) <= 3:
                        bonus += 18
            return bonus

        score += r7(chess.WHITE)
        score -= r7(chess.BLACK)

        # Drawish compression for very bare material
        non_pawn_white = non_pawn_black = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.piece_type != chess.PAWN:
                if p.color == chess.WHITE: non_pawn_white += 1
                else: non_pawn_black += 1
        if non_pawn_white <= 1 and non_pawn_black <= 1:
            score = int(score * 0.7)

        return score



    @staticmethod
    def _terminal_score(board: chess.Board, ply_from_root: int) -> int | None:
        if board.is_checkmate():
            return -(Engine.MATE - ply_from_root)
        # Draws
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves()
        ):
            return 0
        return None
    
        # --- TT/Zobrist helpers ---

    @staticmethod
    def _piece_index(piece: chess.Piece) -> int:
        # order: P N B R Q K (white) then p n b r q k (black)
        base = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}[piece.piece_type]
        return base if piece.color == chess.WHITE else base + 6

    def _zobrist(self, board: chess.Board) -> int:
        """Compute Zobrist key for board (pieces, side, castling, ep-file)."""
        key = 0
        # pieces
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                key ^= self._zob_pieces[self._piece_index(p)][sq]
        # side to move
        if board.turn == chess.BLACK:
            key ^= self._zob_side
        # castling rights: WK, WQ, BK, BQ
        if board.has_kingside_castling_rights(chess.WHITE):
            key ^= self._zob_castle[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            key ^= self._zob_castle[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            key ^= self._zob_castle[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            key ^= self._zob_castle[3]
        # en-passant (file only; simple & fast)
        if board.ep_square is not None:
            file_idx = chess.square_file(board.ep_square)
            key ^= self._zob_ep[file_idx]
        return key

    # Score packing for mates (store relative to root ply)
    def _to_tt(self, score: int, ply: int) -> int:
        if score >= self.MATE - 10000:
            return score + ply   # mate for side to move sooner → bigger
        if score <= -self.MATE + 10000:
            return score - ply
        return score

    def _from_tt(self, score: int, ply: int) -> int:
        if score >= self.MATE - 10000:
            return score - ply
        if score <= -self.MATE + 10000:
            return score + ply
        return score


    # ---------- move ordering ----------

    def _ordered_moves(self, board: chess.Board, first_move: chess.Move | None = None) -> list[chess.Move]:
        moves = list(board.legal_moves)
        if first_move in moves:
            moves.remove(first_move)

        captures, checks, castles, quiets = [], [], [], []
        for m in moves:
            if board.is_capture(m):  # includes en passant captures
                captures.append(m)
                continue

            if board.is_castling(m):
                castles.append(m)
                continue

            board.push(m)
            is_chk = board.is_check()
            board.pop()
            (checks if is_chk else quiets).append(m)

        # MVV-LVA for captures (EP is a capture, so it’s already here)
        captures.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)

        ordered = captures + checks + castles + quiets
        if first_move is not None:
            ordered.insert(0, first_move)
        return ordered



    # ---------- core negamax + alpha-beta ----------

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
        ply_from_root: int, deadline: float | None) -> tuple[int, chess.Move | None]:
        if self.stop_flag:
            return 0, None

        self.nodes += 1
        # periodic time check
        if deadline is not None and (self.nodes & 1023) == 0 and time.time() > deadline:
            self.stop_flag = True
            return 0, None

        # Terminal or leaf
        if depth == 0 or board.is_game_over():
            t = Engine._terminal_score(board, ply_from_root)
            if t is not None:
                return t, None
            s = self._evaluate(board)
            return (s if board.turn == chess.WHITE else -s), None

        alpha_orig = alpha

        # ---------- TT PROBE ----------
        zkey = self._zobrist(board)
        tt_move = None
        entry = self.tt.get(zkey)
        if entry and entry["depth"] >= depth:
            # Bound handling
            stored = self._from_tt(entry["score"], ply_from_root)
            if entry["flag"] == "EXACT":
                self.tt_hits += 1
                return stored, entry.get("best_move")
            elif entry["flag"] == "LOWER" and stored >= beta:
                self.tt_hits += 1
                return stored, entry.get("best_move")
            elif entry["flag"] == "UPPER" and stored <= alpha:
                self.tt_hits += 1
                return stored, entry.get("best_move")
            tt_move = entry.get("best_move")

        best_move = None
        a = alpha

        # Move loop (TT best move first if available)
        for m in self._ordered_moves(board, first_move=tt_move):
            board.push(m)
            sc, _ = self._negamax(board, depth - 1, -beta, -a, ply_from_root + 1, deadline)
            board.pop()
            if self.stop_flag:
                return 0, None
            val = -sc
            if val > a:
                a, best_move = val, m
                if a >= beta:
                    break

        # ---------- TT STORE ----------
        flag = "EXACT"
        if a <= alpha_orig:
            flag = "UPPER"
        elif a >= beta:
            flag = "LOWER"

        self.tt[zkey] = {
            "depth": depth,
            "score": self._to_tt(a, ply_from_root),
            "flag": flag,
            "best_move": best_move,
        }

        return a, best_move


    def _root_search(self, board: chess.Board, depth: int, alpha: int, beta: int,
                    deadline: float | None) -> tuple[int | None, chess.Move | None, bool]:
        best_move = None
        best_val = -10_000_000
        completed = True

        # Pre-order moves (good ordering makes time usage better)
        moves = self._ordered_moves(board)

        for m in moves:
            # Time check before starting a child
            if self.stop_flag or (deadline is not None and time.time() > deadline):
                completed = False
                break

            board.push(m)
            sc, _ = self._negamax(board, depth - 1, -beta, -alpha, 1, deadline)
            board.pop()
            if self.stop_flag:
                completed = False
                break

            val = -sc
            if val > best_val or best_move is None:
                best_val, best_move = val, m
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break  # cutoff

            # Light periodic time check even when no cutoff
            if (self.nodes & 1023) == 0 and deadline is not None and time.time() > deadline:
                completed = False
                break

        # If we couldn't finish any child, signal None score so caller can keep last completed iteration.
        if best_move is None:
            return None, None, False

        return best_val, best_move, completed


    # ---------- public drivers (root) ----------

    def search(self, board: chess.Board, depth: int, deadline: float | None = None) -> tuple[int, chess.Move | None]:
        self.nodes = 0
        self.stop_flag = False
        score, move = self._negamax(board, depth, -10_000_000, 10_000_000, 0, deadline)
        return score, move

    def search_fixed_depth(self, board: chess.Board, depth: int, time_limit_sec: float | None = None) -> tuple[int, chess.Move | None]:
        deadline = time.time() + time_limit_sec if time_limit_sec else None
        return self.search(board, depth=depth, deadline=deadline)

    def search_iterative(self, board: chess.Board, time_allocated: float, max_depth: int | None = None) -> tuple[int, chess.Move | None]:
        self.nodes = 0
        self.stop_flag = False
        deadline = time.time() + float(time_allocated)

        best_move = None
        evaluation = 0
        depth = 1
        while True:
            if max_depth is not None and depth > max_depth:
                break
            score, move = self._negamax(board, depth, -10_000_000, 10_000_000, 0, deadline)
            # if time up or stop requested, keep the best from last completed iteration
            if self.stop_flag or time.time() > deadline:
                break
            if move is not None:
                best_move = move
                evaluation = score
            depth += 1

        return evaluation, best_move

    def search_timed(self, board: chess.Board, time_limit_sec: float, max_depth: int | None = None) -> tuple[int, chess.Move | None]:
        self.nodes = 0
        self.stop_flag = False
        deadline = time.time() + float(time_limit_sec)

        best_move = None
        best_eval = 0
        depth = 1

        while True:
            if max_depth is not None and depth > max_depth:
                break
            if self.stop_flag or time.time() > deadline:
                break

            alpha, beta = -10_000_000, 10_000_000

            score, move, completed = self._root_search(board, depth, alpha, beta, deadline)

            # If we didn't even finish a single child at this depth, stop and return last completed result.
            if score is None or move is None:
                break

            # Update "last best so far" *even if* depth wasn't fully completed.
            best_move = move
            best_eval = score

            # Stop if time is over; otherwise go deeper.
            if not completed or time.time() > deadline:
                break

            depth += 1

        return best_eval, best_move
