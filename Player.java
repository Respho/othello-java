import java.util.*;
import java.io.*;
import java.math.*;

class Player
{
	//CodinGame format
	//White = 1, Black = 0, Empty = .
	//ID = 0 means you are black player
	/*
		........
		........
		........
		...10...
		...01...
		........
		........
		........
	*/

	//othellosaurus format
    //White = 2, Black = 1, Empty = 0
    //Move format, int move, (char)('a' + move % 8)  + "" + (8 - move / 8)
    //Inner data representation is bottom row first
    //Interface uses upper row first
	/*
	public static final int[] START = new int[]
	    {0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 1, 2, 0, 0, 0, 
		 0, 0, 0, 2, 1, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 Board.BLACK};
    */

    public static void main(String args[])
    {
        Scanner in = new Scanner(System.in);
        int id = in.nextInt(); // id of your player.
        System.err.println("Id " + id);
        int boardSize = in.nextInt();

    	Othello othello = new Othello();

        int nextToMove = (id == 0) ? Board.BLACK : Board.WHITE;
        //System.err.println("nextToMove " + nextToMove);

        // game loop
        while (true)
        {
        	//Read and populate array
        	int[] BOARD = new int[]
			    {0, 0, 0, 0, 0, 0, 0, 0, 
				 0, 0, 0, 0, 0, 0, 0, 0, 
				 0, 0, 0, 0, 0, 0, 0, 0, 
				 0, 0, 0, 1, 2, 0, 0, 0, 
				 0, 0, 0, 2, 1, 0, 0, 0, 
				 0, 0, 0, 0, 0, 0, 0, 0, 
				 0, 0, 0, 0, 0, 0, 0, 0, 
				 0, 0, 0, 0, 0, 0, 0, 0, 
				 nextToMove};

            for (int i = 0; i < boardSize; i++)
            {
                String line = in.next(); // rows from top to bottom (viewer perspective).
                //System.err.println(line);
                for (int j = 0; j < 8; j++)
                {
                	int piece = 0;
                	//Workaround to flip the board
                	if (id == 0)
                	{
	                	if (line.charAt(j) == '1') piece = 1; //Black
	                	if (line.charAt(j) == '0') piece = 2; //White
                	}
                	else
                	{
	                	if (line.charAt(j) == '1') piece = 2; //Black
	                	if (line.charAt(j) == '0') piece = 1; //White
                	}
                	BOARD[i * 8 + j] = piece;
                }
            }

            //
            Board board = new Board(BOARD);
            othello.updateBoard(board);
			//System.err.println(board.toString());

            //
            int actionCount = in.nextInt(); // number of legal actions for this turn.
            for (int i = 0; i < actionCount; i++)
            {
                String action = in.next(); // the action
            }

            int move = othello.getComputerMove();
            String notation = Utils.getMoveNotation(move);
            System.err.println("bestMove " + move + " " + notation);
			System.out.println(notation);
        }
    }

    public static void mainX(String args[])
    {
    	Othello othello = new Othello();

    	othello.getComputerMove();
    	othello.getComputerMove();
    }
}

class Othello
{
	private Board gameBoard; // currently displayed position
	private final Stack<Board> gameHistory; // positions that have occurred so far

    private double maxTimeSeconds = 0.145;

	private String outputText;

	// Agents controlling white and black
	// Weights for end game found through linear regression on the final score
	// Other weights found through hill climbing algorithm
	private final Agent whiteBot =
		new Agent(new Evaluator(new int[][] {
				{8, 85, -40, 10, 210, 520},
			    {8, 85, -40, 10, 210, 520},
			    {33, -50, -15, 4, 416, 2153},
			    {46, -50, -1, 3, 612, 4141},
			    {51, -50, 62, 3, 595, 3184},
			    {33, -5,  66, 2, 384, 2777},
			    {44, 50, 163, 0, 443, 2568},
			    {13, 50, 66, 0, 121, 986},
			    {4, 50, 31, 0, 27, 192},
			    {8, 500, 77, 0, 36, 299}},
			new int[] {0, 55, 56, 57, 58, 59, 60, 61, 62, 63}),
			false, 100, maxTimeSeconds);
	private final Agent blackBot =
		new Agent(new Evaluator(new int[][] {
				{8, 85, -40, 10, 210, 520},
			    {8, 85, -40, 10, 210, 520},
			    {33, -50, -15, 4, 416, 2153},
			    {46, -50, -1, 3, 612, 4141},
			    {51, -50, 62, 3, 595, 3184},
			    {33, -5,  66, 2, 384, 2777},
			    {44, 50, 163, 0, 443, 2568},
			    {13, 50, 66, 0, 121, 986},
			    {4, 50, 31, 0, 27, 192},
			    {8, 500, 77, 0, 36, 299}},
			new int[] {0, 55, 56, 57, 58, 59, 60, 61, 62, 63}),
			false, 100, maxTimeSeconds);

	/** Creates a new GUI */
	public Othello()
	{
    	gameBoard = new Board();
		gameHistory = new Stack<Board>();
	}

    public void updateBoard(Board board)
    {
    	gameBoard = board;
    }

	/** Plays the move at (x, y) if it is legal */
	/*
	public void tryMove(int x, int y) {
		if(gameBoard.moveLegal(x, y)) {
			gameHistory.push(gameBoard);
			gameBoard = new Board(gameBoard, Utils.getIndex(x, y));
			if(gameBoard.legalMoves == 0) { // forced pass
				gameBoard = new Board(gameBoard, Board.PASS);
			}
		}
	}
	*/

    public int getComputerMove()
    {
    	int move;
		if (gameBoard.mover == Board.WHITE) {
		    move = whiteBot.getMove(new Board(gameBoard));
		} else {
			move = blackBot.getMove(new Board(gameBoard));
		}

        return move;
    }

	/** Prints white's static evaluation for the board and the current features */
	public void printEval() {
		int m = gameBoard.mover == Board.WHITE ? 1 : -1;
		System.err.println("SCORE: " + m * whiteBot.e.eval(gameBoard));
		System.err.println("MOBILITY: " + m * Evaluator.mobility(gameBoard));
		System.err.println("FRONTIER: " + m * Evaluator.frontier(gameBoard));
		System.err.println("PLACEMENT: " + m * Evaluator.placement(gameBoard));
		System.err.println("STABILITY: " + m * Evaluator.stability(gameBoard));
		System.err.println("CORNER GRAB: " + m * Evaluator.cornerGrab(gameBoard));
		System.err.println();
	}

	/** Sets the text on the textDisplay to the board's score */
	public void setText() {
		boolean gameOver = gameBoard.gameOver;
		int whiteScore = Utils.bitCount(gameBoard.pieces[Board.WHITE]);
		int blackScore = Utils.bitCount(gameBoard.pieces[Board.BLACK]);
		int m =gameBoard.mover;
	}
}

/**
 * Class for bitboard utilities
 * Also stores look up tables for fast computation
 */
class Utils {
	// Whether pre-computations are done
	public static boolean precompuationsDone = false;

	// The amount of bitshifting necessary to move one square in the
	// given direction (up, right, up-right, up-left)
	public static final int[] shift = {1, 8, 9, 7};
	// gives the squares that can be shifted in the given direction and
	// orientation while staying on the board
	public static final long[][] shiftable =
		{{0xfefefefefefefefeL, 0x7f7f7f7f7f7f7f7fL},
		 {0xffffffffffffff00L, 0x00ffffffffffffffL},
		 {0xfefefefefefefe00L, 0x007f7f7f7f7f7f7fL},
		 {0x7f7f7f7f7f7f7f00L, 0x00fefefefefefefeL}};
	public static final long[][] edges =
		{{0x0101010101010101L, 0x8080808080808080L},
		 {0x00000000000000ffL, 0xff00000000000000L},
		 {0x01010101010101ffL, 0xff80808080808080L},
		 {0x80808080808080ffL, 0xff01010101010101L}};
	public static long[] frontierContributers = new long[4];
	// some other useful bitboards
	public static long corners = 0x8100000000000081L;
	public static long center = 0x00003c3c3c3c0000L;

	// stores the number of bits in the given 16-bit number
	public static final int[] bitCount = new int[65536];
	// stores a randomly generated bit string to be XORed with a board's hash
	// code for a piece at the given index and of the given color
	public static final int[][] hashChange = new int[64][2];
	// stores the bit string to be XORed with the board's hash code given the
	// white pieces, black pieces, and number of the given row
	public static final int[][][] rowHashChange = new int[256][256][8];
	// generalized the 4x4 SQUARE_VALUES in Evaluator to the whole board
	public static final int[][] fullSquareScore = new int[8][8];
	// stores the piece placement value for the given set of
	// white pieces, black pieces, and number of the given row
	public static final int[][][] rowScore = new int[256][256][8];
	// stores mobility score for given number of white and black moves;
	public static final int[][] mobilityScore = new int[64][64];

	/** Precomputes values for later look up */
	public static void precompute() {
		for(int i = 0; i < 4; i++) {
			frontierContributers[i] = ~(edges[i][0] | edges[i][1]);
		}

		// bitcount setup
		for(int i = 0; i < 65536; i++) {
			int n = i;
			int count = 0;
			while (n != 0) {
				count++;
				n &= (n - 1);
			}
			bitCount[i] = count;
		}

		// hash change setup
		for(int i = 0; i < 64; i++) {
			for(int j = 0; j < 2; j++) {
				for(int k = 0; k < 31; k++) {
					if(Math.random() < 0.5) {
						hashChange[i][j] |= (1 << k);
					}
				}
			}
		}

		// piece placement value setup
		for(int x = 0; x < 4; x++) {
			for(int y = 0; y < 4; y++) {
				fullSquareScore[x][y] = Evaluator.SQUARE_SCORE[x][y];
				fullSquareScore[7 - x][y] = Evaluator.SQUARE_SCORE[x][y];
				fullSquareScore[x][7 - y] = Evaluator.SQUARE_SCORE[x][y];
				fullSquareScore[7- x][7 - y] = Evaluator.SQUARE_SCORE[x][y];
			}
		}

		// precompute hashChange and score for an arbitrary row of pieces
		// iterate through all possible configurations of white and black pieces
		for(int white = 0; white < 256; white++) {
			for(int black = 0; black < 256; black++) {
				if((white & black) != 0) {
					continue;
				}
				// iterate through all rows
				for(int y = 0; y < 8; y++) {
					int score = 0;
					// iterate through all squares in the current row
					for(int x = 0; x < 8; x++) {
						int index = getIndex(x, y);
						if((white & (1 << x)) != 0) {
							rowHashChange[white][black][y] ^= hashChange[index][Board.WHITE];
							score += fullSquareScore[x][y];
						} else if((black & (1 << x)) != 0) {
							rowHashChange[white][black][y] ^= hashChange[index][Board.BLACK];
							score -= fullSquareScore[x][y];
						}
					}
					rowScore[white][black][y] = score;
				}
			}
		}

		// precompute mobility scores
		for(int i = 0; i < 64; i++) {
			for(int j = 0; j < 64; j++) {
				mobilityScore[i][j] =
					(int)Math.sqrt(Evaluator.MOBILITY_FACTOR * i) -
					(int)Math.sqrt(Evaluator.MOBILITY_FACTOR * j);
			}
		}

		precompuationsDone = true;
	}

	/** Prints the given bitboard (for debugging purposes) */
	public static void printBitboard(long BB) {
	    for(int y = 7; y >= 0; y--) {
	    	for(int x = 0; x <= 7; x++) {
	    		System.err.print((1 & (BB >> getIndex(x, y))) + " ");
	    	}
	    	System.err.println();
	    }
	    System.err.println();
	}

	/** Returns the index (0 - 63) corresponding to the square at (x, y) */
	public static int getIndex(int x, int y) {
		return x + 8*y;
	}

	/**
	 * Returns the notation for the square at index.
	 * Columns are labeled a-h from the left column to the right one
	 * Rows are labeled 1-8 from the top to the bottom.
	 */
	public static String getMoveNotation(int move) {
		return move == -1 ? "pass" : (char)('a' + move % 8)  + "" + (8 - move / 8);
	}

	/** Returns the number of ones in the given bit string */
	public static int bitCount(long b) {
		return bitCount[(int)(b & 65535)]
		     + bitCount[(int)((b >> 16) & 65535)]
		     + bitCount[(int)((b >> 32) & 65535)]
		     + bitCount[(int)((b >> 48) & 65535)];
	}

	/**
	 * Fast bitscan method I found online. Returns the index of the first bit
	 * in the given bitString
	 *
     * @author Matt Taylor
     * @return index 0..63
     * @param bb a 64-bit word to bitscan, should not be zero
     */
	private static final int[] foldedTable = {
    	63,30, 3,32,59,14,11,33,
    	60,24,50, 9,55,19,21,34,
    	61,29, 2,53,51,23,41,18,
    	56,28, 1,43,46,27, 0,35,
    	62,31,58, 4, 5,49,54, 6,
    	15,52,12,40, 7,42,45,16,
    	25,57,48,13,10,39, 8,44,
    	20,47,38,22,17,37,36,26,
    };
	public static int bitScanForward(long b) {
    	b ^= (b - 1);
        int folded = ((int)b) ^ ((int)(b >>> 32));
        return foldedTable[(folded * 0x78291ACF) >>> 26];
    }
}

/**
 * Entry for transposition table
 */
class TableEntry {
	public byte type; // type of entry (see final variables in Node)
	public int v; // value of this entry
	public byte depth; // depth of search which gave this entry's value

	/** Creates new TableEntry */
	public TableEntry(int v, byte type) {
		this.v = v;
		this.type = type;
		this.depth = Node.searchDepth;
	}
}

/**
 * Represents a Node in a game tree.
 *  Search uses:
 *  - Negamax search with alpha-beta pruning
 *  - Transposition tables
 *  - Iterative deepening with move ordering
 *  - History heuristic
 *  - Killer move heuristic
 *
 *  Optional:
 *   - Negascout search: This makes search faster in most positions but slower
 *     when what is considered the best line changes a frequently at high ply.
 *     Unfortunately, this is exactly when we want search to be deep, so in general
 *     it seems to (very slightly) hurt performance
 */
class Node {
	// types of stored evaluations in the transposition table
	public static final byte EXACT = 0;
	public static final byte LOWER_BOUND = 1;
	public static final byte UPPER_BOUND = -1;

	// just a big power of two
	public static final int WIN_MULTIPLIER = 4194304;

	// max ply at which to check if computations have gone over the time limit
	public static final int FORCE_STOP_PLY = 6;
	// num plies at which to record moves for visual display
	public static final int RECORD_MOVE_PLY = 4;

	public static int orderPly; // max ply at which to do move-ordering
	public static int transposePly; // max ply at which to check transposition
									// table for repeated position
	public static int hashPly; // max ply at which to enter nodes in
							   // transposition table
	public static int negascoutPly; // ply at which to use negascout algorithm

	public static long stopTime; // when to stop searching
	public static byte searchDepth; // depth at which to use static evaluation
	public static int nodesSearched; // number of nodes visited this search
	public static boolean doneStaticEval; // whether a static evaluation has
										  // been done this search
	public static Evaluator evaluator; // Evaluator for static evaluations
	public static final HashMap<Integer, TableEntry> transpositionTable =
		new HashMap<Integer, TableEntry>(1000000, 0.5f);

	public Board b; // Current board position for this search
	public byte ply; // Current ply for this search
	public int bestMove; // Best move found from b
	public int bestValue; // Score of the best move found from b
	public Node bestChild;  // This node's best child node

	// previously found good moves for history heuristic
	long strongMoves;
	long lastStrongMoves;

	/**
	 * Sets when to do various search algorithms based on the current
	 * search depth.
	 */
	public static void setDecisionPlies(boolean negascout) {
		orderPly = Math.min(searchDepth - 4, 9);
		transposePly = Math.min(searchDepth - 3, 10);
		hashPly = Math.max(orderPly + 1, transposePly);
		negascoutPly = negascout ? orderPly - 1 : -1;
	}

	/** Creates a new Node */
	public Node(Board b, byte ply) {
		this.b = b;
		this.ply = ply;
	}

	/** Search */
	public int negaMax(int alpha, int beta) {
	    // stop searching if gone over time
		if(searchDepth == 0 || (ply <= FORCE_STOP_PLY && System.nanoTime() > stopTime)) {
			searchDepth = 0;
			return 0;
		}

		nodesSearched++;

		// game is over, return score of final position
		if(b.gameOver) {
			return store(WIN_MULTIPLIER * Evaluator.pieces(b), EXACT);
		}

		// forced pass
		if(b.legalMoves == 0) {
			Node child = new Node(new Board(b, Board.PASS), (byte)(ply + 1));
			int childValue = -child.negaMax(-beta, -alpha);
			if(childValue > alpha) {
				if(childValue >= beta) {
					return store(childValue, LOWER_BOUND);
				}
				if(ply <= RECORD_MOVE_PLY) {
					bestMove = Board.PASS;
					bestValue = alpha;
					bestChild = child;
				}
				return store(childValue, EXACT);
			}
			return store(alpha, UPPER_BOUND);
		}

		// if we have seen this position before in the current search,
		// avoid repeated computation by using its stored value
		if(ply <= transposePly && transpositionTable.containsKey(b.hashCode())) {
			TableEntry e = transpositionTable.get(b.hashCode());
			if(e.depth == searchDepth) {
				if(e.type == EXACT) {
					return e.v;
				} else if(e.type == LOWER_BOUND) {
					alpha = Math.max(alpha, e.v);
				} else {
					beta = Math.min(beta, e.v);
				}
			}
			if(alpha >= beta) {
				return alpha;
			}
		}

		// at search depth, return static evaluation function
		if(ply >= searchDepth) {
			doneStaticEval = true;
			return store(evaluator.eval(b), EXACT);
		}

		// use history heuristic 2 plies after move ordering and killer move
		// heuristic after that
		boolean killerMoveHeuristic = true;
		boolean historyHeuristic = (ply == orderPly + 1 || ply == orderPly + 2);
		boolean prepareForHistory = (ply == orderPly - 1 || ply == orderPly);

		// move ordering for better alpha-beta pruning performance
		Board[] children = null;
		int numChildren = 0;
		if(ply <= orderPly) {
			children = new Board[31];
			while(b.legalMoves != 0) {
				Board c = children[numChildren] = new Board(b, b.getNextMove());
				c.value = -evaluator.eval(c);
				if(transpositionTable.containsKey(c.hashCode())) {
					TableEntry e = transpositionTable.get(c.hashCode());
					c.value += 67108864 * e.depth;
					c.value -= 4096 * (e.v + e.type);
				}
				numChildren++;
			}
			Arrays.sort(children, 0, numChildren);

			// record the best couple moves for history heuristic
			if(prepareForHistory) {
				lastStrongMoves = strongMoves;
				strongMoves = 0;
				for(int i = 0; i < numChildren/3; i++) {
					strongMoves |= (1L << children[i].lastMove);
				}
			}
		}

		// expand this node
		byte type = UPPER_BOUND;
		int n = 0;

		while(b.legalMoves != 0 || n < numChildren) {
			Board nextBoard;

			if(numChildren != 0) {
				// move ordering
				nextBoard = children[n++];
			} else if(historyHeuristic) {
				// history heuristic: try out moves that were found to be good previously first
				long moves = (b.legalMoves & lastStrongMoves);
				if(moves == 0) {
				 	historyHeuristic = false;
				 	nextBoard = new Board(b, b.getNextMove());
				} else {
					nextBoard = new Board(b, b.getNextMove(moves));
				}
			} else if(killerMoveHeuristic) {
				// killer move heuristic: try corner moves first
				long moves = (b.legalMoves & Utils.corners);
				if(moves == 0) {
					 killerMoveHeuristic = false;
					 nextBoard = new Board(b, b.getNextMove());
				} else {
					nextBoard = new Board(b, b.getNextMove(moves));
				}
			} else {
				// regular move generation
				nextBoard = new Board(b, b.getNextMove());
			}

			Node child = new Node(nextBoard, (byte)(ply + 1));

			// pass on the best couple moves for history heuristic
			if(prepareForHistory) {
				child.strongMoves = strongMoves;
				child.lastStrongMoves = lastStrongMoves;
			} else if(ply == orderPly + 1) {
				child.strongMoves = lastStrongMoves;
				child.lastStrongMoves = strongMoves;
			}

			// NegaScout search
			int childValue;
			if(ply <= negascoutPly && n > 1) {
				long l = nextBoard.legalMoves;
				childValue = -child.negaMax(-alpha - 1, -alpha);
				if(childValue > alpha && childValue < beta) {
					nextBoard.legalMoves = l;
					childValue = -child.negaMax(-beta, -childValue);
				}
			} else {
				childValue = -child.negaMax(-beta, -alpha);
			}

			// new best move found!
			if(childValue > alpha) {
				type = EXACT;
				alpha = childValue;
				if(ply <= RECORD_MOVE_PLY) {
					bestMove = nextBoard.lastMove;
					bestValue = alpha;
					bestChild = child;
				}
			}

			// alpha-beta pruning
			if(alpha >= beta) {
				return store(alpha, LOWER_BOUND);
			}
		}

		return store(alpha, type);
	}

	/** Stores the given value and entry type in the transposition table */
	public int store(int v, byte type) {
		if(ply <= hashPly) {
			transpositionTable.put(b.hashCode(), new TableEntry(v, type));
		}
		return v;
	}
}

/**
 * Othello static evaluator
 */
class Evaluator {
	public static final int MOBILITY_FACTOR = 10000;

	// weights for heuristics given number of pieces on the board
	public int[][] weightsForNumPieces;

	/**
	 * Constructs a new evaluator
	 * weightsForTimings is an array of weights. Timings is an array that
	 * determines at which number of pieces on the board to use the matching
	 * set of weights.
	 * When the current number of pieces is not set in timings, weights are
	 * linearly interpolated.
	 */
	public Evaluator(int[][] weightsForTimings, int[] timings) {
		weightsForNumPieces = new int[65][weightsForTimings[0].length];

		for(int m = 0; m <= 64; m++) {
			// determine which set of weights to use
			int w = 0;
			for(int i = 0; i < timings.length; i++) {
				if(m <= timings[i]) {
					w = i;
					break;
				}
			}

			// first set of weights: just return them
			if(w == 0) {
				weightsForNumPieces[m] = weightsForTimings[0];
				continue;
			}

			// linearly interpolate between the set of weights given for the
			// current number of moves and the previous set of weights
			double factor = ((double)m - timings[w - 1]) / (timings[w] - timings[w - 1]);
			for(int i = 0; i < weightsForTimings[w].length; i++) {
				weightsForNumPieces[m][i] =
					(int)Math.rint(factor * weightsForTimings[w][i]
			                    + (1 - factor) * weightsForTimings[w - 1][i]);
			}
		}
	}

	/**
	 * Returns a static evaluation for b
	 */
	public int eval(Board b) {
		int score = 0;
		int[] weights = weightsForNumPieces[b.numPieces];

		if(weights[0] != 0) {
			score += weights[0] * mobility(b);
		}
		if(weights[1] != 0) {
			score += weights[1] * frontier(b);
		}
		if(weights[2] != 0) {
			score += weights[2] * pieces(b);
		}
		if(weights[3] != 0) {
			score += weights[3] * placement(b);
		}
		if(weights[4] != 0) {
			score += weights[4] * stability(b);
		}
		if(weights[5] != 0) {
			score += weights[5] * cornerGrab(b);
		}

		return score;
	}

	/**
	 * Returns the number of legal moves available to the player about to move
	 * minus the number of legal moves available to the other player
	 */
	public static int mobility(Board b) {
		long opponentMoves = b.getMoves(b.opponent);
		return Utils.mobilityScore[Utils.bitCount(b.legalMoves)][Utils.bitCount(opponentMoves)];
	}

	/**
	 * Returns the number of spaces adjacent to opponent pieces minus the
	 * the number of spaces adjacent to the current player's pieces.
	 */
	public static final int frontier(Board b) {
		long moverPieces = b.pieces[b.mover];
		long opponentPieces = b.pieces[b.opponent];
		long spaces = ~(moverPieces | opponentPieces);

		long pfront = 0;
		long ofront = 0;
		// check for empty spaces in each direction
		for(int direction = 0; direction < 4; direction++) {
			int shift = Utils.shift[direction];
			long mask = Utils.frontierContributers[direction];
			pfront |= (spaces & ((moverPieces & mask)
					>>> shift));
			pfront |= (spaces & ((moverPieces & mask)
					<< shift));
			ofront |= (spaces & ((opponentPieces & mask)
					>>> shift));
			ofront |= (spaces & ((opponentPieces & mask)
					<< shift));
		}

		return Utils.bitCount(ofront) - Utils.bitCount(pfront);

	}

	/**
	 * Returns the number of pieces owned by the player about to move minus
	 * the number of pieces owned by the other player
	 */
	public static int pieces(Board b) {
		return Utils.bitCount(b.pieces[b.mover])
			 - Utils.bitCount(b.pieces[b.opponent]);
	}

	/**
	 * Returns the number of stable disks owned by the player about to move
	 * minus the number of stable disks owned by the other player
	 */
	public static int stability(Board b) {
		return stableDisks(b, b.mover) - stableDisks(b, b.mover ^ 1);
	}

	/**
	 * Returns the number of stable pieces owned by the player about to move minus
	 * the number of stable pieces owned by the other player
	 */
	public static int stableDisks(Board b, int p) {
		long pPieces = b.pieces[p];
		long stable = Utils.corners & pPieces;
		long newStable = 0;

		while(stable != newStable) {
			stable = newStable;
			newStable = pPieces;
			for(int dir = 0; dir < 4; dir++) {
				newStable &= (Utils.edges[dir][0] | Utils.edges[dir][1]
			        | (stable << Utils.shift[dir]) | (stable >>> Utils.shift[dir]));
			}
		}

		return Utils.bitCount(stable);
	}

	// value of controlling the given square
	public static final int[][] SQUARE_SCORE =
	{{ 100, -10,   8,   6},
	 { -10, -25,  -4,  -4},
	 {   8,  -4,   6,   4},
	 {   6,  -4,   4,   0}};
	/**
	 * Returns the piece placement score of the current player minus the piece
	 * placement score of the opponent. See SQUARE_SCORE for values.
	 */
	public static int placement(Board b) {
		long playerPieces = b.pieces[b.mover];
		long opponentPieces = b.pieces[b.opponent];
		int score = 0;

		// Use lookup table in Utils to compute placement value one row at at time
		for(int y = 0; y < 8; y++) {
			score += Utils.rowScore[(int)(playerPieces & 255)]
			                       [(int)(opponentPieces & 255)][y];

			playerPieces >>= 8;
			opponentPieces >>= 8;
		}

		return score;
	}

	/**
	 * Returns 1 if the current player can take a corner with its next move
	 * and 0 if otherwise.
	 */
	public static int cornerGrab(Board b) {
		return (b.legalMoves & Utils.corners) == 0 ? 0 : 1;
	}

	/**
	 * Returns 1 if the current player has parity (that is, they will move last)
	 * and -1 if otherwise (currently unused feature)
	 */
	public static int parity(Board b) {
		return b.opponent == (b.numPieces + b.mover) % 2 ? 1 : -1;
	}
}

/**
 * Representation for game state in Othello
 */
class Board implements Comparable<Board> {
	public static final int WHITE = 0;
	public static final int BLACK = 1;
	public static final int PASS = -1;
	
	// the starting configuration, used mainly for testing purposes
	// 1 for white piece, 2 for black, last index tells which player moves first
	public static final int[] START = new int[]
	    {0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 1, 2, 0, 0, 0, 
		 0, 0, 0, 2, 1, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 0, 0, 0, 0, 0, 0, 0, 0, 
		 Board.BLACK};
		 
	public int mover; // current player to move
	public int opponent; // the other player 
	public long[] pieces = new long[2]; // bitboards representing the pieces on the
										// board (one for white pieces, one for black)
	
	public boolean gameOver; // whether this game is over
	public int lastMove; // the last move played on this board
	public int numPieces; // number of pieces on this board
	public long legalMoves; // bitboard representing legal squares the current player can move
	
	public int value; // used for ranking boards (with the compareTo method)
	public int zobrist; // Zobrist hash code for this
					    // see http://en.wikipedia.org/wiki/Zobrist_hashing
	
	/**
	 * Creates a Board with the default configuration
	 */
	public Board() {
		this(START);
	}
	
	/**
	 * Creates a new board with the given start configuration
	 */
	public Board(int[] startConfig) {
		// do precomputations
		if(!Utils.precompuationsDone) {
			Utils.precompute();
		}
		
		// translate START to Board position
		for(int x = 0; x < 8; x++) {
			for(int y = 0; y < 8; y++) {
				// 7 - y because the bitboard is indexed so 0 is the lower left square
				int configIndex = Utils.getIndex(x, 7 - y); 
				int boardIndex = Utils.getIndex(x, y);
				// add pieces to the current board
				if(startConfig[configIndex] == 1) { 
					pieces[WHITE] |= (1L << boardIndex);
					numPieces++;
				} else if(startConfig[configIndex] == 2) {
					pieces[BLACK] |= (1L << boardIndex);
					numPieces++;
				}
			}
		}
		mover = START[64];
		opponent = mover ^ 1;
		gameOver = false;
		
		legalMoves = getMoves(mover);
	}
	
	/**
	 * Creates a duplicate of b
	 */
	public Board(Board b) {
		mover = b.mover;
		opponent = b.opponent;
		pieces = Arrays.copyOf(b.pieces, 2);
		gameOver = b.gameOver;
		legalMoves = b.legalMoves;
		lastMove = b.lastMove;
		numPieces = b.numPieces;
	}
	
	/**
	 * Creates new board given last position and current move
	 */
	public Board(Board lastBoard, int move) {
		// copy fields over
		mover = lastBoard.opponent;
		opponent = lastBoard.mover;
		pieces = Arrays.copyOf(lastBoard.pieces, 2);
		numPieces = lastBoard.numPieces;
		lastMove = move;
		
		// copy variables because this is faster than doing repeated array lookups
		long pPieces = pieces[mover];
		long oPieces = pieces[opponent];
		long oNegated = ~pieces[opponent];
		if(move != PASS) {
			numPieces++;
			
			long flips = 0;
			// compute bitboard for flipped pieces
			// try all directions (right, up, upright, upleft) forward and backward
			for(int direction = 0; direction < 4; direction++) {
				for(int orientation = 0; orientation < 2; orientation++) {
					// some bitboard magic
					// border is to stop wrapping the pieces on the edge over when we shift
					long border = Utils.shiftable[direction][orientation];
					int shift = Utils.shift[direction];
					long testFlips = 0;
					long loc = (1L << move);
					long tmp = 0;
					while(loc != 0) {
						loc &= border;
						loc = (orientation == 0 ? loc >>> shift : loc << shift);
						tmp = loc;
						loc &= oNegated;
						loc &= pPieces;
						testFlips |= loc;
					}
					
					if((tmp & oPieces) != 0) {
						flips |= testFlips;
					}
				}
			}
			
			// flip pieces
			pieces[opponent] |= flips; 
			pieces[mover] &= ~flips;
			// add in newly placed piece
			pieces[opponent] |= (1L << move);
		}
		
		// generate new moves and check if the game is over
		legalMoves = getMoves(mover);
		if(legalMoves == 0 && move == PASS) {
			gameOver = true;
		}
	}

	/**
	 * Generates a bitboard representing the possible moves for p
	 */
	public long getMoves(int p) {
		int o = p ^ 1;
		long m = 0;
		
		//copy variables because this is faster than doing repeated array lookups
		long pNegated = ~pieces[p];
		long oPieces = pieces[o];
		long oNegated = ~pieces[o];
		// try all directions (right, up, upright, upleft) forward and backward
		for(int direction = 0; direction < 4; direction++) {
			for(int orientation = 0; orientation < 2; orientation++) {
				// more bitboard magic
				long border = Utils.shiftable[direction][orientation];
				int shift = Utils.shift[direction];
				long potentials = pieces[p];
				
				// do initial shift once because must flip at least one piece to have a legal move
				potentials &= border;
				potentials = (orientation == 0 ? potentials >>> shift : potentials << shift);
				potentials &= pNegated;
				potentials &= oPieces;
				
				while(potentials != 0) {
					potentials &= border;
					potentials = (orientation == 0 ? potentials >>> shift : potentials << shift);
					potentials &= pNegated;
					m |= (potentials & oNegated);
					potentials &= oPieces;
				}
			}
		}
		return m;
	}
	
	/**
	 * Returns zobrist hash code for this
	 */
	public int hashCode() {
		// hash value already computed, so return it
		if(zobrist != 0) {
			return zobrist;
		}
		
		long white = pieces[WHITE];
		long black = pieces[BLACK];
		zobrist = (mover << 31);
		// Use lookup table in Utils to compute the hash code one row at at time
		for(int y = 0; y < 8; y++) {
			zobrist ^= Utils.rowHashChange[(int)(white & 255)]
			                              [(int)(black & 255)][y];
			white >>= 8;
			black >>= 8;
		}
		return zobrist;
	}
	
	/**
	 * Used to iterate through the legal moves of this board
	 */
	public int getNextMove() {
		return getNextMove(legalMoves);
	}
	
	/**
	 * Used iterate through the moves coinciding with the given bitboard
	 */
	public int getNextMove(long moves) {
		int moveIndex = Utils.bitScanForward(moves);
		legalMoves &= ~(1L << moveIndex);
		return moveIndex;
	}
	
	/**
	 * Returns true iff it is legal for player to place a piece at (x, y).
	 */
	public boolean moveLegal(int x, int y) {
		long mask = (1L << Utils.getIndex(x, y));
		return ((mask & legalMoves) != 0);
	}
	
	/**
	 * Returns the owner of the piece at (x, y) or -1 if that square is empty.
	 */
	public int pieceAt(int x, int y) {
		long mask = (1L << Utils.getIndex(x, y));
		if((pieces[WHITE] & mask) != 0) {
			return WHITE;
		} else if((pieces[BLACK] & mask) != 0) {
			return BLACK;
		}
		return -1;
	}
	
	/**
	 * Prints out an array that can be used to make a new board.
	 * see the variable START for format
	 */
	public String toString() {
		 String config = "{";
		 for(int y = 7; y >= 0; y--) {
		    for(int x = 0; x < 8; x++) {
		    	if(pieceAt(x, y) == WHITE) {
		    		config += "1, ";
		    	} else if(pieceAt(x, y) == BLACK) {
		    		config += "2, ";
		    	} else {
		    		config += "0, ";
		    	}
		    }
		    config += "\n ";
		 }
		 config += mover == WHITE ? "Board.WHITE};" : "Board.BLACK};";
		 return config;
	}
	
	/**
	 * Compares this to b
	 */
	public int compareTo(Board b) {
		return b.value - value;
	}
}

/**
 * Computer Othello player
 */
class Agent {
	public final Evaluator e; // determines this agent's static
					          // evaluation function
	public final int maxDepth; // the maximum ply this agent is allowed to search
	public long maxTime; // the maximum time in seconds this agent is
						 // allowed to think for a move.
	private final boolean negaScout; // whether to use the negaScout algorithm

	/** Creates a new agent */
	public Agent(Evaluator e, boolean negaScout, int maxDepth, double maxTime) {
		this.e = e;
		this.negaScout = negaScout;
		this.maxDepth = maxDepth;
		this.maxTime = (long)(maxTime * 1e9);
	}

	/** Searches and returns the agent's move */
	public int getMove(Board b) {
		//g.locked = true;
		//g.clearComputerOutput();

		Node n = new Node(b, (byte)0);
		long startTime = System.nanoTime();
		int bestMove = 0;

		// reset node fields
		Node.nodesSearched = 0;
		Node.evaluator = e;
		Node.transpositionTable.clear();

		// iterative deepening search
		for(Node.searchDepth = 1; Node.searchDepth <= maxDepth; Node.searchDepth++) {
			// stop evaluating if we're past our time limit
			Node.stopTime = startTime + maxTime;
			if(System.nanoTime() > Node.stopTime) {
				break;
			}
			Node.doneStaticEval = false;
			Node.setDecisionPlies(negaScout);

			// search
			n.b.legalMoves = n.b.getMoves(n.b.mover);
			n.negaMax(-Node.WIN_MULTIPLIER * 128, Node.WIN_MULTIPLIER * 128);

			if(System.nanoTime() > Node.stopTime) {
				break;
			}

			bestMove = n.bestMove;

			// print <current search depth> (<score of best move>) <optimal line>
			String s = Integer.toString(Node.searchDepth);
			if(Math.abs(n.bestValue) >= Node.WIN_MULTIPLIER) {
			    // game is solved: print winner and final score with optimal play
				s += (" (" + (
					n.bestValue * (n.b.mover == Board.WHITE ? 1 : -1) > 0 ?
					"White wins with score " : "Black wins with score ")
					+ (Math.abs(n.bestValue / Node.WIN_MULTIPLIER)) + ") ");
			} else {
				if(b.numPieces + Node.searchDepth >= 56) {
					// endgame: score printed so a stable disc is worth 1 point
					s += String.format(" (%1.2fe) ", n.bestValue /
									((float)Node.evaluator.weightsForNumPieces[Math.min(63, b.numPieces + Node.searchDepth)][4]));
				} else {
					// rest of the game: score printed so owning a corner is worth 1 point
					s += String.format(" (%1.2f) ", n.bestValue /
								(100.0 * Node.evaluator.weightsForNumPieces[b.numPieces + Node.searchDepth][3]));
				}
			}
			Node m = n;
			while(m.bestChild != null) {
				s += (Utils.getMoveNotation(m.bestMove) + " ");
				m = m.bestChild;
			}

            //g.extendOutput(s);
            System.err.println(s);

			// last search did no static evaluations so can stop searching
			// (the remainder of the game is solved)
			if(!Node.doneStaticEval) {
				break;
			}
		}

		long endTime = System.nanoTime();

        System.err.println("Searched " + Node.nodesSearched + " in " + (endTime - startTime) / 1000000 + "ms, " + (int)(1e9 * Node.nodesSearched / (endTime - startTime)) + "nps");
        /*
		g.extendOutput("NODES SEARCHED: " + Node.nodesSearched);
		g.extendOutput(String.format("SECONDS IN THOUGHT: %.3f\n",
				((endTime - startTime) / 1e9)));
		g.extendOutput(String.format("NODES PER SECOND: %.0f\n",
				(1e9 * Node.nodesSearched / (endTime - startTime))));
        */

		n.b.legalMoves = n.b.getMoves(n.b.mover);
		int move = bestMove;
		int x = move % 8;
	    int y = move / 8;
        /*
	    g.tryMove(x, y);
	    g.locked = false;
        */
	    return bestMove;
	}
}
