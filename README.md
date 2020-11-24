# othello-java
This is a command line adaptation of an Othello AI in Java, which facilitates automated benchmarking, and submitting to competitions, etc.

# Usage
For each turn, the AI reads from standard input and outputs move to standard output.
Accepts a single digit then a 8x8 board state.

```java
//White = 1, Black = 0, Empty = .
//ID = 0 means you are black player
/*
	0
	........
	........
	........
	...10...
	...01...
	........
	........
	........
*/
```

The chosen move is output in standard notation, a1 - h8.

# AI Features

This makes use of several advanced structures related to board games:
- Bitboard game state representations for fast computation.
- Transposition tables using Zobrist hashing.
- Negamax and NegaScout search with alpha beta pruning.
- Iterative deepening with move ordering. Saved evaluations are used for ordering moves at low ply, various heuristics are used for higher ply.
- A machine-learning-tuned static evaluation function with a special evaluator for endgames.

# AI Parameters

The strength of the AI is mainly determined by its allowed time.

```java
class Othello
{
    private double maxTimeSeconds = 0.2; // 200 milliseconds
}
```

# Run

Tested in JDK 1.5


# Credits

Othello AI - othellosaurus
https://github.com/clarkkev/othello-ai
