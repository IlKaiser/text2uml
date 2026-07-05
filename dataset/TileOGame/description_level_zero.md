## Classes
- **Game** — one played at a time (no pause, no save), its objective find hidden tile, it ends on hidden tile with the landing player winning.
- **Designer** — it defines whole game, its board layout, hidden tile, players' starting positions, action tile locations, deck of 32 action cards.
- **Player** — 2–4, they take turns (Player 1 starts, then Player 2, Player 3 optional, Player 4 optional), it rolls die, it moves piece.
- **PlayingPiece** — distinct color per piece, it moves along connected tiles.
- **Board** — it holds tiles and connection pieces.
- **Tile** — its color (white to black on visit), it connectable on right, left, top, bottom (at most one per side).
- **ActionTile** — regular tile turning to action, it reverts to regular tile for a designer-specified number of turns, its identity hidden until landed on.
- **ConnectionPiece** — it links two adjacent tiles, pile of 32 spares.
- **Deck** — 32 action cards, they chosen from predefined choices (extra-turn roll, connect two adjacent tiles from 32 spares, remove connection piece to spare pile, move piece to an arbitrary non-current tile, lose next turn).
- **ActionCard** — first card drawn on landing an action tile, its instructions followed.
- **Die** — it rolled by the current player.

## Relationships
- Game and Player (1 -> 2..4), they taking turns moving pieces along connected tiles.
- Designer and Game (1 -> 1), it defining the whole game and its board layout.
- Player and PlayingPiece (1 -> 1), each of a different color.
- Board and Tile (1 -> 1..*), it placing them and indicating hidden tile plus their action tile locations.
- Tile and Tile (right, left, top, bottom sides, at most one per side), they connected via connection pieces.
- Tile and ConnectionPiece (each connecting two adjacent tiles).
- ConnectionPiece and spare pile (32 spares), they added on removal, drawn on connect.
- Board and ActionTile (1 -> 0..*), each of them reverting to a regular tile for a designer-specified number of turns.
- Deck and ActionCard (1 -> 32), it drawing the first of them on an action-tile landing.
- Player and Die (1 -> 1), it rolling on its turn.
