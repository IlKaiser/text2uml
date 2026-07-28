## Classes
- GameArea — name (String)
- GameElement — x (protected long), y (private long) determining its position
- Shape — a GameElement
- Object — a GameElement
- Opponent — a GameElement, lives (protected int), can be killed

## Relationships
- GameArea and GameElement, composition (1 -> 1..*).
- GameArea and GamePiece, composition (1 -> 1..2).
- Shape and Shape, connection (any number, 0..*).
- GameArea and Opponent, known as the boss (1 -> 1), it being the single most important role.
- GameElement generalizes Shape, Object, and Opponent (it is either a shape, an object, or an opponent).
