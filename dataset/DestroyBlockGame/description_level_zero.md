## Classes

- **User** — unique username, password (same for player and admin), login choice of admin mode or play mode.
- **Player** — always-present role of a user, three lives at start, at most one game at a time (no parallel play).
- **Admin** — optional role of a user, sole creator and designer of a game.
- **Game** — unique name, publication status, save at level end or on pause, resumable, next level only on player confirmation, minimum speed, speed increase factor, maximum length, minimum length.
- **HallOfFame** — total scores, high scores for competition.
- **Block** — color, points (1–1000, per admin), one grid cell.
- **Level** — number (Level 1 up to max 99), starting block arrangement, optional random flag, fixed block count at start.
- **GridCell** — position (e.g., 1/1, 2/1, 1/2, ...).
- **Ball** — certain speed and direction, drop from center, bounce off top and two side walls, out-of-bounds at bottom wall.
- **Paddle** — middle-bottom position, left/right movement by player, length from maximum to minimum.

## Relationships

- User and Player (1 -> 1), it is always exactly one of them.
- User and Admin (1 -> 0..1), it is optionally one of them.
- Admin and Game (1 -> 0..*, its sole creation right), one of them per game (Game 1 -> 1).
- Game and HallOfFame (1 -> 1), it has its own.
- Admin and Block (1 -> 1..*), it defines their set with color and points 1–1000.
- Game and Level (1 -> 1..99, per admin), it numbers them from Level 1.
- Level and Block (1 -> 1..*, its starting arrangement per admin, same count at start, optional random).
- Block and GridCell (1 -> 1), it sits in one of them.
- Game and Ball (1 -> 1), it centers the ball, its minimum speed rising per level.
- Player and Paddle (1 -> 1), they move it left/right, its length from maximum to minimum per level.
- Player and Game (0..* -> 0..*, play only on its publication, same one replayable, different ones, one at a time), a user possibly an admin for another and never both for it.
- Ball and Block (1 -> 0..*), on hit it bounces, they disappear, the player scores their points, its last-block hit advancing the level.
- Player and HallOfFame (1..* -> 1), they compete for high score in it, their total score displayed at game end on all-lives loss or last-level finish.
