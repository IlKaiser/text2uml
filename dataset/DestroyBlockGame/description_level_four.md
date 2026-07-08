## Design Game

The DestroyBlock application has a game admin. The admin designs a DestroyBlock game. The application then has players. The players play the game. The players compete for an entry. This entry belongs to the game's hall of fame.

A user has a unique username. A user is always a player. A user is optionally an admin. A user has one password. The user uses this password as a player. The user uses this password as an admin. A user logs into the application. The user chooses the admin mode. The user may instead choose the play mode. Only an admin may create a game.

Each game has a unique name. Each game has its own hall of fame. The admin designs a game. The admin defines a set of blocks. Each block has a color. Each block is worth some points. The admin specifies these points. The points range between 1 and 1000.

A game has several levels. The admin defines these levels. The application numbers the levels. The numbering starts with Level 1. The maximum number of levels is 99. The admin handles each level. The admin specifies a starting arrangement. This arrangement holds blocks. The application uses a grid system. The application places each block in one cell. One block sits at the top left corner. This block has grid position 1/1. Another block sits to its right. This block has grid position 2/1. Another block sits below the first. This block has grid position 1/2. The pattern continues. The admin may also define a level as random. A random level selects the top blocks randomly. The blocks come from the defined set. The admin defines this set.

Each level begins with some blocks. The number of these blocks is the same for every level. The admin also defines this number. The ball has a speed. The speed increases with each level. The speed starts at its minimum. The paddle has a length. The length reduces gradually with each level. The length starts at its maximum. The length reduces to its minimum. The admin specifies the minimum speed. The admin specifies the speed increase factor. The admin specifies the maximum length. The admin specifies the minimum length.

## Play Game

A player can play a game. The game admin publishes the game first. A game begins. A level also begins. Then the application places the blocks. The blocks go to the top. The top belongs to the play area. The admin specifies these placements in the design phase. The application places the ball in the center. The center belongs to the play area. The ball drops in a straight line. The ball drops towards the bottom. The player has a paddle. The application positions the paddle in the middle. The paddle sits at the bottom. The player moves the paddle to the right or left. The paddle stays at the bottom. The player tries to bounce the ball towards the blocks. The ball moves at a certain speed. The ball moves in a certain direction. The ball bounces back from the top wall. The ball also bounces back from the right side wall. The ball also bounces back from the left side wall.

The ball may hit a block. Then the ball bounces back. Then the block disappears. Then the player scores points. These points belong to the hit block.

The ball may hit the last block. Then the player advances to the next level. The ball may reach the bottom wall. Then the ball is out-of-bounds. Then the player loses one life. The player starts a game with three lives. The player may lose all three lives. The player may instead finish the last level. Then the game ends. Then the application displays the total score. The application shows it in the game's hall of fame.

A level may end. Then the application saves the game. The player may pause the game. Then the application saves the game. The player can resume a paused game. The next level does not start automatically. The player confirms the next level. Then the next level starts.

A user may be a player for one game. The same user may be an admin for another game. A user cannot be both for the same game. Each game has only one admin. Players compete against each other. They compete for the high score. The high score belongs to the game's hall of fame. A player may play different games. A player may play the same game multiple times. A player plays only one game at any time. A player does not play games in parallel.
