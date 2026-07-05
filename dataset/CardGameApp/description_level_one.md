You are designing a card game app for German clubs. You are thinking about the system's data structure in more detail. The customer gave you this specification:

A game has 4 players, a deck of cards, and a scoring table. A deck has up to 48 cards. Cards have a comparison method for card height. This method can return lower, equal, or higher. The suits of the cards are diamonds, hearts, spades, and clubs.

The scoring table has any number of rounds. Each round contains the winning players and the point value. A round can be a jack round and a solo round at the same time. A mandatory solo round is a special case of a solo round. A solo round has one type of solo. The solo can be a trump solo, a jack solo, or a queen solo. A solo round also has exactly one solo player.

Each object has a unique ID attribute. This attribute lets you find the object.
