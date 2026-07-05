You are designing a card game app for German clubs. You want to think about the system's data structure in more detail. The customer has given you the following specification.

A game consists of 4 players, a deck of cards, and a scoring table. A deck holds up to 48 cards. Each card has a comparison method for card height. This method can return lower, equal, or higher. The card suits are diamonds, hearts, spades, and clubs.

The scoring table consists of any number of rounds. Each round contains the winning players and the point value. A round can be both a jack round and a solo round. Mandatory solo rounds are a special case of solo rounds. Each solo round has one type of solo. This type can be a trump, jack, or queen solo. Each solo round also has exactly one solo player.

Every object has a unique ID attribute. This ensures that the object can be found.
