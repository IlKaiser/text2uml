## Classes

- Object — ID (unique, ensures findability)
- Game — composed of players, deck, scoring table
- Player — role in game
- Deck — collection of cards (up to 48)
- Card — suit (diamonds, hearts, spades, clubs), comparison method for card height (lower, equal, higher)
- ScoringTable — collection of rounds
- Round — winning players, point value
- JackRound — round variant
- SoloRound — type of solo (trump, jack, queen)
- MandatorySoloRound — special case of solo round

## Relationships

- Game and Player (1 -> 4), it including exactly four of them.
- Game and Deck (1 -> 1).
- Game and ScoringTable (1 -> 1).
- Deck and Card (1 -> 0..48).
- ScoringTable and Round (1 -> 0..*, any number of them).
- Round and Player (winning players), (1 -> 1..*).
- Round and JackRound (a round may be one, inheritance).
- Round and SoloRound (a round may be one, inheritance; both jack and solo simultaneously possible).
- SoloRound and MandatorySoloRound (special case, inheritance).
- SoloRound and Player (solo player), (1 -> 1, exactly one).
- Object and all classes (each is one, ID inherited).
