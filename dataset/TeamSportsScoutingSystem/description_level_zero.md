## Classes

- **HeadCoach** — identifies designated player profiles, decides short list (with HeadScout).
- **Director** — makes official offer for a player.
- **Scout** — notes players on long list, submits scouting reports.
- **HeadScout** — evaluates long list, sets up scouting assignments, decides short list (with HeadCoach), recommends player for signing.
- **PlayerProfile** — designated profile for future signings.
- **TargetPosition** — code (e.g. GK, LB).
- **PlayerAttribute** — name, value.
- **Player** — noted, investigated, short-listed, offered.
- **LongList** — periodically evaluated.
- **ShortList** — short-listed players.
- **ScoutingAssignment** — investigates a specific player thoroughly.
- **ScoutingReport** — pros, cons, recommendation (e.g. key player, first team player, reserve team player, prospective player, not a good signing).

## Relationships

- HeadCoach identifies PlayerProfiles (1 -> 0..*), it detailing them for future signings.
- PlayerProfile includes TargetPositions (1 -> 0..*, e.g. GK, LB) and PlayerAttributes (1 -> 0..*, name/value).
- Scout notes Players (1 -> 0..*) on LongList (1 -> 0..1), they seeming to match a designated target profile (at any time).
- HeadScout evaluates LongList (1 -> 1, periodically) and sets up ScoutingAssignments (1 -> 0..*, for its team).
- ScoutingAssignment investigates one Player (1 -> 1, thoroughly).
- Scout, on completion of a ScoutingAssignment (1 -> 1), submits ScoutingReport (1 -> 1) about the Player.
- HeadCoach and HeadScout (comparing first scouting results for a PlayerProfile) move Players to ShortList (1 -> 0..*).
- ShortListed Player undergoes further ScoutingAssignments (1 -> 0..*, several rounds, some by HeadScout himself).
- HeadScout recommends Player for signing (1 -> 0..*), then Director makes an official offer (1 -> 0..*) for it.
