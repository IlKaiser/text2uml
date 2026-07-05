## Classes

- **Location** — a museum-inside-the-museum (part of or an entire floor or wing), host of exhibitions
- **Exhibition** — its planning start (at least two years before its opening date), its stages of advancement
- **Employee** — museum staff (junior or senior), coordinator and/or coach
- **ExhibitionItem** — a desired item (e.g. early-period item, pencil drawing with corresponding painting, sunflower painting)
- **Piece** — a candidate work for an item (e.g. a "Sunflower" painting)
- **Collector** — possessor of candidate pieces

## Relationships

- **Location** and **Exhibition** (1 -> 0..*, a series of them developed for it).
- It and **Employee** (0..* -> 1, one of them assigned to it as coordinator).
- Its desired items (**ExhibitionItem**), a series of them defined first (1 -> 0..*).
- It and **Piece** (1 -> 1..*), some of them with one unique piece, some with several potential ones (from different collectors).
- It and **Collector** (0..* -> 1, each of them requested from one of them).
- **ExhibitionItem** and it (the system tracking, per it, what pieces are requested from which of them).
- **Employee** and it ("coaching", junior 0..* -> senior 1, one of them assigned to it as coach).
