## Classes

- **Song** — title, year, length, genre; identified only by its combination with Artist (same titles possible)
- **Artist** — name (unique identifier), date of birth, URL (e.g., Wikipedia page)
- **User** — (unique) ID, name, address
- **RegularUser** — a User, buys music
- **BusinessUser** — a User, VAT number, uploads/delivers content
- **Single** — a Song
- **Hit** — a Song
- **Album** — track number (position of each Hit), composed of Hits (no Singles)
- **Suggestion** — an Album turned into a recommendation

## Relationships

- Song and Artist (many -> exactly 1), it always belonging to one of them (its uniqueness derived from this combination).
- User and its two types, RegularUser and BusinessUser (it can be the former on some occasions, e.g. downloading a single or album, and the latter at other times, e.g. uploading self-made songs).
- BusinessUser and Song (1 -> 0..*), it uploading only individual ones, them classified as Single or Hit.
- RegularUser and Single (it buys the latter directly).
- Album and Hit (1 -> multiple, no Singles), each of them with its track number.
- RegularUser and Album (it composes the latter, from multiple Hits).
- Album and Suggestion (its RegularUser's version of it turned into a suggestion to other RegularUsers with similar purchasing behavior).
