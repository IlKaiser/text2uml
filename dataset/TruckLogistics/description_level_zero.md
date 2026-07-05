## Classes
- **Location** — address.
- **Driver** — first name, last name.
- **Vehicle** — vehicle number.
- **Truck** — weight (a Vehicle).
- **Car** — number of seats (a passenger Vehicle).
- **Van** — a Vehicle.
- **Order** — date of execution.

## Relationships
- **Location** and **Driver** (1 -> 0..*), it grouping them.
- **Location** and **Vehicle** (1 -> 0..*), it grouping them.
- **Vehicle** and its subtypes **Truck**, **Car**, **Van** (generalization, 1 -> 3 types).
- **Driver** and **Order** (execution), each driver carrying a maximum of one order assigned at any one time (1 -> 0..1).
- **Order** and **Driver** (assignment), it having at least one (1 -> 1..*).
- **Order** and **Location** (origin), it always having one (1 -> 1).
- **Order** and **Location** (destination), it always having one (1 -> 1).
- **Driver** and **Vehicle** (driving record), it assigned up to one driver (0..1 -> 0..*).
