## Classes

- **Center** — unique name (e.g., Fitplaza, my6pack), address (atomic).
- **Room** — number (unique within its center: 1, 2, 3, ...), maximum capacity.
- **Person** — first name, family name, birth date (combination unique).
- **Trainer** — a Person, diploma.
- **Session** — date, starting hour.
- **IndividualSession** — a Session (no trainer).
- **GroupSession** — a Session, type (e.g., aerobics, bodystyling).

## Relationships

- **Center** and **Room**: it has one or more of them (1 -> 1..*).
- **Room** and **Session**: at a given start hour of a given day, it hosts at most one of them (individual or group), so 1 -> 0..* overall.
- **Person** and **Session**: it participates in them as participant (0..* -> 0..*, including prospects with none yet), either individual or group.
- **Trainer** and **GroupSession**: it supervises them, each group session requiring exactly one trainer (0..* including interns with none yet -> exactly 1 per session).
- **Trainer** and **IndividualSession**: none (individual sessions run without a trainer, 0).
