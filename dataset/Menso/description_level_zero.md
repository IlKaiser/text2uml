## Classes

- **Company** — "De Lijn", possesses busses and trajectories, runs the MENSO project.
- **Bus** — carries a 'line' number, assignable to a trajectory.
- **Line** — groups overlapping trajectories, has a category (MENSO), partly same/partly different routes (e.g. "Line 2 Campus", "Line 2 Boskant").
- **Trajectory** — approved by town authorities, operational at start of next calendar year, may be suspended, may be cancelled.
- **TownAuthority** — approves, refuses without appeal, or requests changes to trajectories.
- **MensoTicket** — season ticket, valid one year, price per trip depends on line category, price not known in advance.
- **Customer** — MENSO user, personal data (home address, office address, working hours).
- **Card** — MENSO card, introduced in the ticket machine on entering the bus.
- **TicketMachine** — next to the bus door, registers time, date, line of the trip.
- **Trip** — time, date, line.
- **Invoice** — monthly, total amount, discount up to 20%.
- **Category** — assigned to lines, determines trip price.

## Relationships

- Company and Bus, it owning many of them (1 -> 0..*).
- Company and Trajectory (1 -> 0..*), each one subject to approval by TownAuthority.
- Trajectory and TownAuthority (0..* -> 1..*, approval refused without appeal, or change requested then re-requested, repeatable a number of times).
- Bus and Trajectory (0..* -> 0..1, assignable only after it becomes operational).
- Line and Trajectory (1 -> 1..*, overlapping, mostly the same but not entirely, e.g. shorter last section or bifurcation with two alternated end points).
- Bus and Line (0..* -> 1, via its 'line' number).
- Line and Category (1..* -> 1).
- Customer and MensoTicket (1 -> 0..1, subscribed via MENSO).
- Customer and Card (1 -> 0..1, forgotten card forcing the usual payment: buying a ticket or using another card type).
- Card and TicketMachine (0..* -> 1, introduced on entering the bus).
- TicketMachine and Trip (1 -> 0..*, registering its time, date, line).
- Trip and Line (0..* -> 1).
- Customer and Invoice (1 -> 0..*, monthly, in any order with payment, next one possibly received before the first is paid).
- Invoice and Trip (1 -> 1..*, past month's trips, discount up to 20% per total amount).
