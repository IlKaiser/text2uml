## Classes

- **Kinepolis** — owner, publisher of programs, canceller of shows.
- **Cinema** — located in a town (own town), holder of movie copies.
- **Theatre** — has a number of seats, hosts shows.
- **Show** — programmed per day, published (Wednesday–Tuesday), cancellable, has starting time.
- **Movie** — title played via copies.
- **Copy** — a cinema's own instance of a movie, at one premise, movable to another cinema (per programming).
- **Program** — published every Sunday for the next week (Wednesday to Tuesday).
- **Reservation** — online, seat-numbered, credit-card-paid, cancellable (up to 24h before start), made (up to 1h before start).
- **CreditCard** — debited only after the cancellation period.
- **Offer** — new seat (same movie, another theatre), acceptable or declinable, expiry 24h before start.
- **Ticket** — seat-numbered; at-entrance ones anonymous, non-cancellable, non-refundable; free one sent home.
- **Customer** — reservation holder, offer accepter/decliner, ticket buyer.
- **Display** — at the ticketing desk, available seats (per theatre seat count).

## Relationships

- Kinepolis owns many Cinemas (1 -> 1..*, each in a different town).
- Cinema has many Theatres (1 -> 1..*).
- It programs many Shows per day (Theatre, 1 -> 0..*).
- Cinema holds many Copies (1 -> 1..*, one per movie minimum, more for simultaneous plays).
- Movie has many Copies (1 -> 0..*).
- Copy sits at one Cinema (0..* -> 1, shown only in its theatres, movable to another cinema per programming).
- Kinepolis publishes one Program per week (1 -> 1, Wednesday to Tuesday).
- It publishes many Shows (Program, 1 -> 0..*).
- Show (published on the website) accepts many Reservations from Customers (1 -> 0..*, up to 1h before start).
- Customer makes many Reservations (1 -> 0..*, cancellable up to 24h before start).
- Reservation uses one CreditCard (0..* -> 1, debited only after the cancellation period).
- Kinepolis cancels a Show (1 -> 0..*, only with no seat reservations, else they are cancelled first).
- Cancelled Reservation yields one Offer, a new seat for the same movie in another theatre (1 -> 0..1, expiry 24h before start).
- Customer accepts or declines it (Offer, 1 -> 0..*).
- Customer (offer declined, expired, or none offerable) gets one free Ticket (1 -> 0..1, sent home).
- Customer (without a reservation) buys one Ticket at the entrance (1 -> 0..*, seat-numbered, anonymous, non-cancellable, non-refundable).
- Show has one ticketing-desk Display (1 -> 1, available seats per theatre seat count).
