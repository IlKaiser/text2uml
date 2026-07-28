## Classes

- **Cruise** — number, name, start date, end date
- **Provider** — name, location
- **Employee** — name, social security number
- **Guest** — name, VIP (yes/no)
- **Ship** — name, number of decks, length, number of passengers
- **Booking** — booking number (unique), booking date
- **TravelAgency** — name, address
- **DestinationArea** — name, sea name
- **Show** — title, duration
- **Ticket** — (issued per cruise)

## Relationships

- **Provider** organizes **Cruise** (1 -> 0..*), a specific one.
- **Cruise** requires **Employee** (1 -> many), each of them holding a name and social security number.
- **Cruise** and **Guest** (many -> many), it offering participation across several cruises.
- **Ship** and **Cruise** (1 -> 0..*), it usable for different ones.
- **Cruise** and **TravelAgency** (0..* -> 1), booked through it via **Booking**.
- **Cruise** heads for **DestinationArea** (0..* -> 1), a specific one.
- **Cruise** and **Show** (1 -> 0..*), several of them offered to guests.
- **Cruise** and **Ticket** (1 -> 0..*), several of them issued, each belonging to exactly one cruise.
- **Guest** and **Ticket** (1 -> 0..*), any number of them, only one per cruise.
