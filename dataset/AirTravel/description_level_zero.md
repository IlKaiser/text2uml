## Classes
- **Aircraft** — name, type, year of manufacture, date of next inspection
- **Flight** — flight number, date
- **Passenger** — name, passport number
- **Ticket** — ticket number, price, upgrade desired (personalized, per flight)
- **PassengerAircraft** — number of seats (special Aircraft, used exclusively for passenger transport)
- **SeatCategory** — designation, entertainment program (offered/not), number of seats
- **Airline** — name
- **FlightAttendant** — name, date of employment, chief steward/stewardess flag
- **Pilot** — name, date of employment, license
- **Airport** — name, address, number of runways

## Relationships
- **Aircraft** performs **Flight**, each flight assigned to one aircraft (1 -> 0..*).
- **Passenger** and **Flight**, via **Ticket** (many-to-many, one ticket per passenger per flight), it takes part in several flights and each flight includes several of them.
- **Ticket** to **SeatCategory** (each ticket for one specific category).
- **PassengerAircraft** to **SeatCategory** (1 -> 0..*, several categories).
- **Aircraft** to **Airline** (0..1, optional membership), it employs several **FlightAttendant** and **Pilot** (1 -> 0..*).
- **FlightAttendant** to **Flight** (several of them work on it).
- **Pilot** to **Flight**, one as captain (flying the aircraft) plus one or two others as co-pilots (1 captain, 1..2 co-pilots).
- **Aircraft** to **Airport** (0..1 home airport).
- **Flight** to **Airport** (one departure, one destination).
