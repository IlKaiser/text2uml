## Classes
- **Sober** — taxi company, operates its own fleet of self-driving cabs, plants a tree per customer with 20 ride-share bookings.
- **Cab** — self-driving car (Sober-owned or privately registered).
- **CarOwner** — registers its own car as a Sober cab (taxi service during non-use), tracked by Sober.
- **Customer** — number, name; lead customer (the payer) for ride-hail.
- **RideHail** — immediate, on-demand; time of pick-up/drop-off, location of pick-up/drop-off, ride duration, distance, number of passengers (max six), fee (time- and distance-based), type of request (via Sober App or hand-waving, its gesture identified by Sober's deep-learning image recognition), number and name of lead customer.
- **RideShare** — a.k.a. carpooling (reduces costs, traffic congestion, carbon footprint); time of pick-up/drop-off, location of pick-up/drop-off, ride duration, distance, number and names of all customers (max ten), upfront negotiated fee (flexible pricing, more customers → lower fee per customer).
- **Accident** — date, location, damage amount.

## Relationships
- **Sober** and **Cab** (1 -> 0..*, its own fleet plus privately registered cars).
- **CarOwner** and **Cab** (1 -> 0..*, its registered private cars).
- **Sober** and **CarOwner** (1 -> 0..*, tracked by it for privately registered cabs).
- **Sober** and its two service types, **RideHail** and **RideShare** (it offers both).
- **RideHail** and **Customer** (1 -> 1..6, one of them the lead customer who pays).
- **RideShare** and **Customer** (1 -> 1..10).
- **Cab** and **Accident** (1 -> 0..*, per car).
