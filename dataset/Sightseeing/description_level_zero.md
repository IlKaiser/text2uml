## Classes

- **City** — name, number of inhabitants
- **Attraction** — name, address
- **GuidedTour** — duration, number
- **Person** — name
- **Visitor** — a Person
- **Guide** — a Person
- **Discount** — amount

## Relationships

- City and Attraction (1 -> several), it offering its several Attractions.
- Attraction and GuidedTour (it, 1 -> various tours offered for it).
- GuidedTour and Visitor (it, 1 -> 0..20, max 20).
- GuidedTour and Guide (it, 1 -> 1, one required).
- Visitor and GuidedTour (certain Visitors, certain tours), via Discount (its amount known).
- Person and Visitor (it, a Visitor, with names).
- Person and Guide (it, a Guide, with names).
