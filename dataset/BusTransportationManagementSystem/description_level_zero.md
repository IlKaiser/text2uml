## Classes

- **Driver** — name, unique auto-assigned ID, sick-leave flag (cannot be scheduled while on sick leave)
- **Bus** — unique licence plate (up to 10 characters, inclusive), repair-shop flag (cannot be assigned to a route while in the repair shop)
- **Route** — unique number (city-staff-determined, max 9999)
- **Shift** — type (morning, afternoon, night), for a particular bus on a particular day
- **CityStaff** — assigns buses to routes, posts driver schedules, adds/deletes drivers and buses (no update)
- **BTMS** — the system managing drivers, buses, routes, shifts, and the daily overview
- **Overview** — per-day, per-route display (licence plates of assigned buses, entered shifts, IDs and names of assigned drivers; sick drivers or in-repair buses highlighted)

## Relationships

- CityStaff (1) assigns Buses to Routes, up to a year in advance, several per day (Route -> 0..* Buses per day).
- Bus (1 per day -> 0..1 Route), it may take different Routes on different days.
- Route (1) has exactly three Shifts (1 -> 3: morning, afternoon, night).
- CityStaff (1) posts schedules, assigning Drivers to Shifts, up to a year in advance (Shift, for a particular Bus on a particular day, 1 -> 0..* Drivers).
- Driver (1 -> 0..* Shifts per day, no limit, possibly two at the same time), city staff assigning each of them to a shift for a particular bus on a particular day.
- Overview (1, per day) shows Routes (1 -> 0..*) with their assigned Buses, entered Shifts, and Drivers (IDs and names), it highlighting sick Drivers and in-repair Buses.
