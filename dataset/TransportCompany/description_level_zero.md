## Classes

- **Employee** — name, social security number
- **Driver** — a kind of Employee (name, social security number)
- **Planner** — a kind of Employee (name, social security number, username, password)
- **AdministrativeStaff** — a kind of Employee (name, social security number, username, password, field of activity)
- **Customer** — name, billing address
- **Vehicle** — license plate number, mileage
- **RefrigeratedTruck** — a type of Vehicle
- **SmallVan** — a type of Vehicle
- **LoadingPlatform** — a type of Vehicle
- **BoxTruck** — a type of Vehicle
- **RepairShop** — name, address
- **Repair** — duration, cost
- **Order** — comment, start time, finish time, pick-up address, delivery address
- **PartnerCompany** — name, address

## Relationships

- Company employs Drivers (1 -> ~60) and other Employees (planners, administrative staff) (1 -> 6..7).
- Vehicle and Repair (1 -> 0..*), each Repair recording an Employee (who took it to the shop) and a RepairShop.
- RepairShop and Repair (1 -> 0..*).
- Driver and Order (1 -> 0..*, assigned), it also being known which Driver completed which Order with which Vehicle.
- Planner and Order (1 -> 0..*, the Planner having assigned it to a Vehicle).
- Vehicle and Order (1 -> 0..*).
- Customer and Order (1, client -> 0..*).
- Order and its placement, either with an AdministrativeStaff (1) or with a PartnerCompany (1).
