## Classes
- House — address, building material (wood, brick, or concrete)
- Basement — size, name
- EarthBasement — humidity level (a Basement)
- ConcreteBasement — (a Basement)
- SemiDetachedHouse — own garden (yes/no), number of windows (a House)
- Carport — double carport (yes/no), flat roof (yes/no)
- Garage — automatic garage door (yes/no)
- Company — name, address
- JobLog — number of hours, agreed price

## Relationships
- House and Basement, optional (1 -> 0..1).
- Basement and its subtypes (EarthBasement, ConcreteBasement), exactly two disjoint types (complete).
- SemiDetachedHouse and Carport, exactly two (1 -> 2), alternative to Garage (either/or).
- SemiDetachedHouse and Garage, one (1 -> 1), alternative to the two Carports (either/or).
- House and Company, serviced by several (0..* -> 0..*), each pairing carrying out a job recorded in a JobLog (1 -> 0..*).
