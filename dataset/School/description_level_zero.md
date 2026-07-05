## Classes
- **Person** — firstName, lastName (abstract superclass of Student and Teacher)
- **Student** — firstName, lastName
- **Teacher** — firstName, lastName, email
- **School** — name
- **Room** — name (abstract superclass of Classroom and OtherRoom)
- **Classroom** — name, capacity
- **OtherRoom** — name, size (square meters; e.g. gym)
- **ClassGroup** — name (e.g. 5a)

## Relationships
- Student and School (attendance, exactly one school per student, 1 -> 0..*).
- Teacher and School (teaching, maximum three schools per teacher, 0..3), with hours known per link.
- Teacher and School (principal role, chosen from teachers, exactly one principal per school, only one school per principal, 1 -> 0..1).
- School and Room (it consists of a large number of rooms, 1 -> 0..*).
- ClassGroup and Student (between 10 and 31 students per group, exactly one group per student, 1 -> 10..31).
- ClassGroup and Classroom (exactly one classroom per group and vice versa, 1 -> 1).
