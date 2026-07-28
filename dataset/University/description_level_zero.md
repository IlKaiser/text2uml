## Classes

- **University** — total number of employees.
- **Faculty** — name.
- **Institute** — name, address.
- **Employee** — social security number, name, email address (research or administrative staff).
- **Research Assistant (RA)** — field of research (subtype of Employee).
- **Lecturer** — RA teaching courses (subtype of RA).
- **Project** — name, start date, end date.
- **Course** — ID (unique number), name, weekly duration (hours).

## Relationships

- University and Faculty (1 -> 1..*), it consisting of several faculties.
- Faculty and Institute (1 -> 1..*), it made up of various institutes.
- Faculty and Dean (Employee) (1 -> 1), its dean an employee of the university.
- RA and Institute (0..* -> 1..*), it assigned to at least one institute.
- RA and Project (0..* -> 0..*, hours), it involved for a certain number of hours.
- Lecturer (RA) and Course (0..* -> 0..*), some RAs teaching them.
