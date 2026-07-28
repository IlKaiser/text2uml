## Classes

- **Employee** — name, social security number
- **CreativeEmployee** — (subtype of Employee)
- **Director** — assistant's name, Oscar flag (has Oscar), a creative employee
- **Actor** — completed education, Oscar flag (has Oscar), a creative employee
- **Technician** — field of activity, an employee
- **FilmSet** — location
- **Film** — title, year of release, genre (thriller, action, horror, or comedy)
- **Screenwriter** — name, most successful film
- **Screenplay** — title, version, date, number of scenes, plot
- **Novel** — title, author
- **Concept** — (basis alternative)

## Relationships

- Employee and FilmSet (0..* -> 1), several of them on it.
- Director/Actor and Oscar, both potential winners (flag per person).
- FilmSet and Film (0..* -> 0..*), it for several of them, each at several of it.
- Film and Screenplay (1 -> 1), it on a specific one, that one for exactly one of it.
- Screenwriter and Screenplay (1 -> 0..*), it to several, each from exactly one of it.
- Screenplay and Novel/Concept, its basis (either one), Novel with title and author.
- Actor and Screenplay (0..* -> 1), its readers preparing for their respective roles.
- Director and Screenplay (1 -> 1), its implementer (per the specifications).
