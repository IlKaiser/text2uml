## Classes

- **University** — top-level entity, host of research groups.
- **Department** — organizational unit grouping research groups.
- **ResearchGroup (RG)** — belongs to a department, acts as WP leader, source/target of "service delivery".
- **Researcher** — member of exactly one RG, assignable to WPs (any WP, independent of the leading RG).
- **Project** — research effort, decomposed into Work Packages.
- **WorkPackage (WP)** — unit of a project, has one RG as leader, has assigned team-member researchers.
- **Assignment** — record of a researcher as team member of a WP, traceable for "service delivery", terminated and re-created on RG change or WP-leadership re-assignment.

## Relationships

- University and ResearchGroup (RG), many (1 -> 0..*), with them belonging to different Departments.
- Department and ResearchGroup (RG), grouping them (1 -> 0..*).
- Researcher and ResearchGroup (RG), it a member of exactly one (0..* -> 1).
- Project and WorkPackage (WP), decomposed into them (1 -> 0..*).
- WorkPackage (WP) and ResearchGroup (RG), it having exactly one as WP leader (0..* -> 1).
- Researcher and WorkPackage (WP), it assignable to any of them (independent of the leading RG) via Assignment (0..* -> 0..*).
- Assignment and Researcher, linking one (0..* -> 1), terminated and re-created on its RG change.
- Assignment and WorkPackage (WP), linking one (0..* -> 1), re-recorded on its leadership re-assignment (for correct "service delivery" recording).
