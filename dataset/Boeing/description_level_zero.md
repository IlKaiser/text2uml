## Classes

- **Boeing** — aircraft seller (builds only "on demand", after a sales agreement, demo versions out of scope).
- **AirlineCompany** — customer, buyer of aircrafts.
- **Aircraft** — airplane (very expensive, built on demand).
- **Contract** — sales regulation (global, stipulating common elements: delivery conditions, legal aspects).
- **Acquisition** — single airplane purchase (chosen model, negotiated price, chosen options & customizations, delivery date).
- **Salesperson** — Boeing employee managing contracts.

## Relationships

- **Boeing** and **AirlineCompany**, sales of aircrafts regulated via contracts (1 -> 0..*).
- **Contract** and **AirlineCompany**, it binds one airline (0..* -> 1).
- **Contract** and **Acquisition**, it may consist of several of them (1 -> 1..*).
- **Acquisition** and **Aircraft**, each yields one built airplane (1 -> 1).
- **Salesperson** and **Contract**, each contract managed by one of them (assigned may change over time, always one available for the client), they act for several (1 -> 1..*).
- **AirlineCompany** and **AirlineCompany**, mother-daughter relationships (main airline having a low-cost daughter, tracked to follow aircrafts shifted to partner airlines) (0..1 -> 0..*).
