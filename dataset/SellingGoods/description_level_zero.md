## Classes
- **Customer** — head-office address, delivery addresses, VAT number.
- **Address** — head-office type, delivery-location type.
- **Product** — current stock level (manually adjusted end of each working day), availability (not always available).
- **Order** — paid-in-full requirement, paid-date, shipped-date, fully-shipped mark, payment timing (directly at placement or after delivery).
- **OrderLine** — quantity ordered (of one product), shipped/to-be-shipped status.
- **Invoice** — sent for an order, paid in full.
- **Shipment** — group of order lines shipped together, FSM status (packed, shipped, received), missing/damaged tracking.

## Relationships
- **Customer** and **Address**, its head office (1 -> 1) plus its delivery locations / warehouses (1 -> 0..*).
- It and its **Orders** (1 -> 0..*).
- **Order** and its **OrderLines**, one or more of them (1 -> 1..*).
- They and **Product**, a particular one (0..* -> 1).
- It and its stock (its current level, 1 -> 1).
- **Order** and **Invoice**, payable directly at its placement or after its delivery, in full (1 -> 1).
- **Shipment** and **OrderLine**, it grouping them shipped together (1 -> 1..*), covering partial deliveries of one or more of them.
- **Order** and **Shipment**, its shipments (1 -> 0..*).
