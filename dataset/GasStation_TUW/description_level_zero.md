Classes:

GasStation — has a Name.
Pump — a fuel dispenser, with Type, InUse, RefillThreshold, and Blocked flags.
CardHolder — a customer holding a fuel card, with Name and a Suspended flag.
RefuelTurn — a single card-based refueling session, identified by RefuelTurnNumber.
CashTurn — a cash-based refueling session (no attributes, no customer, no invoice).
Invoice — a bill with Number, Discount, and status.
InvoiceLine — a single line item on an invoice, with Number.

Relationships:

A GasStation contains many Pumps (1 → 0..*), each Pump belonging to one station.
Each Pump records many RefuelTurns and many CashTurns (the two ways it gets used: by card or by cash).
A CardHolder performs many RefuelTurns and receives many Invoices.
An Invoice is composed of many InvoiceLines.
Each RefuelTurn maps to at most one InvoiceLine (0..1) — a refueling event optionally gets billed as a single line.