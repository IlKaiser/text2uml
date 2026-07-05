## Classes

- **Product** — unique name, description, net price, manufacturing company.
- **Country** — VAT rate, currency, conversion rate to euro (used for gross price calculation).
- **Representative** — first name, last name, social security number, telephone number, email address.
- **Customer** — name, address, telephone number, email address.
- **Order** — unique identifier, order quantity, order date, desired delivery date.

## Relationships

- **Product** manufactured in one **Country** (manufacturer's country) (1 -> 1).
- **Product** sold in **Country** (not every product in every country, queryable) (0..* -> 0..*).
- **Representative** responsible for exactly one **Country** (1 -> 1).
- **Customer** resells goods in **Country** (the countries where it resells) (1 -> 0..*).
- **Representative** places **Order** for **Customer** (orders from customers placed by representatives) (1 -> 0..*).
- **Order** refers to exactly one **Product** (1 -> 1).
- **Representative** distributes **Product** (0..* -> 0..*).
