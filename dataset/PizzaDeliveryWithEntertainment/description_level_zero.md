## Classes

- **User** — birthday, name, address, unique identifier (uniquely identifiable)
- **BusinessOwner** — LinkedIn account (added to its profession network); a User
- **HungryCustomer** — delivery address; a User
- **Entertainer** — stage name, short bio, price per 30 minutes; a User
- **PizzaRestaurant** — zip code, address, phone number, website, opening hours
- **Pizza** — name (margarita, quattro stagioni, etc.), crust structure (classic Italian, deep dish, cheese crust), price (unique per restaurant, distinguishable even with same name/price)
- **Order** — ID, date and time placed (logged), latest time of delivery, number of people
- **EntertainmentOrder** — type of entertainment, duration; a special Order (not every one is such)

## Relationships

- **BusinessOwner** and **PizzaRestaurant**, ownership (1 -> 0..*), it owning a number of them.
- **PizzaRestaurant** and **Pizza**, offering (1 -> 0..*), it offering a number of them.
- **HungryCustomer** and **Order**, placement (1 -> 0..*), it making them.
- **Order** and **Pizza**, containing one or more of them (1 -> 1..*).
- **EntertainmentOrder** and **Entertainer**, fulfilled by exactly one of them (0..* -> 1).
- **Entertainer** and **PizzaRestaurant**, work relationship (0..* -> 0..*), it indicating its availability by day (Monday, Tuesday, Wednesday, etc.) per chosen one.
