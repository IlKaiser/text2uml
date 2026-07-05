## Classes

- **HBMS** — central booking system, lists available offers, forwards preliminary booking parameters, sends five best special offers, stores past bookings, calculates reliability rating, cancels/reimburses unconfirmed bookings.
- **Traveller** — name, billing information (incl. company name and address), optional travel preferences (breakfast included, free wifi, 24/7 front desk, etc.), credit card information, reliability rating (calculated from its past bookings).
- **Search** — city, arrival date, departure date, number of rooms, room type (single, double, twin), minimum hotel rating (stars), tentative budget (max. cost per night), optional further travel preferences.
- **Offer** — available accommodation deal for a travel period (city, dates, room type, rating, price).
- **Booking** — travel period, rooms, cancellation deadline (optional); regular or preliminary; finalized; pre-paid (paid immediately, non-reimbursable) or paid-at-hotel (paid during stay).
- **PreliminaryBooking** — key parameters (price, city area, hotel rating, key preferences, unique booking identifier), forwarded to competitor hotels for a 24-hour competition window.
- **SpecialOffer** — competing offer, provided by other hotels within 24 hours.
- **Hotel** — located in a city at an address, participating provider of deals, announces available room types for a period, informs it of fully-booked room types, confirms bookings within 24 hours.
- **HotelChain** — group possibly running it.
- **City** — location of hotels and searches.

## Relationships

- Traveller and HBMS (1 -> 1, it registered via name, billing information, optional preferences).
- Traveller and Search (1 -> 0..*, it specifying city, dates, room count, room type, minimum rating, budget, optional preferences).
- Search and Offer (1 -> 0..*, it listing all its available hotel offers for the given travel period).
- Traveller and Booking (1 -> 0..*, it creating either a preliminary one or a completed regular one).
- Booking and PreliminaryBooking (1 -> 0..1, it specialising the former).
- HBMS and Hotel (1 -> 0..*, it forwarding its preliminary booking parameters, the traveller's preferences and its reliability rating to competitor ones).
- Hotel and SpecialOffer (1 -> 0..*, competitor ones providing them within the next 24 hours).
- HBMS and SpecialOffer (1 -> 5, it sending the five best to the traveller after its 24-hour deadline).
- Traveller and SpecialOffer (1 -> 0..1, it switching to the new one or proceeding with its original preliminary booking).
- Booking and CreditCard (1 -> 1, it provided to finalize the former).
- Hotel and Booking (1 -> 0..*, it confirming each finalized one within 24 hours, needing to send a confirmation to the traveller).
- HBMS and Booking (1 -> 0..*, it cancelling an unconfirmed completed one after 24 hours, reimbursing the traveller for a pre-paid one).
- Booking and Traveller (1 -> 1, its cancellation before the deadline carrying no consequences, its cancellation after the deadline charging 1-night accommodation to them).
- Hotel and Traveller (1 -> 0..*, it offering them financial compensation on its cancellation of a confirmed booking).
- HBMS and Booking (1 -> 0..*, it storing all its past information per traveller).
- Hotel and City (1 -> 1, it located in it at a particular address).
- Hotel and HotelChain (0..* -> 0..1, it possibly run by the latter).
- Hotel and RoomType (1 -> 0..*, it announcing its available types for a period, informing HBMS of its fully-booked ones).
