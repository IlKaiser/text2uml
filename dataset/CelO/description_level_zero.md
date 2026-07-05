## Classes

- **Organizer** — first name, last name, email address (also its username), postal address, phone number, password; manager of an event, sometimes also an attendee of it.
- **Attendee** — first name, last name, email address (its username, from the invitation), password (set on account creation), attendance status (will attend / maybe / cannot attend).
- **Event** — kind (from a list, e.g. birthday party, graduation party, or new), occasion, start date/time, end date/time, invitation status.
- **EventKind** — reusable event type (selectable from a list or new).
- **Location** — name, address (from a list or new).
- **Checklist** — event-specific list of its tasks.
- **Task** — status (needs to be done / has been done / not applicable), designation (for the organizer or for an attendee), reusable for the next event of its kind.

## Relationships

- Event and Organizer (1 -> 1..*), one of them for a small event, several of them for larger ones.
- Organizer and Attendee (1 -> 0..*), via its email invitations (their first/last names, their email addresses).
- Event and EventKind (0..* -> 1), it selected or created by the organizer.
- Event and Location (0..* -> 1), it from a list or new (its name, its address).
- Event and Attendee (0..* -> 0..*), each of them (will attend / maybe / cannot attend) part of its invitation status (replied / not yet replied, coming for sure / maybe).
- Event and Checklist (1 -> 1), it presented to the organizer on its selection.
- Checklist and Task (1 -> 0..*), new ones added by the organizer for its next event of the same kind.
- Task and Attendee (0..* -> 0..*), those of them designated for attendees shown only to confirmed ones, each of them selecting their own so the organizer sees who is bringing what.
