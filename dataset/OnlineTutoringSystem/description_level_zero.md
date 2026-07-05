## Classes
- **OTS** — online tutoring system, used by students and tutors.
- **Tutor** — name, email address, bank account, may also be a student.
- **Student** — name, email address, browses offers, makes requests, pays.
- **Subject** — e.g. mathematics, science, literature.
- **TutoringOffer** — level of expertise (e.g. primary school, high school, university level), hourly price (may be subject specific).
- **Availability** — weekly slot (e.g. Thursdays 10:00–11:30).
- **TutoringRequest** — level of tutoring, suggested target date and time of first session.
- **TutoringSession** — target time, agreed/turn-up slot, may be cancelled by student or tutor; student cancels <24h → 75% of price paid; tutor cancels <24h → 25% discount on next session to same student; may spawn a follow-up.
- **Payment** — method (credit card or wire transfer).
- **Award** — tutor-of-the-month, monthly, best tutors per subject.

## Relationships
- OTS and its users, students and tutors (many, 1 -> 0..*), a tutor possibly also being a student.
- Tutor and TutoringOffer (1 -> 0..*), offered online after registration.
- TutoringOffer and Subject (0..*  -> 1), it carrying a subject-specific level of expertise and hourly price.
- Tutor and Availability (1 -> 0..*), specified weekly by it.
- Student and TutoringOffer (browsing, 1 -> 0..*), filtered by a specific Subject.
- Student and TutoringRequest (1 -> 0..*), it directed at the designated Tutor.
- TutoringRequest and Tutor (0..* -> 1), it confirmed by it or countered with another slot.
- TutoringRequest and TutoringSession (1 -> 1), agreed and attended by both parties.
- TutoringSession and TutoringSession (follow-up, 1 -> 0..*), agreed during a session by student and tutor.
- Student and TutoringSession (cancellation, 1 -> 0..*), with 75% of price due on cancellation under 24 hours.
- Tutor and TutoringSession (cancellation, 1 -> 0..*), with a 25% next-session discount to the same student on cancellation within 24 hours.
- Student and Payment (1 -> 0..*), it settling a session (credit card or wire transfer) after tutoring.
- TutoringSession and Payment (1 -> 1), covering the session's price.
- Tutor and Award (1 -> 0..*), granted per Subject at each month's end.
