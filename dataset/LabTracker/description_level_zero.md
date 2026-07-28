## Classes
- **Doctor** — practitioner number (numeric), signature (digital, an image of its actual signature), full name, address, phone number
- **Patient** — health number (alpha-numeric), first name, last name, date of birth, address, phone number
- **Requisition** — valid-from date, repetition (number of times; interval weekly/monthly/every-half-year/yearly, same pattern for all its tests)
- **Test** (examination; interchangeable terms) — group, duration (defined by the lab network, identical at each lab, unchanged by quantity for some kinds, e.g. several of them on one blood sample), access type (appointment-required e.g. x-ray, walk-in only e.g. blood test, or sample-drop-off e.g. urine/stool)
- **Result** — value (negative or positive), accompanying report
- **Appointment** — confirmation number, date, start/end times, change/cancellation fee (incurred within 24 hours)
- **Lab** — address, business hours (day start time to end time, no breaks, unchanged week to week, open every day), name, registration number, own fee
- **LabNetwork** — defines each test's duration

## Relationships
- A Doctor (1 -> 0..*) creates Requisitions (via its number, signature, name, address, phone, plus the valid-from date), cannot prescribe a Test for itself (may prescribe to another of them).
- A Requisition (1 -> 1) shows one Patient (via its health number, name, date of birth, address, phone).
- A Requisition (1 -> 1..*) combines Tests (single group only — only blood, or only ultrasound; no mixing of blood and ultrasound).
- The LabNetwork (1 -> 0..*) defines duration for each Test, enabling appointment scheduling.
- A Doctor (1) and a Patient (1) each view Results (0..*) of each Test (its value negative/positive, plus its report).
- A Patient (1 -> 0..*) makes Appointments for a Requisition (selecting the desired Lab by its address and business hours; one appointment at a time for repeated-test requisitions).
- An Appointment (1 -> 1) confirms via one Lab (its confirmation showing confirmation number, date, start/end times, plus its name and registration number; changeable or cancellable at any time, fee within 24 hours).
- A Lab (0..* -> 1..*) offers all Tests and determines its own fee and business hours.
