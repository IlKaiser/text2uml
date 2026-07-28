## Classes
- **Student** — follows/registers for courses, requests appointments.
- **TeachingAssistant** — provides help (only for courses it teaches).
- **Course** — subject of questions and appointments.
- **Appointment** — booking between a student(s) and a teaching assistant (always for one specific course).

## Relationships
- **Student** registers for **Course** (many-to-many), a prerequisite for requesting an appointment for it.
- **TeachingAssistant** teaches **Course** (many-to-many), it providing help only for these.
- **Appointment** links to **Course** (many -> 1, one specific course).
- **Appointment** links to **TeachingAssistant** (many -> 1, one particular assistant, TA for that course).
- **Appointment** links to **Student** (student(s) per appointment), each student requesting it only for a course it is registered for.
