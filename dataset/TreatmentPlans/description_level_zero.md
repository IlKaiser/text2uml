## Classes

- **Hospital** — applies for permissions, visited by patients, executes examinations
- **Doctor** — affiliated to a hospital, prescribes examinations, poses diagnosis, defines actual treatment, sends patients
- **Patient** — comes with a particular problem
- **Examination** — of only one type (advised or freely chosen)
- **ExaminationType** — "cardiologic test", "echography", "radiology", "scanner", NMR, …
- **Permission** — a hospital's allowance for an examination type
- **Problem** — a type of problem, identified via diagnosis
- **TreatmentPlan** — standard plan per problem type, lists advised examination types
- **ActualTreatment** — based on a treatment plan, consists of advised examinations and freely chosen medical actions
- **MedicalAction** — freely chosen action within an actual treatment

## Relationships

- Doctor prescribes Examination (1 -> 0..*) for Patient (1 -> 0..*).
- Examination and ExaminationType, every single one of it being of only one type (0..* -> 1).
- Hospital applies for Permission (1 -> 0..*), each pairing it with one ExaminationType (0..* -> 1), not all types allowed for all hospitals (some, e.g. NMR, ungranted).
- Hospital executes Examination (1 -> 0..*), only according to its Permission.
- TreatmentPlan and Problem (per type of problem, 1 -> 1), it listing advised ExaminationType (1 -> 0..*).
- Patient visits Hospital (0..* -> 1), it being the hospital of the affiliated Doctor.
- Doctor (affiliated to the visited Hospital, 0..* -> 1) poses diagnosis for Patient's Problem (1 -> 0..*).
- Doctor defines ActualTreatment (1 -> 0..*), based on an existing TreatmentPlan (0..* -> 1) for the identified Problem.
- ActualTreatment and Patient (0..* -> 1), it deviating from the standard TreatmentPlan.
- ActualTreatment consists of advised Examination (1 -> 0..*) plus freely chosen MedicalAction (1 -> 0..*).
- Doctor sends Patient to another Hospital (1 -> 0..*) for the desired Examination, its own hospital lacking the Permission.
