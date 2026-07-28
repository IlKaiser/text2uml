## Classes

- **HospitalHouseMD** — hospital, it accepts patients (rare, often unexplainable symptoms), it maintains a treatment database per illness.
- **Patient** — sick person, it signs/retracts a consent form (to move its treatments to a higher stage).
- **Doctor** — it makes a diagnosis, it is responsible for downgrading treatments; one of them acting as head doctor.
- **HeadDoctor** — it approves a diagnosis (made by another doctor) before any treatment.
- **Nurse** — it performs higher-rate follow-up checks.
- **Diagnosis** — attempt to explain a patient's illness, it requires head-doctor approval.
- **Illness** — condition, its priority score (high, low), it is changeable at any time.
- **Treatment** — applied to a diagnosis, executed accordingly, its stage (1 weakest / 2 / 3); at stage 1 it needs no consent and starts immediately, it moves to 2 or directly to 3, at stage 3 it needs higher follow-up, it can downgrade to lower stages.
- **ConsentForm** — signed by the patient, it enables free upgrade of all its treatments (stage 1 -> 2 -> 3) until retraction.

## Relationships

- Patient and Doctor, assigned (1 -> 1..*), the latter trying to make a diagnosis for it.
- Doctor and Diagnosis, authored (1 -> 0..*), it making them.
- HeadDoctor and Diagnosis, approval (1 -> 0..*), it required before any treatment.
- Illness and Treatment, its database of all possible ones (1 -> 0..*), they applicable to any diagnosis.
- Diagnosis and Treatment, applied and executed (1 -> 0..*), it accordingly executing them.
- Patient and Treatment, its applied ones (1 -> 0..*), the stage-3 ones needing a higher follow-up rate for it.
- Patient and ConsentForm, signed then retractable (1 -> 0..1), it enabling free upgrade of all its treatments.
- Nurse and Patient, higher-rate follow-up checks (0..* -> 0..*), it checked more often for its stage-3 treatments.
- Doctor and Treatment, downgrading to lower stages (1 -> 0..*), it responsible for them.
- Illness and Treatment, priority stage upgrade (1 -> 0..*), on its priority score high all its applied ones immediately placed into stage 3 (no patient consent needed).
