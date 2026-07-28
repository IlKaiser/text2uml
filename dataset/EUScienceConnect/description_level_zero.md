## Classes

- **Person** — person ID (unique), phone number, keywords (key research topics)
- **ResearchInstitution** — institution code (unique)
- **ScientificArticle** — DOI (unique), title; abstract, either peer-reviewed paper or technical report
- **PeerReviewedPaper** — citation count
- **TechnicalReport**
- **Publisher** — name (identifier)
- **Journal** — name (given by its publisher), impact factor (scientific impact)

## Relationships

- Person and ResearchInstitution, employment (1 -> 0..*), each person working for one of them.
- Person and ScientificArticle, authorship (1..* authors -> 0..*), storing each author's position (for multiple authors).
- Person (reviewer) and PeerReviewedPaper, review (0..* -> 0..*), it recording who reviewed it.
- ScientificArticle and its subtypes PeerReviewedPaper, TechnicalReport (disjoint, complete).
- TechnicalReport and ResearchInstitution, publication (0..* -> 1 single institution), it publishing multiple of them.
- Publisher and Journal, publication (1 -> 0..*), journals possibly sharing a name across different publishers.
- Journal and ResearchInstitution, subscription (0..* -> 0..*).
- PeerReviewedPaper and Journal, publication (0..* -> 0..1), only they (not technical reports) appearing in it.
