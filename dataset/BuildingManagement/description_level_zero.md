## Classes
- WebPortal — version number, URL address (e.g. "http://www.ebuntu-nrw.org"), catalog of buildings with energy information.
- Building — unique ID, name, energy class (A, B, C, D), address, year of construction.
- Entry — name (additional entry), optionally released publicly.
- TextEntry — textual information (specialization of Entry).
- NumberEntry — floating point number, unit (freely definable text) (specialization of Entry).
- Image — file name, optional profile-photo flag.
- User — unique user name, email address, password.
- Administrator — ID card number (plus its User information), a special User.
- Comment — its text on buildings.

## Relationships
- WebPortal offers Building (1 -> 0..*), a catalog with energy information.
- Building has one owner (User), assigned as owner (0..* -> 1).
- User (owner) adds Entry to its buildings (1 -> 0..*).
- Entry, specialized by TextEntry and NumberEntry (generalization), 1 -> 2 subtypes.
- Building holds Image (1 -> 0..*), exactly one of them (1) its profile photo.
- WebPortal registers User (1 -> 0..*, via unique user name, email confirmation link).
- Administrator, a special User (generalization), 1 -> 1.
- User comments on other users' Building via Comment (1 -> 0..*).
- Administrator sets up User accounts (its setup function) and deletes Comment (1 -> 0..*).
- Visitor searches Building (full-text search) and views publicly released Entry (1 -> 0..*).
