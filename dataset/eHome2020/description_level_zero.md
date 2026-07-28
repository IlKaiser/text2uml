## Classes

- **SensorController** — stores apartment information, authenticates inhabitants (username, password), holds sensor values.
- **Sensor** — vendor-id (unique), description, unit (CELSIUS or WATT), threshold.
- **TemperatureSensor** — a Sensor, valid value range (min, max).
- **PowerSensor** — a Sensor, total power consumption (counter reading), plug-and-play between power-socket and appliance.
- **Apartment** — address, email address (optional).
- **Inhabitant** — username, password, administrative rights (some, enabling sensor configuration).
- **Room** — description.
- **Value** — timestamp, value (floating point), last-value quick-access.
- **E-Monitor** — external system, checks all thresholds automatically, sends alert messages via E-Home-2020.

## Relationships

- **SensorController** and **Sensor** (1 -> 1..*, multiple, connected), it reporting their values to it.
- **SensorController** and **Apartment** (1 -> 1), it storing its information.
- **Apartment** and **Inhabitant** (1 -> 1..*), they log on to the SensorController, query values, specify thresholds.
- **Apartment** and **Room** (1 -> 2..*, at least two).
- **Sensor** and **Room** (1..* -> 1, exactly one), it (re)assignable by administrative Inhabitants.
- **Sensor** and **Value** (1 -> 0..*, measured, queryable, stored, with direct last-value access).
- **E-Monitor** and **Sensor** (1 -> 1..*, all), it checking their thresholds.
