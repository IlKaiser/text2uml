## Classes

- **SHAS** — smart home automation system, recognizes activation/deactivation changes, updates its infrastructure map, records triggers via timestamp
- **User** — manages smart home automation tasks
- **SmartHome** — located at a physical address
- **Room** — may contain sensors and actuators
- **Sensor** — device with unique device identifier, activated/deactivated
- **Actuator** — controller device with unique device identifier, activated/deactivated, of types (temperature sensor, movement sensor, light controller, lock controller)
- **SensorReading** — measured value, timestamp
- **ControlCommand** — predefined (e.g. lockDoor, turnOnHeating), timestamp, status (requested, completed, failed, etc.)
- **ActivityLog** — records all sensor readings and control commands
- **AutomationRule** — precondition (Boolean expression from relational terms via AND, OR, NOT), action (sequence of control commands), created/edited/activated/deactivated (only deactivated ones editable)
- **Owner** — sets up and manages relevant alerts, creates/edits/activates/deactivates rules
- **Precondition** — Boolean expression, relational terms referring to rooms, sensors, actuators, sensor readings, control commands
- **RelationalTerm** — atomic, refers to rooms, sensors, actuators, sensor readings, control commands

## Relationships

- SHAS and SmartHome (it, 1 -> 0..*, for various users), automatically managing its automation tasks.
- SmartHome and Room (1 -> 1..*).
- Room and Sensor (it, 1 -> 0..*).
- Room and Actuator (1 -> 0..*, of different types).
- Sensor and SensorReading (it, 1 -> 0..*, provided periodically during operation).
- Actuator and ControlCommand (it, 1 -> 0..*, sent with timestamp and status).
- SHAS and ActivityLog (1 -> 1, records of all its sensor readings and control commands).
- ActivityLog and SensorReading (it, 1 -> 0..*).
- ActivityLog and ControlCommand (1 -> 0..*).
- Owner and SmartHome (it, 1 -> 0..*, ownership).
- Owner and AutomationRule (it, 1 -> 0..*, created/edited/activated/deactivated).
- AutomationRule and Precondition (it, 1 -> 1).
- AutomationRule and ControlCommand (its action, a sequence, 1 -> 0..*).
- Precondition and RelationalTerm (it, 1 -> 1..*, connected by AND, OR, NOT).
- RelationalTerm and Room/Sensor/Actuator/SensorReading/ControlCommand (they, 0..* -> 0..*, atomic references).
- AutomationRule and AutomationRule (it, 0..* -> 0..*, dependency or conflict in a complex hierarchy).
- SHAS and AutomationRule (it, 1 -> 0..*, timestamped trigger record of an active one).
