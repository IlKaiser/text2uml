A smart home automation system exists. It is called SHAS. SHAS serves various users. Users manage smart home automation tasks. This management is automatic.

A smart home has a physical address. A smart home consists of several rooms. A room may contain sensor devices. A room may contain actuator devices. Actuators are also called controllers. Sensors have different types. Actuators have different types. One type is a temperature sensor. One type is a movement sensor. One type is a light controller. One type is a lock controller. Each sensor has a device identifier. This identifier is unique. Each actuator has a device identifier. This identifier is unique.

A user activates a new sensor. A user activates a new actuator. A user deactivates a sensor. A user deactivates an actuator. SHAS recognizes this change. SHAS updates its infrastructure map.

SHAS can be operational. A sensor device provides sensor readings periodically. A sensor reading records the measured value. A sensor reading records the timestamp.

Control commands are predefined. One command is lockDoor. One command is turnOnHeating. SHAS sends control commands to the actuator devices. Each control command has a timestamp. Each control command has a status. One status is requested. One status is completed. One status is failed.

SHAS records all sensor readings. SHAS records all control commands. These readings belong to a smart home. These commands belong to a smart home. SHAS records them in an activity log.

The owner sets up relevant alerts. These alerts belong to a smart home. The owner manages these alerts. The owner sets up automation rules.

An automation rule has a precondition. An automation rule has an action. The precondition is a Boolean expression. Relational terms build the Boolean expression. Basic Boolean operators connect the relational terms. One operator is AND. One operator is OR. One operator is NOT. Atomic relational terms may refer to rooms. They may refer to sensors. They may refer to actuators. They may refer to sensor readings. They may refer to control commands. The action is a sequence. The sequence contains control commands.

Here is a sample rule. The precondition combines two terms. Term one is: actualTemperature by Device #1244 in Living Room < 18. Term two is: window is closed. The two terms use AND. The action is: turnOnHeating in Living Room.

Owners create automation rules. Owners edit automation rules. Owners activate automation rules. Owners deactivate automation rules. Owners edit only deactivated rules. A rule can depend on other rules. A rule can conflict with other rules. This allows a complex rule hierarchy. An owner can design this hierarchy.

An active rule can be triggered. SHAS records this event. SHAS uses a timestamp.
