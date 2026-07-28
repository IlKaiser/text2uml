A smart home automation system (SHAS) lets various users manage smart home automation tasks automatically. A smart home has a physical address. It consists of several rooms. Each room may contain sensor devices and actuator (controller) devices of different types. Examples are temperature sensor, movement sensor, light controller, and lock controller. Each sensor and actuator has a unique device identifier. When a new sensor or actuator is activated or deactivated, SHAS recognizes the change. It then updates its infrastructure map.

When SHAS is operational, a sensor device provides sensor readings periodically. Each reading records the measured value and the timestamp. In a similar way, a predefined set of control commands can be sent to the actuator devices. Examples are lockDoor and turnOnHeating. Each command includes the timestamp and the status of the command, such as requested, completed, or failed. SHAS records all sensor readings and control commands for a smart home in an activity log.

The owner can set up and manage relevant alerts in a smart home by setting up automation rules. An automation rule has a precondition and an action. The precondition is a Boolean expression. It is built from relational terms connected by basic Boolean operators (AND, OR, NOT). Atomic relational terms may refer to rooms, sensors, actuators, sensor readings, and control commands. The action is a sequence of control commands. For example, a sample rule could specify:

when actualTemperature by Device #1244 in Living Room < 18 and window is closed
then turnOnHeating in Living Room

Owners can create, edit, activate, and deactivate automation rules. Only deactivated rules can be edited. Rules can also depend on or conflict with other rules. In this way, a complex rule hierarchy can be designed. SHAS records whenever an active rule was triggered, using a timestamp.
