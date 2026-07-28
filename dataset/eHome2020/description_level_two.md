You are a project manager at a medium-sized software company. You are responsible for successfully implementing the system described below. In the first 5 tasks of this exam, you will produce a specification step by step.

**Short description:**
This system is an infrastructure and software platform for capturing energy consumption data in private homes.

**The customer:**
HomeTec is a small company in the field of building engineering. HomeTec manufactures sensors that can be installed in private homes. Like your own company, HomeTec is located in Aachen.

**Detailed system description:**
The system “E-Home-2020” consists of multiple sensors. These sensors report their values to a connected SensorController. The SensorController also stores information about the apartment and the apartment’s inhabitants. The inhabitants can log on to the SensorController using a username and password. After logging on, the inhabitants can query the values of each installed sensor. The inhabitants can also set a threshold for each sensor.

An external system called “E-Monitor” is attached to the “E-Home-2020” system. “E-Monitor” automatically checks all thresholds for each sensor. If a sensor’s threshold is exceeded, and if an email address has been specified for the apartment, then “E-Monitor” sends alert messages through the “E-Home-2020” system.

Some inhabitants have administrative rights. These administrative rights let those inhabitants configure sensors. For example, an inhabitant with administrative rights can assign a sensor to a new room or change a sensor’s description.

At present, HomeTec manufactures two types of sensors: temperature sensors and power sensors. Power sensors are installed in a plug-and-play fashion between the power socket and the appliance. Temperature sensors have a valid value range, defined by a minimum and a maximum. Power sensors store the total power consumption as a counter reading. All sensors have a unique vendor-id, a description (for example, “temperature radiator living room” or “power consumption microwave oven”), and a unit. The possible units are CELSIUS and WATT. Each sensor is assigned to exactly one room.

An apartment has an address and consists of at least two rooms. Each room has a description (for example, “living room”). For each sensor, one can query the measured values, and these measured values are stored in the system. Each value has a timestamp and the value itself as a floating point number. For quick access, one can read the last value of a sensor directly.

**Goal of project:**
The goal of the project is to develop a prototype. The prototype works on a specific target platform in a specific environment. The prototype provides the essential functionality.
