You are a project manager at a medium-sized software company. You are responsible for building the system described below. In the first 5 tasks of this exam you will write a specification step by step.

Short description:
This system is an infrastructure and software platform. This system captures energy consumption data in private homes.

The customer:
HomeTec is a small company. HomeTec works in building engineering. HomeTec makes sensors. These sensors can be installed in private homes. HomeTec is located in Aachen. Your company is also located in Aachen.

Detailed system description:
The system is called "E-Home-2020". The system has a set of multiple sensors. Each sensor reports the sensor values to a connected SensorController. The SensorController also stores information about the apartment. The SensorController also stores information about the inhabitants. The inhabitants can log on to the SensorController. The inhabitants log on with a username and a password. The inhabitants can query the values of each installed sensor. The inhabitants can set a threshold for each sensor.

An external system is called "E-Monitor". "E-Monitor" is attached to the "E-Home-2020" system. "E-Monitor" checks all thresholds for each sensor automatically. Sometimes a sensor's threshold is exceeded. An email address may be set for the apartment. If the threshold is exceeded and an email address is set, then "E-Monitor" sends alert messages. "E-Monitor" sends the alert messages through the "E-Home-2020" system.

Some inhabitants have administrative rights. Administrative rights let these inhabitants configure sensors. For example, these inhabitants can assign a sensor to a new room. These inhabitants can also change a sensor's description.

Right now, HomeTec makes two types of sensors. The two types are temperature sensors and power sensors. A power sensor is installed in a plug-and-play way. A power sensor is placed between the power socket and the appliance. A temperature sensor has a valid value range. The valid value range has a min and a max. A power sensor stores the total power consumption as a counter reading. Every sensor has a unique vendor-id. Every sensor has a description. An example description is "temperature radiator living room". Another example description is "power consumption microwave oven". Every sensor has a unit. The possible units are CELSIUS and WATT. Each sensor is assigned to exactly one room.

An apartment has an address. An apartment has at least two rooms. Each room has a description. An example description is "living room". For each sensor you can query the measured values. The measured values are stored in the system. Each value has a timestamp. Each value has a value as a floating point number. For quick access, you can read the last value of a sensor directly.

Goal of project:
The goal is to develop a prototype. The prototype works on a specific target platform. The prototype works in a specific environment. The prototype provides the essential functionality.
