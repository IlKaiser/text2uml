You are a project manager. You work at a medium-sized software company. You are responsible for one system. This system is described below. The implementation must succeed. This exam has 5 first tasks. You will produce a specification. You will produce it step-by-step.

Short description:
This is an infrastructure. This is a software platform. It captures energy consumption data. This data comes from private homes.

The customer:
HomeTec is a small company. HomeTec works in building engineering. HomeTec manufactures sensors. These sensors can be installed in private homes. HomeTec sits in Aachen. Your company also sits in Aachen.

Detailed system description:
The system is called "E-Home-2020". The system contains multiple sensors. Each sensor reports its values. Each sensor reports them to a SensorController. Each sensor connects to the SensorController.

The SensorController stores information. This information describes the apartment. This information includes the inhabitants.

The inhabitants live in the apartment. An inhabitant can log on to the SensorController. An inhabitant uses a username. An inhabitant uses a password. An inhabitant can query values. An inhabitant queries values for each installed sensor. An inhabitant can specify a threshold. An inhabitant specifies a threshold for each sensor.

An external system is called "E-Monitor". E-Monitor is attached to the system. E-Monitor checks all thresholds. E-Monitor checks thresholds for each sensor. E-Monitor checks them automatically.

A sensor's threshold may be exceeded. An email address may be specified. This address belongs to the apartment. Then E-Monitor sends alert messages. E-Monitor sends the messages. E-Monitor uses the "E-Home-2020" system.

Some inhabitants have administrative rights. These rights let them configure sensors. Configuration includes assigning sensors to a new room. Configuration includes changing the sensor's description.

HomeTec currently manufactures two types of sensors. The first type is temperature sensors. The second type is power sensors. A power sensor is installed in a plug-and-play fashion. It is installed between the power-socket and the appliance.

A temperature sensor has a valid value range. The range has a min. The range has a max.

A power sensor stores the total power consumption. It stores this as a counter reading.

Every sensor has a unique vendor-id. Every sensor has a description. An example description is "temperature radiator living room". Another example is "power consumption microwave oven". Every sensor has a unit. One possible unit is CELSIUS. Another possible unit is WATT.

Each sensor is assigned to exactly one room. An apartment has an address. An apartment contains at least two rooms. A room has a description. An example is "living room".

One can query the measured values. One queries them for each sensor. The system stores these values. Each value has a timestamp. Each value has a corresponding value. This value is a floating point number. A sensor has a last value. One can access this last value directly. This gives quick access.

Goal of project:
The goal is to develop a prototype. The prototype works on a specific target platform. It works in a specific environment. It provides essential functionality.
