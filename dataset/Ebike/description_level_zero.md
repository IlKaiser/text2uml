## Classes

- **EBike** — composed of a frame, a drive system, and a controller.
- **Frame** — made out of steel.
- **Wheel** — inserted into it.
- **DriveSystem** — composed of a motor.
- **Motor** — part of it.
- **Battery** — removable, its stored energy (measured in Watt-hours, Wh).
- **Controller** — its state (On, Off, Charging), it controls the battery and commands the drive system.
- **BasicController** — a variant of it.
- **AdvancedController** — a variant of it, it estimates the next Date (its maintenance inspection).

## Relationships

- **EBike** and **Frame** (composition, 1 -> 1), it composed of the frame.
- **EBike** and **DriveSystem** (composition, 1 -> 1), it composed of the drive system.
- **EBike** and **Controller** (composition, 1 -> 1), it composed of it (the controller).
- **Frame** and **Wheel**, two of them inserted into each frame (composition, 1 -> 2).
- **DriveSystem** and **Motor** (composition, 1 -> 1), it composed of it (the motor).
- **EBike** and **Battery** (removable, connectable to it, 1 -> 0..1).
- **Controller** and **Battery** (control, only one connected, 1 -> 0..1), it controls it.
- **Controller** and **DriveSystem** (command, 1 -> 1), it commands it.
- **Controller** and **BasicController** (generalization, 1 -> 1), it generalizes to the basic variant.
- **Controller** and **AdvancedController** (generalization, 1 -> 1), it generalizes to the advanced variant, estimating the next inspection Date.
