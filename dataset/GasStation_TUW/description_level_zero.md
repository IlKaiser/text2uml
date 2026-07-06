## Classes
- **GasStation** — ID (unique), address
- **FuelPump** — number (identifier), self-service (yes/no)
- **Fuel** — name (unique identifier), octane rating
- **DailyPrice** — date, price per liter (identified by date + Fuel + GasStation)
- **FuelPurchase** — ID (unique), liters dispensed

## Relationships
- GasStation (1) -> FuelPump (0..12, each identified by its number), it comprising up to twelve of them.
- FuelPump (0..*) -> Fuel (1..*, offered fuels).
- GasStation (1) -> DailyPrice (0..*, archived daily prices).
- Fuel (1) -> DailyPrice (0..*), it applying to one of them.
- DailyPrice (1) -> FuelPurchase (0..*, price at which it was purchased).
- FuelPump (1) -> FuelPurchase (0..*, pump at which it was purchased).
- Fuel (1) -> FuelPurchase (0..*, fuel purchased).
