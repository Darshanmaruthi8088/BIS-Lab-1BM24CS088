import numpy as np

hours = np.arange(24)

price = np.array([
    0.08, 0.07, 0.07, 0.06, 0.06, 0.08,
    0.10, 0.13, 0.15, 0.12, 0.10, 0.09,
    0.08, 0.09, 0.11, 0.14, 0.18, 0.22,
    0.25, 0.24, 0.20, 0.16, 0.12, 0.10
])

load = np.array([
    2.0, 1.8, 1.7, 1.6, 1.5, 1.8,
    2.5, 3.0, 3.5, 3.2, 2.8, 2.5,
    2.4, 2.6, 2.8, 3.2, 4.0, 4.5,
    5.0, 4.8, 4.2, 3.8, 3.0, 2.5
])

battery_capacity = 10.0
initial_soc = 5.0
minimum_soc = 2.0
maximum_soc = 10.0

max_charge_power = 3.0
max_discharge_power = 3.0

charge_efficiency = 0.95
discharge_efficiency = 0.95

dt = 1.0


def simulate_battery(schedule):
    soc = initial_soc
    soc_values = []
    grid_import = []
    penalty = 0.0

    for t in range(24):

        power = schedule[t]

        if power >= 0:
            energy_stored = power * charge_efficiency * dt
            new_soc = soc + energy_stored
        else:
            discharge_power = abs(power)
            energy_removed = (
                discharge_power / discharge_efficiency
            ) * dt
            new_soc = soc - energy_removed

        if new_soc > maximum_soc:
            penalty += (new_soc - maximum_soc) ** 2
            new_soc = maximum_soc

        if new_soc < minimum_soc:
            penalty += (minimum_soc - new_soc) ** 2
            new_soc = minimum_soc

        soc = new_soc
        soc_values.append(soc)

        grid_power = load[t] + power

        if grid_power < 0:
            penalty += abs(grid_power) ** 2
            grid_power = 0

        grid_import.append(grid_power)

    final_soc_penalty = (soc - initial_soc) ** 2
    penalty += 1000 * final_soc_penalty

    total_cost = np.sum(
        np.array(grid_import) * price * dt
    )

    objective = total_cost + 1000 * penalty

    return (
        np.array(soc_values),
        np.array(grid_import),
        total_cost,
        penalty,
        objective
    )


num_particles = 40
num_iterations = 150

w = 0.7
c1 = 1.5
c2 = 1.5

np.random.seed(42)

particles = np.random.uniform(
    -max_discharge_power,
    max_charge_power,
    (num_particles, 24)
)

velocities = np.zeros((num_particles, 24))

personal_best_positions = particles.copy()
personal_best_scores = np.full(num_particles, np.inf)

global_best_position = None
global_best_score = np.inf


for i in range(num_particles):

    _, _, _, _, score = simulate_battery(
        particles[i]
    )

    personal_best_scores[i] = score

    if score < global_best_score:
        global_best_score = score
        global_best_position = particles[i].copy()


for iteration in range(num_iterations):

    for i in range(num_particles):

        r1 = np.random.rand(24)
        r2 = np.random.rand(24)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 *
            (personal_best_positions[i] - particles[i])
            + c2 * r2 *
            (global_best_position - particles[i])
        )

        particles[i] = particles[i] + velocities[i]

        particles[i] = np.clip(
            particles[i],
            -max_discharge_power,
            max_charge_power
        )

        _, _, _, _, score = simulate_battery(
            particles[i]
        )

        if score < personal_best_scores[i]:

            personal_best_scores[i] = score
            personal_best_positions[i] = particles[i].copy()

        if score < global_best_score:

            global_best_score = score
            global_best_position = particles[i].copy()

    if (iteration + 1) % 10 == 0:
        print(
            f"Iteration {iteration + 1:3d} | "
            f"Best Objective = {global_best_score:.4f}"
        )


optimal_schedule = global_best_position

(
    soc,
    grid_import,
    optimized_cost,
    penalty,
    objective
) = simulate_battery(optimal_schedule)


baseline_grid = load.copy()

baseline_cost = np.sum(
    baseline_grid * price * dt
)

savings = baseline_cost - optimized_cost

savings_percentage = (
    savings / baseline_cost
) * 100


print("\n")
print("=" * 90)
print("       BATTERY ENERGY STORAGE SCHEDULING USING PSO")
print("=" * 90)

print(f"Battery Capacity       : {battery_capacity:.1f} kWh")
print(f"Initial SOC            : {initial_soc:.1f} kWh")
print(f"Minimum SOC            : {minimum_soc:.1f} kWh")
print(f"Maximum SOC            : {maximum_soc:.1f} kWh")
print(f"Maximum Charge Power   : {max_charge_power:.1f} kW")
print(f"Maximum Discharge Power: {max_discharge_power:.1f} kW")

print("\n")
print("-" * 90)

print(
    f"{'Hour':<6}"
    f"{'Load(kW)':<12}"
    f"{'Price($/kWh)':<16}"
    f"{'Battery(kW)':<16}"
    f"{'SOC(kWh)':<14}"
    f"{'Grid(kW)':<12}"
)

print("-" * 90)

for t in range(24):

    if optimal_schedule[t] > 0:
        action = optimal_schedule[t]
    else:
        action = optimal_schedule[t]

    print(
        f"{t:02d}:00  "
        f"{load[t]:<12.2f}"
        f"{price[t]:<16.2f}"
        f"{action:<16.2f}"
        f"{soc[t]:<14.2f}"
        f"{grid_import[t]:<12.2f}"
    )

print("-" * 90)

print("\n")
print("=" * 50)
print("              FINAL RESULTS")
print("=" * 50)

print(f"Electricity cost WITHOUT battery : ${baseline_cost:.2f}")
print(f"Electricity cost WITH PSO battery: ${optimized_cost:.2f}")
print(f"Cost savings                     : ${savings:.2f}")
print(f"Saving percentage                : {savings_percentage:.2f}%")

print(f"\nInitial battery SOC              : {initial_soc:.2f} kWh")
print(f"Final battery SOC                : {soc[-1]:.2f} kWh")

print(
    f"Maximum SOC reached              : "
    f"{np.max(soc):.2f} kWh"
)

print(
    f"Minimum SOC reached              : "
    f"{np.min(soc):.2f} kWh"
)

print("\nBattery Scheduling Decision:")

for t in range(24):

    if optimal_schedule[t] > 0.05:
        print(
            f"{t:02d}:00 -> CHARGE "
            f"({optimal_schedule[t]:.2f} kW)"
        )

    elif optimal_schedule[t] < -0.05:
        print(
            f"{t:02d}:00 -> DISCHARGE "
            f"({abs(optimal_schedule[t]):.2f} kW)"
        )

    else:
        print(
            f"{t:02d}:00 -> IDLE"
        )

print("=" * 50)
