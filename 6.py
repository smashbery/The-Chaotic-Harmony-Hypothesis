import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Simulation Parameters
num_particles = 100  # Number of particles
space_size = 10  # Size of the 2D space
foundry_positions = [np.array([0.0, 0.0]), np.array([5.0, 5.0]), np.array([-5.0, -5.0])]  # Initial foundry locations
gravitational_constants = [0.1, 0.08, 0.12]  # Initial gravitational constants for each foundry
compression_radii = [0.5, 0.7, 0.4]  # Initial compression radii for each foundry
collision_radius = 0.2  # Radius within which particles can collide and merge
frames = 200  # Number of simulation frames
angular_velocity = 0.05  # Rotational force multiplier
foundry_growth_rate = [0.01, -0.005, 0.008]  # Growth/shrink rates for each foundry
foundry_motion_vectors = [np.array([0.01, 0]), np.array([-0.01, 0.01]), np.array([0.02, -0.01])]  # Motion vectors

# Initialize Particle Properties
np.random.seed(42)  # For reproducibility
particle_positions = np.random.uniform(-space_size, space_size, (num_particles, 2))
particle_masses = np.random.uniform(1, 5, num_particles)  # Assign random masses
particle_states = np.zeros(num_particles, dtype=int)  # States (0: raw, 1: compressed, 2: ejected)
particle_temperatures = np.zeros(num_particles)  # Initialize temperatures
particle_trails = [[] for _ in range(num_particles)]  # Initialize trails for each particle

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))

for frame in range(frames):
    ax.clear()
    ax.set_xlim(-space_size, space_size)
    ax.set_ylim(-space_size, space_size)
    ax.set_title(f"Cosmic Foundry Simulation - Frame {frame}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Update foundry positions and properties
    for i, (fp, growth_rate, motion_vector) in enumerate(zip(foundry_positions, foundry_growth_rate, foundry_motion_vectors)):
        foundry_positions[i] += motion_vector  # Move the foundry
        compression_radii[i] = max(0.1, compression_radii[i] + growth_rate)  # Adjust radius, ensuring it stays positive
        gravitational_constants[i] = max(0.05, gravitational_constants[i] + growth_rate * 0.1)  # Adjust gravity

    # Calculate distances to each foundry
    all_distances = [cdist(particle_positions, [fp]).flatten() for fp in foundry_positions]

    # Collision detection and merging
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            if particle_states[i] == 0 and particle_states[j] == 0:
                dist = np.linalg.norm(particle_positions[i] - particle_positions[j])
                if dist <= collision_radius:
                    # Merge particles
                    total_mass = particle_masses[i] + particle_masses[j]
                    particle_positions[i] = (
                        (particle_positions[i] * particle_masses[i] + particle_positions[j] * particle_masses[j]) / total_mass
                    )
                    particle_masses[i] = total_mass
                    particle_positions[j] = np.array([np.nan, np.nan])  # Remove the second particle
                    particle_states[j] = -1  # Mark as inactive

    for i, position in enumerate(particle_positions):
        if particle_states[i] < 0:
            continue  # Skip inactive particles

        # Determine the closest foundry
        distances = np.array([d[i] for d in all_distances])
        closest_foundry_idx = np.argmin(distances)
        closest_foundry_pos = foundry_positions[closest_foundry_idx]
        distance = distances[closest_foundry_idx]
        gravitational_constant = gravitational_constants[closest_foundry_idx]
        compression_radius = compression_radii[closest_foundry_idx]

        if particle_states[i] == 0:  # Raw state
            # Apply gravitational pull based on mass
            direction = closest_foundry_pos - position
            gravitational_force = (gravitational_constant * particle_masses[i]) / (distance + 1e-5)  # Avoid division by zero

            # Add rotational force
            perpendicular_direction = np.array([-direction[1], direction[0]])  # Perpendicular vector
            particle_positions[i] += (gravitational_force * direction / np.linalg.norm(direction))
            particle_positions[i] += angular_velocity * perpendicular_direction / np.linalg.norm(perpendicular_direction)

            # Increase temperature as particle approaches the foundry
            particle_temperatures[i] = (1 / (distance + 1e-5)) * 100

            # Check for compression
            if distance <= compression_radius:
                particle_states[i] = 1

        elif particle_states[i] == 1:  # Compressed state
            # Eject as Hawking radiation
            ejection_angle = np.random.uniform(0, 2 * np.pi)
            ejection_velocity = np.random.uniform(0.5, 1.0)
            particle_positions[i] += ejection_velocity * np.array([np.cos(ejection_angle), np.sin(ejection_angle)])
            particle_states[i] = 2

        # Update particle trails
        if particle_states[i] >= 0:  # Active particles only
            particle_trails[i].append(particle_positions[i].copy())

    # Plot energy fields around foundries
    for fp, cr in zip(foundry_positions, compression_radii):
        circle = plt.Circle(fp, cr, color="green", fill=False, linestyle="--", alpha=0.5)
        ax.add_artist(circle)

    # Plot particle trails
    for trail in particle_trails:
        if len(trail) > 1:
            trail = np.array(trail)
            ax.plot(trail[:, 0], trail[:, 1], color="gray", linewidth=0.5, alpha=0.5)

    # Plot particles
    for state, color in zip([0, 1, 2], ["blue", "orange", "red"]):
        state_indices = np.where(particle_states == state)[0]
        temperatures = particle_temperatures[state_indices] if state == 0 else None
        scatter = ax.scatter(
            particle_positions[state_indices, 0],
            particle_positions[state_indices, 1],
            c=temperatures if state == 0 else color,
            cmap="hot" if state == 0 else None,
            label=f"State {state}"
        )
        if state == 0: scatter.set_clim(0, 1000)  # Adjust temperature color scale

    # Plot foundries
    for fp in foundry_positions:
        ax.scatter(*fp, color="black", s=200, label="Foundry")

    ax.legend()
    plt.pause(0.05)

plt.show()
