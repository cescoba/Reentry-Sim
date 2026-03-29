import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Constants ---
R_EARTH = 6_371_000      # Earth's mean radius, meters
G_EARTH = 9.80665        # surface gravity, m/s^2
GM      = 3.986004418e14 # gravitational parameter, m^3/s^2

# --- Initial Conditions ---
altitude_0  = 120_000    # entry interface, 120 km in meters
velocity_0  = 7_800      # m/s horizontal at entry

# Position: place capsule directly above equator on the x-axis
x0 = R_EARTH + altitude_0
y0 = 0.0
z0 = 0.0

# Velocity: moving horizontally (along y-axis in ECI)
vx0 = 0.0
vy0 = velocity_0
vz0 = 0.0

# Pack into state vector
state0 = np.array([x0, y0, z0, vx0, vy0, vz0])

print("Initial state vector:")
print(f"  Position: ({x0/1000:.1f}, {y0:.1f}, {z0:.1f}) km")
print(f"  Velocity: ({vx0:.1f}, {vy0:.1f}, {vz0:.1f}) m/s")
print(f"  Altitude: {altitude_0/1000:.1f} km")

# --- Capsule Properties ---
MASS    = 12_000     # kg, approximate Dragon capsule mass
CD      = 1.3        # drag coefficient (blunt body capsule)
A_REF   = 10.0       # reference area m^2, approx heat shield diameter

# -- Atmospheric Model --

def atmosphere_density(altitude):

    rho_0 = 1.225 # Sl Density kg/m^3
    H     = 8_500 # meters

    if altitude < 0:
        return rho_0
    if altitude > 80_000:    # above 80km (density is negligible)
        return 1e-8          # near vacuum (drag negligible)

    return rho_0 * np.exp(-altitude / H)

def derivatives(t, state, cd=CD):
    x, y, z, vx, vy, vz = state	# FOR READABILITY - used in r_vec and r_mag to make cleaner

    # --- Position vector and altitude ---
    r_vec = np.array([x, y, z])
    r_mag = np.linalg.norm(r_vec)          # distance from Earth's center
    altitude = r_mag - R_EARTH

    # --- Gravity ---
    # Points toward Earth's center (negative r direction)
    g_vec = -(GM / r_mag**2) * (r_vec / r_mag)

    # --- Atmosphere (US Standard, simple exponential) ---
    rho = atmosphere_density(altitude)

    # --- Aerodynamic drag ---
    v_vec = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_vec)
    v_hat = v_vec / v_mag                  # unit vector in velocity direction

    drag_mag = 0.5 * rho * v_mag**2 * cd * A_REF
    drag_vec = -(drag_mag / MASS) * v_hat  # drag opposes velocity

    # --- Total acceleration ---
    ax = g_vec[0] + drag_vec[0]
    ay = g_vec[1] + drag_vec[1]
    az = g_vec[2] + drag_vec[2]

    # --- Return derivatives ---
    # d/dt [x, y, z, vx, vy, vz] = [vx, vy, vz, ax, ay, az]
    return [vx, vy, vz, ax, ay, az]

derivs = derivatives(0, state0)
print("\nDerivatives at t=0:")
print(f"  dx/dt:  {derivs[0]/1000:.2f} km/s")
print(f"  dy/dt:  {derivs[1]/1000:.2f} km/s")
print(f"  ax:     {derivs[3]:.4f} m/s^2")
print(f"  ay:     {derivs[4]:.4f} m/s^2")

## Simulation ##

def ground (t, state):
    x,y,z = state[0], state[1], state[2]
    r_mag = np.sqrt(x**2 + y**2 + z**2)
    return r_mag - R_EARTH   # returns 0 when altitude = 0 (ground)

ground.terminal  = True   # stop integration when this hits 0
ground.direction = -1     # only trigger when descending

t_0 = 0     # seconds
t_f = 2000   # seconds
sol = solve_ivp( 
    fun = derivatives,
    t_span = (t_0,t_f),
    y0 = state0,
    method = 'RK45',
    max_step = 1.0, 
    dense_output = True,
    events = ground
)

print("\nSimulation complete:")
print(f"  Status:      {sol.message}")
print(f"  Time steps:  {len(sol.t)}")
print(f"  Final time:  {sol.t[-1]:.1f} s")

x   = sol.y[0]
y   = sol.y[1]
z   = sol.y[2]
vx  = sol.y[3]
vy  = sol.y[4]
vz  = sol.y[5]

r_mag    = np.sqrt(x**2 + y**2 + z**2)
altitude = (r_mag - R_EARTH) / 1000     # convert to km

speed = np.sqrt(vx**2 + vy**2 + vz**2) / 1000  # convert to km/s

print(f"  Initial altitude: {altitude[0]:.1f} km")
print(f"  Final altitude:   {altitude[-1]:.1f} km")
print(f"  Initial speed:    {speed[0]:.2f} km/s")
print(f"  Final speed:      {speed[-1]:.2f} km/s")

## Plot Figure ##

fig, axes = plt.subplots(3, 1, figsize = (10, 12))
fig.suptitle('Reentry Trajectory - 3DOF Simulation', fontsize = 14)

# --- Plot 1: Altitude vs Time ---
axes[0].plot(sol.t, altitude, color='steelblue')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Altitude (km)')
axes[0].set_title('Altitude vs Time')
axes[0].grid(True)

# --- Plot 2: Speed vs Time ---
axes[1].plot(sol.t, speed, color='coral')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Speed (km/s)')
axes[1].set_title('Speed vs Time')
axes[1].grid(True)

# --- Plot 3: Speed vs Altitude ---
axes[2].plot(altitude, speed, color='purple')
axes[2].set_xlabel('Altitude (km)')
axes[2].set_ylabel('Speed (km/s)')
axes[2].set_title('Speed vs Altitude')
axes[2].invert_xaxis()   # flip so altitude decreases left to right
axes[2].grid(True)

plt.tight_layout()
plt.savefig('reentry_trajectory.png', dpi=150)
plt.show()

print("\nPlot saved as reentry_trajectory.png")

# --- Monte Carlo Setup ---
N_RUNS = 500     # number of simulation runs

# Uncertainty (standard deviation) on initial conditions
sigma_velocity  = 50      # m/s  — 1-sigma velocity uncertainty
sigma_altitude  = 2_000   # m    — 1-sigma altitude uncertainty
sigma_cd        = 0.05    # drag coefficient uncertainty
sigma_vz        = 50      # out of plane uncertainty

# Store landing results
landing_y   = []
landing_z   = []
landing_t   = []

print(f"\nRunning {N_RUNS} Monte Carlo simulations...")

for i in range(N_RUNS):

    # --- Perturb initial conditions ---
    alt_perturbed = altitude_0 + np.random.normal(0, sigma_altitude)
    vel_perturbed = velocity_0 + np.random.normal(0, sigma_velocity)
    cd_perturbed  = CD + np.random.normal(0, sigma_cd)
    vz_perturbed  = np.random.normal(0, sigma_vz)

    # --- New initial state ---
    x0_mc    = R_EARTH + alt_perturbed
    state_mc = np.array([x0_mc, 0.0, 0.0, 0.0, vel_perturbed, vz_perturbed])

    # --- Integrate using a lambda to pass cd_perturbed ---
    sol_mc = solve_ivp(
        fun      = lambda t, s: derivatives(t, s, cd_perturbed),
        t_span   = (0, 2000),
        y0       = state_mc,
        method   = 'RK45',
        max_step = 1.0,
        events   = ground
    )

    # --- Store landing position ---
    landing_y.append(sol_mc.y[1, -1] / 1000)   # km
    landing_z.append(sol_mc.y[2, -1] / 1000)   # km
    landing_t.append(sol_mc.t[-1])

    if (i + 1) % 50 == 0:
        print(f"  Completed {i + 1}/{N_RUNS} runs...")

# --- Convert to arrays ---
landing_y = np.array(landing_y)
landing_z = np.array(landing_z)
landing_t = np.array(landing_t)

# --- Print Statistics ---
print(f"\nMonte Carlo Results ({N_RUNS} runs):")
print(f"  Landing Y — mean: {np.mean(landing_y):.1f} km  std: {np.std(landing_y):.1f} km")
print(f"  Landing Z — mean: {np.mean(landing_z):.1f} km  std: {np.std(landing_z):.1f} km")
print(f"  Landing time — mean: {np.mean(landing_t):.1f} s  std: {np.std(landing_t):.1f} s")

# --- Plot Landing Ellipse ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(landing_y, landing_z, alpha=0.3, s=10, color='steelblue', label='Landing points')
ax.scatter(np.mean(landing_y), np.mean(landing_z), color='red', s=100, zorder=5, label='Mean landing point')
ax.set_xlabel('Y position (km)')
ax.set_ylabel('Z position (km)')
ax.set_title(f'Landing Dispersion Ellipse — {N_RUNS} Monte Carlo Runs')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('monte_carlo_dispersion.png', dpi=150)
plt.show()

print("\nMonte Carlo plot saved as monte_carlo_dispersion.png")