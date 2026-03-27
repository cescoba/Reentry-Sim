import numpy as np
from scipy.integrate import solve_ivp

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

def derivatives(t, state):
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

    drag_mag = 0.5 * rho * v_mag**2 * CD * A_REF
    drag_vec = -(drag_mag / MASS) * v_hat  # drag opposes velocity

    # --- Total acceleration ---
    ax = g_vec[0] + drag_vec[0]
    ay = g_vec[1] + drag_vec[1]
    az = g_vec[2] + drag_vec[2]

    # --- Return derivatives ---
    # d/dt [x, y, z, vx, vy, vz] = [vx, vy, vz, ax, ay, az]
    return [vx, vy, vz, ax, ay, az]
