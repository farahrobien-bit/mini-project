# Q3.1 Hybrid system: filament replacement
# state z = [x, dx, ix, y, dy, iy, Tn, u2, iLn, phi, ell, tau]

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# motor parameters (same as Q1.4)
r = 0.02            # pulley radius [m]
L_m = 0.005         # motor inductance [H]
R_m = 1.0           # motor resistance [Ohm]
K_m = 0.1           # motor constant [Nm/A]
J_m = 2e-5          # motor inertia [kg m^2]
m = 0.5             # mass of tool head [kg]
mu = 1.0            # friction in x and y direction [N s/m]
J_eq = J_m + 0.5 * m * r**2

# nozzle (same as Q1.5 / Q1.6)
T_e = 20.0          # environment temperature [C]
R_n = 10.0          # nozzle resistor [Ohm]
R_tn = 13.2         # nozzle thermal resistance [K/W]
u_0 = 12.0          # nozzle voltage [V]
C_tn, C_n, L_n = 1.0, 1e-3, 1e-2            # thermal capacity [J/K], capacitor [F], inductor [H]
T_0 = T_e + R_tn / R_n * u_0**2             # nozzle temperature while printing (about 210 C)

# ellipse and print speed (same as Q1.4 / Q1.6)
a_e = 0.04          # semi-axis in x [m]
b_e = 0.025         # semi-axis in y [m]
v = 0.03            # print speed [m/s]
x_c, y_c = 0.06, 0.05                       # centre of the ellipse on the print bed [m]
L_ell = np.pi * (3 * (a_e + b_e) - np.sqrt((3 * a_e + b_e) * (a_e + 3 * b_e)))   # perimeter [m]
Omega = 2 * np.pi * v / L_ell               # angular speed along the ellipse, one round in L/v seconds

# controller
K_p, K_d = 380.0, 4.5
K_ff = K_m / r + R_m * r * mu / (2 * K_m)   # voltage per m/s at constant speed
U_max, v_travel = 24.0, 0.05                # voltage limit [V], travel speed [m/s]

# event values
eps_w, eps_x = 0.2, 0.5e-3                  # stopped [rad/s], arrived [m]
ell_spool, t_C = 0.30, 65.0                 # spool length [m], time of event C [s]
ell_start = 0.24                            # the first spool is almost empty: only 6 cm left [m]
t_wipe, t_load = 2.0, 5.0                   # [s]
T_low, T_high = 80.0, 200.0                 # [C]

saved = [0.0, 0.0]                          # saved print position (x_s, y_s)
C_done = False
colour = "gold"                             # colour of the filament in the printer
last_event = ""


def omega(z):                               # angular speed of the motors (Q1.1)
    return max(abs(z[1] + z[4]), abs(z[1] - z[4])) / r


def move_to(z, x_ref, y_ref, dx_ref=0.0, dy_ref=0.0):
    # PD controller + the voltage that is needed to move with the reference speed (from eq. 47 and 49)
    u_x = K_p * (x_ref - z[0]) + K_d * (dx_ref - z[1]) + K_ff * dx_ref
    u_y = K_p * (y_ref - z[3]) + K_d * (dy_ref - z[4]) + K_ff * dy_ref
    return np.clip(u_x, -U_max, U_max), np.clip(u_y, -U_max, U_max)


def travel(z, x_start, y_start, x_end, y_end):
    # the reference moves in a straight line with speed v_travel (tau is the time in this mode)
    dist = np.hypot(x_end - x_start, y_end - y_start)
    s = min(v_travel * z[11], dist)
    ex, ey = (x_end - x_start) / dist, (y_end - y_start) / dist
    v = v_travel if s < dist else 0.0
    return move_to(z, x_start + s * ex, y_start + s * ey, v * ex, v * ey)


def inputs(mode, z):                        # u = [u_x, u_y, u_n]
    u_n = 0.0 if mode in ["HeatOff", "Loading"] else u_0
    if mode == "Printing":
        p = z[9]
        u_x, u_y = move_to(z, x_c + a_e * np.cos(p), y_c + b_e * np.sin(p),
                           -a_e * Omega * np.sin(p), b_e * Omega * np.cos(p))
    elif mode == "Stop":
        u_x, u_y = move_to(z, saved[0], saved[1])
    elif mode == "MoveV":                   # down to the bottom edge
        u_x, u_y = travel(z, saved[0], saved[1], saved[0], 0.0)
    elif mode == "MoveH":                   # left to the corner (0, 0)
        u_x, u_y = travel(z, saved[0], 0.0, 0.0, 0.0)
    elif mode == "Resume":                  # back to the saved position
        u_x, u_y = travel(z, 0.0, 0.0, saved[0], saved[1])
    else:                                   # parked at the wipe tower
        u_x, u_y = move_to(z, 0.0, 0.0)
    return u_x, u_y, u_n


def f(z, mode):                             # dz/dt = f(z, u), same physics in every mode
    x, dx, ix, y, dy, iy, Tn, u2, iLn, phi, ell, tau = z
    u_x, u_y, u_n = inputs(mode, z)
    return [dx, (r * K_m * ix - 0.5 * r**2 * mu * dx) / J_eq, (-R_m * ix - K_m / r * dx + u_x) / L_m,
            dy, (r * K_m * iy - 0.5 * r**2 * mu * dy) / J_eq, (-R_m * iy - K_m / r * dy + u_y) / L_m,
            (u2**2 / R_n - (Tn - T_e) / R_tn) / C_tn, (iLn - u2 / R_n) / C_n, (u_n - u2) / L_n,
            Omega if mode == "Printing" else 0.0,                       # phi
            np.hypot(dx, dy) if mode in ["Printing", "Wiping"] else 0.0,  # ell
            1.0]                                                       # tau


def guard(t, z, mode):                      # event when this becomes zero
    if mode == "Printing":
        return min(1.0 if C_done else t_C - t, ell_spool - z[10],   # C or F
                   2 * np.pi - z[9])                              # ellipse finished
    if mode == "Stop":
        return omega(z) - eps_w                                     # motors stopped
    if mode in ["MoveV", "MoveH"]:
        return max(omega(z) - eps_w, 0.3 - z[11])                   # (V,0) / (H,0): stopped, after 0.3 s
    if mode == "Wiping":
        return t_wipe - z[11]                                       # E
    if mode == "HeatOff":
        return z[6] - T_low                                         # T low
    if mode == "Loading":
        return t_load - z[11]                                       # loaded
    if mode == "HeatOn":
        return T_high - z[6]                                        # T high
    if mode == "Resume":
        return max(np.hypot(z[0] - saved[0], z[3] - saved[1]) - eps_x, omega(z) - eps_w)


ORDER = ["Printing", "Stop", "MoveV", "MoveH", "Wiping", "HeatOff", "Loading", "HeatOn", "Resume"]


def next_mode(mode, t, z):                  # next mode + reset maps
    global C_done, colour, last_event
    if mode == "Printing":
        if z[9] >= 2 * np.pi - 1e-6:
            return "Done", z                # the whole ellipse is printed
        if not C_done and t >= t_C - 1e-6:
            C_done, last_event = True, "C"
        else:
            last_event = "F"
        saved[0], saved[1] = z[0], z[3]     # save position
    if mode == "Loading":
        z[10] = 0.0                         # new spool: ell = 0
        if last_event == "C":
            colour = "purple"               # new colour is loaded
    z[11] = 0.0                             # timer = 0
    return ORDER[(ORDER.index(mode) + 1) % len(ORDER)], z


# simulation
z = np.array([x_c + a_e, 0, 0, y_c, 0, 0, T_0, u_0, u_0 / R_n, 0, ell_start, 0])
mode, t = "Printing", 0.0
T, Z, Q, COL = [], [], [], []
while t < 140 and mode != "Done":
    event = lambda t, z: guard(t, z, mode)
    event.terminal, event.direction = True, -1
    sol = solve_ivp(lambda t, z: f(z, mode), [t, 140], z, method="LSODA",
                    events=event, max_step=0.02, rtol=1e-6, atol=1e-9)
    T += list(sol.t); Z += list(sol.y.T); Q += [ORDER.index(mode)] * len(sol.t)
    COL += [colour if mode == "Printing" else "travel"] * len(sol.t)
    t, z = sol.t[-1], sol.y[:, -1].copy()
    if len(sol.t_events[0]) > 0:
        print(f"t = {t:6.2f} s: {mode} -> ", end="")
        mode, z = next_mode(mode, t, z)
        print(mode)
T, Z = np.array(T), np.array(Z)

# plots
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].step(T, Q, where="post"); ax[0].set_yticks(range(9)); ax[0].set_yticklabels(ORDER, fontsize=7)
ax[0].invert_yaxis(); ax[0].set_title("Mode")
ax[1].plot(T, Z[:, 0] * 1e3, label="x"); ax[1].plot(T, Z[:, 3] * 1e3, label="y")
ax[1].set_ylabel("position [mm]"); ax[1].legend()
ax[2].plot(T, Z[:, 6], "r"); ax[2].axhline(T_low, ls="--", c="gray"); ax[2].axhline(T_high, ls="--", c="gray")
ax[2].set_ylabel("nozzle T [C]"); ax[2].set_xlabel("time [s]")
plt.tight_layout(); plt.savefig("q31_time.png", dpi=150)

plt.figure(figsize=(6, 5))
plt.plot(Z[:, 0] * 1e3, Z[:, 3] * 1e3, color="lightgray", ls="--", lw=1, label="moving (no printing)")
for col, name in [("gold", "printed with yellow filament"), ("purple", "printed with purple filament")]:
    xs = np.where(np.array(COL) == col, Z[:, 0] * 1e3, np.nan)   # nan = do not draw this point
    ys = np.where(np.array(COL) == col, Z[:, 3] * 1e3, np.nan)
    plt.plot(xs, ys, color=col, lw=3, label=name)
plt.plot(0, 0, "ks", label="wipe tower")
plt.xlabel("x [mm]"); plt.ylabel("y [mm]"); plt.axis("equal"); plt.legend(fontsize=8)
plt.title("Path of the tool head"); plt.tight_layout(); plt.savefig("q31_xy.png", dpi=150)
plt.show()