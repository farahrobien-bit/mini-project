import numpy as np
import matplotlib.pyplot as plt



# Ellipse and print speed
a_e = 0.04          # semi-axis in x [m]
b_e = 0.025         # semi-axis in y [m]
v = 0.03            # print speed [m/s]

# Nozzle
T_e = 20.0          # environment temperature [degC]
R_n = 10.0          # nozzle resistor [Ohm]
R_tn = 13.2         # nozzle thermal resistance [K/W]
u_0 = 12.0          # nozzle voltage [V]
T_0 = T_e + R_tn / R_n * u_0**2     # temperature of the filament leaving the nozzle [degC]

# Print bed
T_b = 60.0          # bed temperature [degC]

# Filament (PLA) and heat transfer
rho = 1240.0        # density [kg/m^3]
c = 1800.0          # specific heat [J/(kg K)]
kappa = 0.13        # thermal conductivity [W/(m K)]
w = 0.4e-3          # side of the square cross-section [m]
h_b = 200.0         # heat transfer filament -> bed [W/(m^2 K)]
h_e = 20.0          # heat transfer filament -> air [W/(m^2 K)]

# Numerics
N = 400             # number of cells along the ellipse
t_end = 60.0        # maximum simulation time [s]
tol_unif = 1.0      # nearly uniform when max(T) - min(T) < tol_unif [K]
t_extra = 2.0       # keep simulating this long after it is uniform [s]

phi = np.linspace(0, 2 * np.pi, 20001)
xe = a_e * np.cos(phi)
ye = b_e * np.sin(phi)

s_fine = np.zeros(len(phi))
for i in range(1, len(phi)):
    stap = np.sqrt((xe[i] - xe[i-1])**2 + (ye[i] - ye[i-1])**2)
    s_fine[i] = s_fine[i-1] + stap

L = s_fine[-1]      # perimeter [m]

# formula to check the perimeter
L_ram = np.pi * (3 * (a_e + b_e) - np.sqrt((3 * a_e + b_e) * (a_e + 3 * b_e)))

dx = L / N          # length of one cell [m]

# Cell centres, cell positions, and the times the nozzle arrives and leaves
s_c = np.zeros(N)
t_start = np.zeros(N)
t_leave = np.zeros(N)
for i in range(N):
    s_c[i] = (i + 0.5) * dx
    t_start[i] = i * dx / v             # nozzle reaches cell i
    t_leave[i] = (i + 1) * dx / v       # nozzle leaves cell i

x_c = np.interp(s_c, s_fine, xe)        # x position of each cell (for plotting)
y_c = np.interp(s_c, s_fine, ye)        # y position of each cell (for plotting)

t_print = L / v     # time to print the whole layer [s]


A = w**2                        # cross-sectional area [m^2]
C = rho * c * A * dx            # heat capacity of one cell [J/K]
G_s = kappa * A / dx            # conductance between two cells [W/K]
G_b = h_b * w * dx              # conductance cell -> bed [W/K]
G_e = h_e * 3 * w * dx          # conductance cell -> air [W/K]

# Temperature at the end (expected)
T_inf = (G_b * T_b + G_e * T_e) / (G_b + G_e)

# Time step (stability)
dt_diff = C / (2 * G_s)         # because of conduction
dt_loss = C / (G_b + G_e)       # because of losses to bed and air
dt_act = dx / v                 # nozzle moving over one cell
dt = 0.2 * min(dt_diff, dt_loss, dt_act)

print(f"Perimeter L          = {L*1e3:.2f} mm (Ramanujan: {L_ram*1e3:.2f} mm)")
print(f"Print time t_print   = {t_print:.2f} s")
print(f"Nozzle temperature   = {T_0:.1f} degC")
print(f"Expected final T_inf = {T_inf:.1f} degC")
print(f"dt = {dt*1e3:.2f} ms  (limits: diff {dt_diff:.3f} s, loss {dt_loss:.3f} s, nozzle {dt_act*1e3:.1f} ms)")


T = np.full(N, np.nan)          # nan means: not printed yet
active = np.zeros(N, dtype=bool)  # True when the nozzle has reached the cell

t = 0.0
step = 0
n_save = max(1, int(round(0.05 / dt)))   # save about every 0.05 s
t_hist = []
T_hist = []
t_uniform = None

while t < t_end:

    # Cells that the nozzle reaches now get the nozzle temperature
    for i in range(N):
        if active[i] == False and t >= t_start[i]:
            T[i] = T_0
            active[i] = True

    # Temperature of every cell (0 for cells that are not printed yet)
    Ta = np.zeros(N)
    for i in range(N):
        if active[i]:
            Ta[i] = T[i]

    # Heat flow between neighbouring cells (from cell i to cell i+1)
    flow = np.zeros(N)
    for i in range(N - 1):
        if active[i] and active[i + 1]:
            q = G_s * (Ta[i] - Ta[i + 1])
            flow[i] = flow[i] - q
            flow[i + 1] = flow[i + 1] + q

    # When the ellipse is closed, the last cell touches the first cell
    if t >= t_print:
        q_wrap = G_s * (T[N - 1] - T[0])
        flow[N - 1] = flow[N - 1] - q_wrap
        flow[0] = flow[0] + q_wrap

    # Heat loss to the bed and to the air
    loss = np.zeros(N)
    for i in range(N):
        loss[i] = G_b * (T_b - Ta[i]) + G_e * (T_e - Ta[i])

    # Forward Euler: new temperature of the printed cells
    dT = np.zeros(N)
    for i in range(N):
        dT[i] = (flow[i] + loss[i]) / C
    for i in range(N):
        if active[i]:
            T[i] = T[i] + dt * dT[i]

    # The cell under the nozzle stays at the nozzle temperature
    for i in range(N):
        if active[i] and t < t_leave[i]:
            T[i] = T_0

    t = t + dt
    step = step + 1

    # Save the temperatures
    if step % n_save == 0:
        t_hist.append(t)
        T_hist.append(T.copy())

    # Check if the layer is (nearly) uniform, only after printing is done
    if t_uniform is None and t > t_print:
        if np.nanmax(T) - np.nanmin(T) < tol_unif:
            t_uniform = t
            t_end = min(t_end, t + t_extra)

t_hist = np.array(t_hist)
T_hist = np.array(T_hist)

if t_uniform is not None:
    print(f"\nLayer nearly uniform (spread < {tol_unif} K) at t = {t_uniform:.2f} s")
    print(f"  = {t_uniform - t_print:.2f} s after printing finished")
    print(f"  mean temperature then: {np.nanmean(T):.1f} degC")
else:
    print("\nLayer did not become uniform within t_end; increase t_end.")


# Max, mean and min temperature at every saved moment
T_max = []
T_mean = []
T_min = []
for j in range(len(t_hist)):
    T_max.append(np.nanmax(T_hist[j]))
    T_mean.append(np.nanmean(T_hist[j]))
    T_min.append(np.nanmin(T_hist[j]))

# Nozzle position at every saved moment (not further than the end of the path)
nozzle_pos = []
for j in range(len(t_hist)):
    pos = v * t_hist[j]
    if pos > L:
        pos = L
    nozzle_pos.append(pos * 1e3)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot a: heat map of T(s,t)
im = ax[0].pcolormesh(s_c * 1e3, t_hist, np.ma.masked_invalid(T_hist),
                      shading="auto", cmap="inferno")
ax[0].plot(nozzle_pos, t_hist, "c--", lw=1, label="nozzle position")
ax[0].axhline(t_print, color="w", lw=0.8, ls=":")
ax[0].set_xlabel("position along path s [mm]")
ax[0].set_ylabel("time t [s]")
ax[0].set_title("Filament temperature T(s, t)")
ax[0].legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=ax[0], label="T [°C]")

# Plot b: max, mean and min temperature over time
ax[1].plot(t_hist, T_max, label="max T")
ax[1].plot(t_hist, T_mean, label="mean T")
ax[1].plot(t_hist, T_min, label="min T")
ax[1].axhline(T_inf, color="k", ls="--", lw=0.8, label=r"$T_\infty$ (expected)")
ax[1].axvline(t_print, color="gray", ls=":", lw=1, label="printing done")
if t_uniform is not None:
    ax[1].axvline(t_uniform, color="r", ls=":", lw=1, label="nearly uniform")
ax[1].set_xlabel("time t [s]")
ax[1].set_ylabel("T [°C]")
ax[1].set_title("Temperature spread in the layer")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("q16_temperature.png", dpi=150)

# Plot c: the ellipse coloured by temperature at four moments
if t_uniform is not None:
    t_last = t_uniform
else:
    t_last = t_hist[-1]
moments = [0.5 * t_print, t_print, t_print + 2, t_last]

fig2, ax2 = plt.subplots(1, len(moments), figsize=(14, 3.6))
for k in range(len(moments)):
    # find the saved moment that is closest to the wanted moment
    j = np.argmin(np.abs(t_hist - moments[k]))
    ax2[k].plot(xe * 1e3, ye * 1e3, color="lightgray", lw=1, zorder=0)
    sc = ax2[k].scatter(x_c * 1e3, y_c * 1e3, c=T_hist[j], cmap="inferno",
                        vmin=T_inf - 5, vmax=T_0, s=10, linewidths=0)
    ax2[k].set_title(f"t = {t_hist[j]:.1f} s")
    ax2[k].set_aspect("equal")
    ax2[k].set_xlabel("x [mm]")
ax2[0].set_ylabel("y [mm]")
fig2.colorbar(sc, ax=ax2, label="T [°C]")
fig2.savefig("q16_ellipse.png", dpi=150, bbox_inches="tight")

plt.show()
print("File saved as cooling_model.py")