import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


r, m, J_m = 0.01, 0.5, 1e-5           
K_m, R_m, L_m = 0.05, 2.0, 2e-3       
mu_x, mu_y = 1.0, 1.0                  
J_eq = J_m + 0.5 * m * r**2

T_e, R_n, R_tn, C_tn = 20.0, 10.0, 13.2, 1.0
C_n, L_n, u_0 = 1e-3, 1e-2, 12.0
T_0 = T_e + R_tn / R_n * u_0**2

# square
L_sq = 0.04                           
c0 = np.array([0.03, 0.03])          
v_print = 0.02                         
th_ref = L_sq / r                    
n_laps = 1                 
          
# controller and event thresholds
K_p, K_d, U_max = 350.0, 4.6, 24.0
eps_th = 1e-3                         
eps_w = 0.01                         

MODES = ["X+", "Y+", "X-", "Y-"]
DIR = {"X+": (1, 0), "Y+": (0, 1), "X-": (-1, 0), "Y-": (0, -1)}

def omega(z):
    return max(abs(z[1] + z[4]), abs(z[1] - z[4])) / r

def inputs(q, z, p_start):
   
    T = 1.875 * L_sq / v_print
    s = min(z[9] / T, 1.0)
    pos = L_sq * (10 * s**3 - 15 * s**4 + 6 * s**5)
    vel = L_sq / T * (30 * s**2 - 60 * s**3 + 30 * s**4) if s < 1 else 0.0
    e = np.array(DIR[q])
    xr, yr = p_start + pos * e
    vxr, vyr = vel * e
    # position follows from the measured angles
    u_x = np.clip(K_p * (xr - z[0]) + K_d * (vxr - z[1]), -U_max, U_max)
    u_y = np.clip(K_p * (yr - z[3]) + K_d * (vyr - z[4]), -U_max, U_max)
    return u_x, u_y, u_0

def f(z, u):
    x, vx, ix, y, vy, iy, Tn, u2, iL, tau, dth_l = z
    u_x, u_y, u_n = u
    return np.array([
        vx,
        (r * K_m * ix - 0.5 * r**2 * mu_x * vx) / J_eq,
        (-R_m * ix - K_m / r * vx + u_x) / L_m,
        vy,
        (r * K_m * iy - 0.5 * r**2 * mu_y * vy) / J_eq,
        (-R_m * iy - K_m / r * vy + u_y) / L_m,
        (u2**2 / R_n - (Tn - T_e) / R_tn) / C_tn,
        (iL - u2 / R_n) / C_n,
        (u_n - u2) / L_n,
        1.0,                              
        (vx + vy) / r,                     
    ])

def guard(q, z):
    if q in ("X+", "Y+"):
        dist_error = th_ref - eps_th - z[10]
    else:
        # Voor X- en Y- wordt z[10] negatief, dus we kijken naar de absolute waarde
        dist_error = th_ref - eps_th - abs(z[10])
        
    return max(dist_error, omega(z) - eps_w)    
#simulation
z = np.array([*[c0[0], 0, 0, c0[1], 0, 0], T_0, u_0, u_0 / R_n, 0.0, 0.0])
t, k = 0.0, 0
T_all, Z_all, Q_all, events = [], [], [], []

while k < 4 * n_laps:
    q = MODES[k % 4]
    p_start = np.array([z[0], z[3]])
    ev = lambda t, z: guard(q, z)
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(lambda t, z: f(z, inputs(q, z, p_start)), (t, t + 20), z,
                    method="LSODA", events=ev, max_step=0.01, rtol=1e-7, atol=1e-10)
    T_all.append(sol.t); Z_all.append(sol.y.T); Q_all += [k % 4] * sol.t.size
    t, z = sol.t[-1], sol.y[:, -1].copy()
    events.append(t)
    z[9], z[10] = 0.0, 0.0                 # reset: tau := 0, dth_l := 0
    k += 1

T = np.concatenate(T_all); Z = np.vstack(Z_all); Q = np.array(Q_all)
th_l, th_r = (Z[:, 0] + Z[:, 3]) / r, (Z[:, 0] - Z[:, 3]) / r
w_l, w_r = (Z[:, 1] + Z[:, 4]) / r, (Z[:, 1] - Z[:, 4]) / r

# accuracy: distance to the ideal square
corners = np.array([c0, c0 + [L_sq, 0], c0 + [L_sq, L_sq], c0 + [0, L_sq], c0])
def dist(p):
    d = []
    for a, b in zip(corners[:-1], corners[1:]):
        s = np.clip(np.dot(p - a, b - a) / L_sq**2, 0, 1)
        d.append(np.linalg.norm(p - (a + s * (b - a))))
    return min(d)
dev = max(dist(p) for p in Z[:, [0, 3]])

print("Corner events at t =", ", ".join(f"{te:.2f} s" for te in events))
print(f"Max deviation from the ideal square: {dev*1e6:.2f} um")
print(f"Nozzle temperature: {Z[:, 6].min():.1f} - {Z[:, 6].max():.1f} degC (heater on)")


# Plots
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(Z[:, 0] * 1e3, Z[:, 3] * 1e3, "k-", lw=1.5, label="printed path")
ax.plot(corners[:, 0] * 1e3, corners[:, 1] * 1e3, "r--", lw=0.8, label="ideal square")
ax.set_aspect("equal"); ax.set_xlim(25, 75); ax.set_ylim(25, 75)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title("Printed square")
ax.legend(loc="center"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("q32_square.png", dpi=150)

fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].step(T, Q, where="post", color="k")
ax[0].set_yticks(range(4)); ax[0].set_yticklabels([f"Draw {q}" for q in MODES])
ax[0].invert_yaxis(); ax[0].set_title("Discrete mode")
ax[1].plot(T, th_l, label=r"$\theta_l$"); ax[1].plot(T, th_r, label=r"$\theta_r$")
ax[1].set_ylabel("motor angle [rad]"); ax[1].legend(loc="upper right")
ax[2].plot(T, w_l, label=r"$\dot\theta_l$"); ax[2].plot(T, w_r, "--", label=r"$\dot\theta_r$")
ax[2].set_ylabel("angular speed [rad/s]"); ax[2].set_xlabel("time [s]"); ax[2].legend(loc="upper right")
for a in ax:
    a.grid(alpha=0.3)
    for te in events:
        a.axvline(te, color="k", lw=0.6, ls=":")
fig.tight_layout(); fig.savefig("q32_time.png", dpi=150)

plt.show()