import numpy as np
import matplotlib.pyplot as plt

r = 0.02       # pulley radi us 
Lm = 0.005     # motor inductance 
Rm = 1.0       # motor resistance 
Km = 0.1       # motor constant 
Jm = 2e-5      # motor inertia 
m = 0.5        # mass of tool head 
mu_x = 1.0     # friction in x direction 
mu_y = 1.0     # friction in y direction 

Jeq = Jm + 0.5 * m * r**2     # equivalent inertia 
cx = r**2 / 2 * mu_x          # friction term x
cy = r**2 / 2 * mu_y          # friction term  y

dt = 0.001
t = np.arange(0, 2 * np.pi, dt)


def design_voltage(pos, c):
    
    vel = np.gradient(pos, dt)
    acc = np.gradient(vel, dt)
    # mechanical equation
    i = (Jeq * acc + c * vel) / (r * Km)
    di = np.gradient(i, dt)
    # electrical equation
    u = Lm * di + Rm * i + Km / r * vel
    return u, i, vel


def simulate(ux, uy, x0, vx0, ix0, y0, vy0, iy0):
   
    n = len(ux)
    x = np.zeros(n)
    vx = np.zeros(n)
    ix = np.zeros(n)
    y = np.zeros(n)
    vy = np.zeros(n)
    iy = np.zeros(n)

    x[0], vx[0], ix[0] = x0, vx0, ix0
    y[0], vy[0], iy[0] = y0, vy0, iy0

    for k in range(n - 1):
        # x system
        x[k+1] = x[k] + dt * vx[k]
        vx[k+1] = vx[k] + dt * (r * Km * ix[k] - cx * vx[k]) / Jeq
        ix[k+1] = ix[k] + dt * (-Rm / Lm * ix[k] - Km / (r * Lm) * vx[k] + ux[k] / Lm)
        # y system
        y[k+1] = y[k] + dt * vy[k]
        vy[k+1] = vy[k] + dt * (r * Km * iy[k] - cy * vy[k]) / Jeq
        iy[k+1] = iy[k] + dt * (-Rm / Lm * iy[k] - Km / (r * Lm) * vy[k] + uy[k] / Lm)

    return x, y, ix, iy


def make_plots(name, colour, x_ref, y_ref, x, y, ux, uy, ix, iy):
    # motor voltages and currents
    um_l = ux + uy
    um_r = ux - uy
    im_l = ix + iy
    im_r = ix - iy
    # motor angles from the simulated position
    theta_l = (x + y) / r
    theta_r = (x - y) / r

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(name, fontsize=14, fontweight='bold')

    ax[0, 0].plot(x_ref, y_ref, 'k--', linewidth=3, label='wanted path')
    ax[0, 0].plot(x, y, colour, linewidth=1.5, label='simulated path')
    ax[0, 0].set_title('Tool head path')
    ax[0, 0].set_xlabel('x [m]')
    ax[0, 0].set_ylabel('y [m]')
    ax[0, 0].set_aspect('equal', 'box')
    ax[0, 0].grid(True)
    ax[0, 0].legend()

    ax[0, 1].plot(t, um_l, label='u_m,l (left)')
    ax[0, 1].plot(t, um_r, label='u_m,r (right)')
    ax[0, 1].set_title('Input voltages')
    ax[0, 1].set_xlabel('Time [s]')
    ax[0, 1].set_ylabel('Voltage [V]')
    ax[0, 1].grid(True)
    ax[0, 1].legend()

    ax[1, 0].plot(t, im_l, label='i_m,l (left)')
    ax[1, 0].plot(t, im_r, label='i_m,r (right)')
    ax[1, 0].set_title('Motor currents')
    ax[1, 0].set_xlabel('Time [s]')
    ax[1, 0].set_ylabel('Current [A]')
    ax[1, 0].grid(True)
    ax[1, 0].legend()

    ax[1, 1].plot(t, theta_l, label='theta_l (left)')
    ax[1, 1].plot(t, theta_r, label='theta_r (right)')
    ax[1, 1].set_title('Motor angles')
    ax[1, 1].set_xlabel('Time [s]')
    ax[1, 1].set_ylabel('Angle [rad]')
    ax[1, 1].grid(True)
    ax[1, 1].legend()

    plt.tight_layout()


# ellipse
A = 0.04
B = 0.025
x_ellipse = A * np.cos(t)
y_ellipse = B * np.sin(t)

ux_e, ix_e, vx_e = design_voltage(x_ellipse, cx)
uy_e, iy_e, vy_e = design_voltage(y_ellipse, cy)

x_e, y_e, ixs_e, iys_e = simulate(ux_e, uy_e,
                                  x_ellipse[0], vx_e[0], ix_e[0],
                                  y_ellipse[0], vy_e[0], iy_e[0])

err_e = np.sqrt((x_e - x_ellipse)**2 + (y_e - y_ellipse)**2)
print('Ellipse: max error =', np.max(err_e) * 1000, 'mm')

make_plots('Elliptic trajectory', 'r-', x_ellipse, y_ellipse,
           x_e, y_e, ux_e, uy_e, ixs_e, iys_e)


# heart
x_heart = 0.005 * (16 * np.sin(t)**3)
y_heart = 0.005 * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))

ux_h, ix_h, vx_h = design_voltage(x_heart, cx)
uy_h, iy_h, vy_h = design_voltage(y_heart, cy)

x_h, y_h, ixs_h, iys_h = simulate(ux_h, uy_h,
                                  x_heart[0], vx_h[0], ix_h[0],
                                  y_heart[0], vy_h[0], iy_h[0])

err_h = np.sqrt((x_h - x_heart)**2 + (y_h - y_heart)**2)
print('Heart: max error =', np.max(err_h) * 1000, 'mm')

make_plots('Heart trajectory', 'b-', x_heart, y_heart,
           x_h, y_h, ux_h, uy_h, ixs_h, iys_h)

plt.show()