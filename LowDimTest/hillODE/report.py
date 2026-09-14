# This python code contains the plotting functions and utils for the report

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..'))) 
from data import SINDy_data
from data import data
from data import equations

def evaluate_traj(model, ode, freq, n_sample, init_high, init_low,
                  dim_x=1, dim_k=1, title=None, T_aux=100):

    # =======================
    # STYLE
    # =======================
    plt.rcdefaults()

    np.random.seed(666)
    dt = 1 / freq
    t = np.arange(0, T_aux, dt)
    T_plot = len(t)

    # =======================
    # TRUE TRAJECTORIES
    # =======================
    dg_true = data.DataGenerator(
        ode, T_aux, freq, n_sample,
        noise_sigma=0.,
        init_high=init_high,
        init_low=init_low
    )
    xt_true = dg_true.xt

    # =======================
    # ESTIMATED TRAJECTORIES
    # =======================
    pred_0_list = []

    for i in range(n_sample):
        correct_param = xt_true[0, i, -dim_k:]
        test_params = np.tile(correct_param, (T_plot, 1))

        if dim_k != 0:
            pred_0 = model.simulate(
                xt_true[0, i, :-dim_k],
                t=t,
                u=test_params
            )
        else:
            pred_0 = model.simulate(
                xt_true[0, i],
                t=t
            )

        pred_0_list.append(pred_0)

    # =======================
    # RESHAPE
    # =======================
    xt_true = xt_true[:len(pred_0), :, :]
    if dim_k != 0:
        xt_true = xt_true[:, :, :-dim_k]

    xt_true = xt_true.squeeze()
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2)).squeeze()

    time_vector = t[:len(pred_0_list)]

    # =======================
    # PLOTTING
    # =======================
    if n_sample == 1:

        # ---------- dim_x = 1 ----------
        if dim_x == 1:

            fig, ax = plt.subplots(1, 1, figsize=(5, 4))

            ax.plot(time_vector, pred_0_list,
                    lw=1.5, color='blue', alpha=0.8,
                    label='Estimated trajectory')

            ax.plot(time_vector, xt_true,
                    lw=1.5, color='red', linestyle='--',
                    label='True trajectory')

            # Start / End
            ax.scatter(time_vector[0], xt_true[0],
                       s=30, color='green', zorder=3,
                       label='Start')

            ax.scatter(time_vector[-1], xt_true[-1],
                       s=30, color='red', zorder=3,
                       label='End')

            ax.set_xlabel("t")
            ax.set_ylabel("$x(t)$")

            ax.grid(True, alpha=0.3)

            if title:
                ax.set_title(title)

            ax.legend(fontsize=9, frameon=False, loc='lower right')
            plt.tight_layout()
            plt.show()

        # ---------- dim_x = 2 ----------
        elif dim_x == 2:

            fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

            # x(t)
            axes[0].plot(time_vector, pred_0_list[:, 0],
                         lw=1.5, color='blue', alpha=0.8,
                         label='Estimated $x(t)$')

            axes[0].plot(time_vector, xt_true[:, 0],
                         lw=1.5, color='red', linestyle='--',
                         label='True $x(t)$')

            axes[0].scatter(time_vector[0], xt_true[0, 0],
                            s=30, color='green', zorder=3,
                            label='Start')

            axes[0].scatter(time_vector[-1], xt_true[-1, 0],
                            s=30, color='red', zorder=3,
                            label='End')

            axes[0].set_ylabel("$x(t)$")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=9, frameon=False)

            # y(t)
            axes[1].plot(time_vector, pred_0_list[:, 1],
                         lw=1.5, color='blue', alpha=0.8,
                         label='Estimated $y(t)$')

            axes[1].plot(time_vector, xt_true[:, 1],
                         lw=1.5, color='red', linestyle='--',
                         label='True $y(t)$')

            axes[1].scatter(time_vector[0], xt_true[0, 1],
                            s=30, color='green', zorder=3,
                            label='Start')

            axes[1].scatter(time_vector[-1], xt_true[-1, 1],
                            s=30, color='red', zorder=3,
                            label='End')

            axes[1].set_xlabel("t")
            axes[1].set_ylabel("$y(t)$")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=9, frameon=False)

            if title:
                fig.suptitle(title)

            plt.tight_layout()
            plt.show()

        # ---------- dim_x >= 3 ----------
        else:

            fig, ax = plt.subplots(
                figsize=(6, 6),
                subplot_kw={'projection': '3d'}
            )

            ax.plot(pred_0_list[:, 0],
                    pred_0_list[:, 1],
                    pred_0_list[:, 2],
                    lw=1.5, color='blue', alpha=0.8,
                    label='Estimated trajectory')

            ax.plot(xt_true[:, 0],
                    xt_true[:, 1],
                    xt_true[:, 2],
                    lw=1.5, color='red', linestyle='--',
                    label='True trajectory')

            ax.scatter(xt_true[0, 0], xt_true[0, 1], xt_true[0, 2],
                       s=30, color='green', label='Start')

            ax.scatter(xt_true[-1, 0], xt_true[-1, 1], xt_true[-1, 2],
                       s=30, color='red', label='End')

            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")

            if title:
                ax.set_title(title)

            ax.legend(fontsize=9, frameon=False)
            plt.tight_layout()
            plt.show()


from scipy.integrate import solve_ivp
import numpy as np

class SRModel:
    def __init__(self, f):
        self.f = f  # la funzione dinamica, es. f = lambda z,t: [z[1], -np.sin(z[0])]
    
    def simulate(self, x0, t, u=None):
        # u ignorato, lo metti solo per compatibilità con SINDy
        sol = solve_ivp(lambda t, z: self.f(z, t), [t[0], t[-1]], x0, t_eval=t, vectorized=False)
        return sol.y.T  # restituisce shape (len(t), dim_x)




#### Parametrized code code

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np

class SRModel_param:
    def __init__(self, f):
        self.f = f
    
    def simulate(self, x0, t, u):
        # se u è un array (n_samples, 1)
        u_fun = interp1d(t, u, axis=0, kind="cubic", fill_value="extrapolate")

        def rhs(ti, z):
            k = float(u_fun(ti))   # interpolazione CORRETTA
            return self.f(z, ti, k)

        sol = solve_ivp(rhs, [t[0], t[-1]], x0, t_eval=t, method="LSODA")
        return sol.y.T


