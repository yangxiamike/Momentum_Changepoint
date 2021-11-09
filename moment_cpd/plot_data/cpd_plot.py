from matplotlib import pyplot as plt
import numpy as np
import os


def plot_cpd_sample(x, y, k,
                   figsize = (15, 8), is_save = False,
                   save_path = ''):
    x_min, x_max = min(x), max(x)
    xx = np.linspace(x_min, x_max, 100000).reshape(100000, 1)
    ## predict mean and variance of latent GP at test points
    mean, var = k.predict_f(xx)
    ## generate 10 samples from posterior
    samples = k.predict_f_samples(xx, 10)
    ## plot
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    ax.scatter(x, y)
    ax.plot(xx, mean, "C0", lw = 2)
    ax.fill_between(
        xx[:, 0],
        mean[:, 0] - 2 * np.sqrt(var[:, 0]),
        mean[:, 0] + 2 * np.sqrt(var[:, 0]),
        color = "blue",
        alpha = 0.8)
    ax.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth = 0.5)

    if is_save:
        plt.savefig(save_path)

def plot_cpd_date(dates, prices, locations, 
                          is_save = False, is_verbose = True,
                          title = 'pic', save_dir = 'data'):
    """
    dates: (list[datetime64]), dates for prices
    prices: (list[float]), prices for underlying asset
    locations: (list[int]), location indices for changepoints
    """
    fig, ax = plt.subplots(1, 1, figsize = (20, 8))
    # plot price curve for underlying asset
    ax.plot(dates, prices, 'k')
    # plot changepoints vertical lines
    p_min, p_max = min(prices), max(prices)
    y_min = 0.9 * p_min
    y_max = 1.1 * p_max
    for loca in locations:
        xs = [dates[loca], dates[loca]]
        ys = [y_min, y_max]
        ax.plot(xs, ys, 'k-')
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_ylabel('Price')
    if is_save:
        plt.savefig(os.path.join(save_dir, f'{title}.png'))
    if is_verbose:
        plt.show() 