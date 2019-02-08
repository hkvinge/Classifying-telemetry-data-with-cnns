# example usage of utils for some quick timeseries
# visualization of a bit of the data.

import utils
import numpy as np
from matplotlib import pyplot

# utils.get_labels() **must** be called after load_all().
data = utils.load_all()
labels = utils.get_labels('t_post_infection')

mouse_ids = utils.get_labels('mouse_id')

# Look at the first thirteen chunks; corresponds to all data from the first mouse.
fig,ax = pyplot.subplots(13,1, sharex=True, sharey=True, figsize=(8,8))

# colors for plots; labels==0 is one end of the spectrum; labels>=5 days is the other end.
colors = pyplot.cm.viridis( np.minimum(1., labels/(5*1440.)) )

for i in range(13):
    ax[i].plot(data[i], lw=2, c=colors[i])

    # prettying the plots a little
    ax[i].set_xticks(np.arange(0,1440+1,1440//4))
    ax[i].xaxis.grid()
#

fig.suptitle('One-day chunks for mouse %s'%mouse_ids[0], fontsize=18)
fig.subplots_adjust(top=0.9)
fig.show()
