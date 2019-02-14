# Example on how to generate partitions for 
# a one-mouse-line-out cross-validation.

import utils
import numpy as np

# load and mask
data = utils.load_all()
mask = utils.generate_mask()
data = data[mask]

# time since infection label, for example.
labels = utils.get_labels('t_post_infection')

# get the line information. Note this 
# label is only for generating the cross-validation, 
# not the labels.
lines = utils.get_labels('line')

partitions = utils.generate_partitions(lines)

# Look at an example to verify.

print('Training lines:')
print( np.unique( lines[partitions[4][0]] ) )

print('Testing line:')
print( np.unique( lines[partitions[4][1]] ) )

data_train = data[ partitions[4][0] ]
data_test = data[ partitions[4][1] ]

labels_train = labels[ partitions[4][0] ]
labels_test = labels[ partitions[4][1] ]

# etc...
