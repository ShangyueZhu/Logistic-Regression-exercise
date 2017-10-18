import scipy
from scipy import ndimage
import cv2
from loaddata import *
import matplotlib.pyplot as plt
from lr_tool import *
from dnn_app_utils_v2 import *

train_set_x, train_set_y, test_set_x, test_set_y = loaddata()

train_set_x = train_set_x.reshape(train_set_x.shape[0],-1).T
train_set_x_flatten = train_set_x / 255

test_set_x = test_set_x.reshape(test_set_x.shape[0],-1).T
test_set_x_flatten = test_set_x / 255

# print(train_set_x.shape)
# n_x = train_set_x.shape[0]   # num_px * num_px * 3
# n_h = 3
# n_y = 1
# layers_dims = (n_x, n_h, n_y)

# parameters = two_layer_model(train_set_x, train_set_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


learning_rates = [0.01, 0.001, 0.0001]

models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
