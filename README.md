# AIPND
** This project was made for the Udacity Nano degree AI Programming with Python **
Flowers image classification project for classifying flower types

# Requirements:
This program is written in python so to run it you need to have these packages installed on your device
1. **torch**
2. **torchvision**
3. **Numpy**
4. **Matplotlib**
5. **Seaborn**
6. **Json**
7. **Time**
8. **PIL**
# Steps:
1. ** NN_model_init.py and train.py and predict.py need to be on the same folder!!! **
2. ** The Neural network is not trained so you have to run train.py to train it, i dont recommend running on your own device, run it on AWS or google colab.
3. Type in your cmd ``` Python train.py /users/Floder_Images_used_for_training_the_Neural_network --save_dir /users/your_own_folder --arch vgg19 --learning_rate 0.001 --hidden_units 512 --epochs --gpu ```
**Everything that starts with -- is optional you dont need to change it! but its ok if you do** the first thing you have to type after typing ``` train.py ``` is the location of the images that you will use to train the model, --save_dir is used for saving the trained model ``` --arch ``` is the archtype used to train
you can choose from ``` vgg19 densenet121 densenet201 ``` then the ``` --learning_rate ``` is set to 0.001 for default no need to change it ``` --hidden_units ``` how many hidden units will be used ```--epochs ``` the number of epochs the model is going to train ``` --gpu ``` turns gpu training on so the model trains on the gpu

4. After training the trained model will be saved depending on where you specified the ``` --save_dir ``` so be careful
5. Go to predict.py to test your model on an image that you have and you want to check what type of flower it is ``` Python predict.py /users/path_to_image /users/path_to_model --gpu ```
6. Have fun with the model if you want to change anything refer to ** Step 3 ** no need to change the source code
