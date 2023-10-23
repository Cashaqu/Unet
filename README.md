# Unet
Generating bubbles and trying to segment them.

## How to use
### Inference
My trained model: https://drive.google.com/file/d/1T3r-n4Inx_8WbG-XC6N6ZO8WE2UP2RuH/view?usp=sharing
Put the trained model to ```./models``` and use:
```sh
python main.py --mode=inference --model_name=best_model.pt --test_size=10
```
After execution you will receive 10 generated masks in the folder ```./data/y_pred```.
Also you can view test pictures and masks in ```./data/X_test``` and ```./data/y_test```
### Train
For train Unet model use:
```sh
$ python main.py --mode=train 
```
You also can put some parameters for example:

```sh
$ python main.py --mode=train --dataset_size=100 --coef_split=0.8 --num_epoch=20 --batch_size=4
```
## Example inference
X_test(572x572):

![X_test](https://github.com/Cashaqu/Unet/blob/master/example_inference/02_X_test.png)

y_test(572x572):

![y_test](https://github.com/Cashaqu/Unet/blob/master/example_inference/02_y_test.png)

y_pred(224x224):

![y_pred](https://github.com/Cashaqu/Unet/blob/master/example_inference/02_y_pred.png)
