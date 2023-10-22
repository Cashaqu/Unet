from data_preprocess.bubbles_generator import GrayScaleDataProvider
#from data_preprocess.dataset_utils import transformation
import torch
from tqdm import tqdm
import matplotlib
import torchvision.transforms as transforms
device='cuda' if torch.cuda.is_available() else 'cpu'
from PIL import Image
import glob

def UNet_eval(model, test_size=10):
    test_size=test_size
    generator = GrayScaleDataProvider(nx=572, ny=572, cnt=20)
    X_test, y_test = generator(test_size)
    print('Generate test data...\n')
    for i in tqdm(range(test_size)):
        matplotlib.image.imsave('./data/X_test/' + "{0:0{k}d}".format(i, k=len(str(test_size))) + '.png',
                               X_test[i, ..., 0])
        matplotlib.image.imsave('./data/y_test/' + "{0:0{k}d}".format(i, k=len(str(test_size))) + '.png',
                               y_test[i, ..., 1])

    transformation = transforms.Compose([transforms.Grayscale(1),
                                        #transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
    images = glob.glob('./data/X_test/*.png')
    print('Generate prediction...\n')
    for image in tqdm(images):
        img = Image.open(image)
        img_trans = transformation(img).unsqueeze(0).to(device)
        y_pred = model(img_trans)
        matplotlib.image.imsave('./data/eval/' + image[-6:-4] + '.png',
                                y_pred[0, 0,...].cpu().detach().numpy())