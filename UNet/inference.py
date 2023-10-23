from data_preprocess.bubbles_generator import GrayScaleDataProvider
#from data_preprocess.dataset_utils import transformation
import torch
from tqdm import tqdm
import matplotlib
import torchvision.transforms as transforms
device='cuda' if torch.cuda.is_available() else 'cpu'
from PIL import Image
import glob


def UNet_eval(path_to_model, test_size):
    model = torch.load(path_to_model)
    generator = GrayScaleDataProvider(nx=572, ny=572, cnt=20)
    X_test, y_test = generator(test_size)
    print('\nGenerate test data... -> ./data/X_test/ and ./data/y_test/')
    for i in tqdm(range(test_size)):
        matplotlib.image.imsave('./data/X_test/' + "{0:0{k}d}".format(i, k=len(str(test_size))) + '.png',
                               X_test[i, ..., 0])
        matplotlib.image.imsave('./data/y_test/' + "{0:0{k}d}".format(i, k=len(str(test_size))) + '.png',
                               y_test[i, ..., 1])

    transformation = transforms.Compose([transforms.Grayscale(1),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
    images = glob.glob('./data/X_test/*.png')
    print('\nGenerate prediction... -> ./data/y_pred/')
    for image in tqdm(images):
        img = Image.open(image)
        img_trans = transformation(img).unsqueeze(0).to(device)
        y_pred = model(img_trans)
        matplotlib.image.imsave('./data/y_pred/' + image[-6:-4] + '.png',
                                y_pred[0, 0,...].cpu().detach().numpy())

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description="Get some hyperparameters.")
#
#     # Get an arg for num_epochs
#     parser.add_argument("--path_to_model",
#                         default='/home/kda/PycharmProjects/Unet/models/best_model.pt',
#                         type=str,
#                         help="path to model")
#
#     # Get an arg for vec_size
#     parser.add_argument("--test_size",
#                         default=10,
#                         type=int,
#                         help="size of test dataset")
#
#     args = parser.parse_args()
#
#     PATH_TO_MODEL = args.path_to_model
#     TEST_SIZE = args.test_size
#
#     UNet_eval(PATH_TO_MODEL, TEST_SIZE)