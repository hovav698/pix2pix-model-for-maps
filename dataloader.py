from torch.utils.data import Dataset
from torchvision import transforms
import params
import PIL.Image as Image
import numpy as np

#the original images in the dataset contains one images that include the targets+the inputs. The following function will split it to two images.

def split_image(image_path):
    im = np.array(Image.open(image_path))
    img1 = im[:, :params.IMG_WIDTH, :]
    img2 = im[:, params.IMG_WIDTH:, :]
    return Image.fromarray(img1), Image.fromarray(img2)

#implement the pytorch Dataloader protocol
class DataLoaderInput(Dataset):

    def __init__(self, image_path):
        self.image_path = image_path

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        filename = self.image_path[idx]
        img1, img2 = split_image(filename)
        input_img = self.transform(img1)
        output_img = self.transform(img2)

        return input_img, output_img

    transform = transforms.Compose([
        transforms.Resize((params.new_dim,params.new_dim)),
        # transforms.RandomCrop(IMG_WIDTH),
        transforms.ToTensor()

    ])
