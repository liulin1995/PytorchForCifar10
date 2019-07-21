from torchvision import transforms
import PIL
import matplotlib.pyplot as plt

mean_imagenet = [0.485, 0.456, 0.406]
std_imagenet = [0.229, 0.224, 0.225]
mean_i = [0.485, 0.456, 0.406]
std_i = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_imagenet, std_imagenet)])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_imagenet, std_imagenet),
])

'''
def show_some_image(rows=3, cols=3):
    plt.interactive(False)
    (inputs, label) = next(iter(vis_loader))
    if inputs.size()[0] < rows*cols:
        print('too many images')
    plt.figure(rows*cols)
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1).set_title(str(label[i].item()))
        img = inputs[i].numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
tfs = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.ToTensor()
])
or 
tfs = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.ToTensor()
])
'''