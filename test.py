import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import numpy as np
#---------Input parameters--------------------------------------------------------------------
modelPath = "46000.torch"  # Path to trained model file
imagePath = "NileRed_How to Clean Sodium Metal-screenshot (4).jpg"  # Test image
maskPath = "NileRed_How to Clean Sodium Metal-screenshot (4).png" # Vessel mask for test image
height=width=800 # target image size, must be the same as the size used in training
#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # transform image
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()]) # transform mask
#---------------------Read image ---------------------------------------------------------
def ReadImageAndMask(imagePath,maskPath): # load image and mask
    imgOrigin2=cv2.imread(imagePath)[:,:,0:3] # load image
    vesMask=  cv2.imread(maskPath,0) # load mask
    img = transformImg(imgOrigin2)
    vesMask = transformAnn(vesMask)
    vesMask=(vesMask>0).type(torch.float32) # convert mask to 0-1 format (from 0-255)
    return img,vesMask,imgOrigin2
#---------------Create pointer mask-----------------------------------------
def CreatePointerMask(mask):
    pointerMask = torch.zeros_like(mask)
    xy=torch.where(mask) # list all xy coordinates within the mask
    xyInd=np.random.randint(0,len(xy[0])) # select random point
    y=xy[0][xyInd]
    x=xy[1][xyInd]
    pointerMask[y - 4:y + 4, x - 4:x + 4] = 1 # mark random point on the pointer mask
    return pointerMask

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
Net.backbone.conv1=torch.nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change input from 3 layers (rgb) to 5 layers (r,g,b,mask,pointer)
Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 2 classes
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set to evaluation mode
img, vesMask, imgOrigin = ReadImageAndMask(imagePath,maskPath)  # read image and mask
plt.imshow(imgOrigin[:,:,::-1])  # Show image
plt.show()


segMap=torch.zeros_like(vesMask)[0] # segmentation map were we stich every new segment we discover
unsegmentedMask=vesMask[0]+0 # map of the region that are not yet segmented
for i in range(1,100):
    print(i)
    pointerMask=CreatePointerMask(unsegmentedMask)  # select point in the unsegmented part of the image
    input=torch.cat([img, vesMask, pointerMask.unsqueeze(0)], 0) # concat the img vessel mask and pointer for the net inpuy
    input = torch.autograd.Variable(input, requires_grad=False).to(device).unsqueeze(0) # set input to gpu/cpu

    with torch.no_grad():

       Prd = Net(input)['out']  # Run net


    seg = (Prd[0][1]>Prd[0][0]) # Get  prediction mask
    #if ((Prd[0][1] > Prd[0][0]).cpu()*(1-unsegmentedMask)).sum()/(Prd[0][1] > Prd[0][0]).cpu().sum()>0.5: continue
    segMap[seg]=i # add predicted mask to the segmentation mask
    unsegmentedMask[seg]=0 # remove predicted mask from the region in the image that are not segmented

    if unsegmentedMask.sum() / vesMask.sum() < 0.05: break # if 95% of the image is segmented finish the proccess

#---------------------------display prediction output
segMap2=segMap.cpu().detach().numpy()
segMap2=cv2.resize(segMap2,(imgOrigin.shape[1],imgOrigin.shape[0]),cv2.INTER_NEAREST) # resize back to original image size
segImage=imgOrigin.copy()
for i in np.unique(segMap2): # mark segmented regions on image
    if i==0: continue
    segImage[:, :, 0][segMap2 == i] = np.random.randint(0, 255) # mark segment region with random color
    segImage[:, :, 1][segMap2 == i] = np.random.randint(0, 255)  # mark segment region with random color
    segImage[:, :, 2][segMap2 == i] = np.random.randint(0, 255) # mark segment region with random color
outImg=np.hstack([segImage,imgOrigin])[:,:,::-1]
plt.imshow(outImg)
plt.show()


