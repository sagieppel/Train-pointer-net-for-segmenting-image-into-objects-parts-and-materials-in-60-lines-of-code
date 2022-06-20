import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
#-------Main train parameters------------------------------------------
Learning_Rate=1e-5
width=height=800 # image width and height
batchSize=4 
TrainFolder="/media/breakeroftime/2T/Data_zoo/LabPicsV1.2/Simple/Train/"
ListImages=os.listdir(os.path.join(TrainFolder, "Image")) # Create list of images
#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # preprocces image
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()]) # preprocces mask
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(): # load random image and  the corresponding annotation
    idx=np.random.randint(0,len(ListImages)) # Select random image
    img=cv2.imread(os.path.join(TrainFolder, "Image", ListImages[idx]))[:,:,0:3] # load image
    insMap =  cv2.imread(os.path.join(TrainFolder, "Instance/", ListImages[idx].replace("jpg","png"))) # load segmentation map
    insMap[insMap == 254] = 0 # 254 stand for unsegmented region, set it to zero
    img = transformImg(img) # transform img
    insMap = transformAnn(insMap) # transform mask
    vesMap = insMap[2,:, :] # in the LabPics dataset the first 3 channel in the instance map is the vessel instances map
    if vesMap.max()<=0: return ReadRandomImage() # if there is no vessels in the image load another image
    matMap = insMap[0, :, :] # in the LabPics dataset the first 3 channel in the instance map is the material  instances map
    while(True):
        x = np.random.randint(vesMap.shape[1]) # pick random point
        y = np.random.randint(vesMap.shape[0])
        if vesMap[y,x]>0: # check if the point is inside a vessel
           vesMask = (vesMap  == vesMap[y,x]).type(torch.float32) # create vessel mask
           matMask = (vesMask*(matMap == matMap[y, x])).type(torch.float32) # create material map (note the material map is limited to the area of the vessel)
           pointerMask = torch.zeros_like(matMask) # create pointer mask
           pointerMask[y - 4:y + 4, x - 4:x + 4] = 1 # the pointer will not be a single pixel but a sqr of 8x8
           break
    img=torch.cat([img,vesMask.unsqueeze(0),pointerMask.unsqueeze(0)],0) # concat the img pointer mask and vessel mask (ROI) into sing iput
    return img,matMask
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,5,height,width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage()
    return images, ann
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes (0/1)
weights=Net.backbone.conv1.weight.data
Net.backbone.conv1=torch.nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change input from 3 layers (rgb) to 5 layers (r,g,b,mask,pointer)
Net.backbone.conv1.weight.data[:,:3,:,:]=weights
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
#----------------Train--------------------------------------------------------------------------
for itr in range(0,100000): # Training loop
   images,ann=LoadBatch() # Load taining batch
   images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
   ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
   Pred=Net(images)['out']# make prediction
   Net.zero_grad()
   criterion = torch.nn.CrossEntropyLoss() # Set loss function
   pr = (Pred[0][1] > Pred[0][0]).cpu().detach().numpy() # convert probability mask to material mask
   Loss=criterion(Pred,ann.long()) # Calculate cross entropy loss
   Loss.backward() # Backpropogate loss
   optimizer.step() # Apply gradient descent change to weight
   print(itr,") Loss=",Loss.data.cpu().numpy()) # display loss
   if itr % 1000 == 0: #Save model weight once every 1k steps permenant file
        print("Saving Model" +str(itr) + "-.torch")
        torch.save(Net.state_dict(),   str(itr) + ".torch")