import torch
import torchvision
import matplotlib.pyplot as plt



def pred_plot_img(model:torch.nn.Module,image_path:str,class_names:list[str] = None,transform = None,device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Makes a prediction on a target image with a trained model and plots the image and the predcition"""
    #Loading an image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    #Divide the image pixel value by 255 to get them between [0,1]
    target_image  = target_image / 255.

    #Transform the data if neccessary 
    if transform:
        target_image = transform(target_image)

    #The model should be on the same device
    model.to(device)

    #Turn the moodel to eval mode
    model.eval()
    with torch.inference_mode():
        #Add an extra dimension to the image i.e Batch_size since it is an image that is taken from outside the dataset 
        target_image = target_image.unsqueeze(0)

        #Making a prediciton with an extra dimension
        target_img_pred = model(target_image.to(device)) #Make sure the image and the model are on the same device

        #Convert logits into probabilites
        target_img_pred_probs = torch.softmax(target_img_pred,dim = 1)

        #Convert prediction probabilites 
        target_img_pred_labels = torch.argmax(target_img_pred_probs, dim = 1)

        #plot the image alongside the prediction and prediction probabilites
        #Removing the batch dimension and rearrgeing the dim to HWC
        plt.imshow(target_image.squeeze().permute(1,2,0))

        if class_names:
            title = f"Pred : {class_names[target_img_pred_labels.item()]} | Prob : {target_img_pred_probs.max().item():.3f}"
        else:
            title = f"Pred : {target_img_pred_labels} | Prob : {target_img_pred_probs.max().cpu():.3f}"

        plt.title(title)
        plt.axis(False)
