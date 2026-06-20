import torch
import torchvision
import matplotlib.pyplot as plt
import random
import data_setup,model_builder




def pred_plot(model : torch.nn.Module ,test_data,class_names,device):
    random.seed(42)
    test_samples = []
    test_labels = []
    for sample , label in  random.sample(list(test_data), k =9):
        test_samples.append(sample)
        test_labels.append(label)
    model.eval()
    model.to(device)
    pred_probs = []
    with torch.inference_mode():
        for sample in test_samples:
            # Prepare the sample
            sample_unsqueezed = torch.unsqueeze(sample, dim=0).to(device)
            # Forward pass
            pred_logit = model(sample_unsqueezed)
            # Get prediction probabilities
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            # Get pred prob off the GPU
            pred_probs.append(pred_prob.cpu())
    pred_probs = torch.stack(pred_probs)
    pred_classes = pred_probs.argmax(dim=1)
    
    # 3. Plot the images and predictions
    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        # Create subplot
        plt.subplot(nrows, ncols, i+1)
        
        # Plot the target images (Rearranging dimensions for Matplotlib to read color)
        plt.imshow(sample.permute(1, 2, 0))
        
        # Find the prediction in text form
        pred_label = class_names[pred_classes[i]]
        # Get the truth label in text form
        truth_label = class_names[test_labels[i]]
        
        # Create a title for the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        
        # Check for equality between pred and truth and change color of title text
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c='g')    
        else:
            plt.title(title_text, fontsize=10, c='r')
            
        plt.axis(False)
        
    plt.tight_layout()
    plt.show()
     
