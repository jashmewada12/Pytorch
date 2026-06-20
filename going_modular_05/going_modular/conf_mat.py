
import argparse
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
import data_setup,model_builder
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Plots a confusion matrix for a saved model")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model .pth file")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing data")
parser.add_argument("--hidden_units", type=int, default=20, help="Hidden units used when training the model")
args = parser.parse_args()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=20),
    transforms.ToTensor()
    ])

    train_dir = "../data/pizza_steak_sushi/train"
    test_dir = "../data/pizza_steak_sushi/test"

    print("[INFO] Loading testing data..")

    _,test_dataloader,class_names,_ = data_setup.create_dataloaders(
        train_dir = train_dir,
        test_dir = test_dir,
        train_transform = TRAIN_TRANSFORM,
        test_transform = TEST_TRANSFORM,
        batch_size = args.batch_size
    )


    print(f"[INFO] Instantiating TinyVGG model with {args.hidden_units} hidden units...")
    model = model_builder.TinyVGG_v1(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    )

    print(f"[INFO] Loading weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    
    y_preds = []
    y_true = []
    print("[INFO] Running inference on test dataset...")
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
            
            # Move to CPU immediately for matplotlib/mlxtend compatibility
            y_preds.append(y_pred.cpu())
            y_true.append(y.cpu())

    y_preds_tensor = torch.cat(y_preds)
    y_true_tensor = torch.cat(y_true)

    print("[INFO] Generating Confusion Matrix plot...")
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
    confmat_tensor = confmat(preds=y_preds_tensor, target=y_true_tensor)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7),
        cmap="Blues"
    )
    plt.title(f"Confusion Matrix: {model.__class__.__name__}", fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
