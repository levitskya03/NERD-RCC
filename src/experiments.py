import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from models import Generator, Discriminator  # Assuming models.py has these classes
import wandb
import os
import argparse

# Initialize Weights and Biases (wandb)
wandb.init(project="NERD-RCC-Experiments")


# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="GAN Additional Tasks")
    parser.add_argument("--task", type=str, required=True, choices=["augment", "extract_features", "fine_tune", "style_transfer"], help="Task to perform")
    parser.add_argument("--model", type=str, default="MNIST", choices=["MNIST", "CIFAR", "SVHN"], help="Pre-trained GAN model to use")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate (for augment task)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension size")
    parser.add_argument("--save_path", type=str, default="output", help="Path to save outputs")
    parser.add_argument("--content_image", type=str, help="Path to content image for style transfer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--dataloader_path", type=str, help="Path to data loader script for fine-tune or extract-features tasks")
    return parser.parse_args()

# Paths to your pre-trained GAN models
PRETRAINED_GAN_MODELS = {
    "MNIST": "trained_models/trained_gan/wgan_gp_MNIST.ckpt",
    "CIFAR": "trained_models/trained_gan/wgan_gp_CIFAR.ckpt",
    "SVHN": "trained_models/trained_gan/wgan_gp_SVHN.ckpt",
}

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the GAN generator
def load_generator(model_path, img_size=(32, 32, 1), latent_dim=128, dnn_size=64):
    generator = Generator(img_size, latent_dim, dnn_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the checkpoint contains nested keys
    if 'generator' in checkpoint:
        generator_weights = checkpoint['generator']
    else:
        generator_weights = checkpoint
    
    # Load the state dict with flexibility
    generator.load_state_dict(generator_weights, strict=False)
    generator.eval()
    return generator


# Load the GAN discriminator
def load_discriminator(model_path, img_size=(32, 32, 3), dnn_size=64):
    discriminator = Discriminator(img_size, dnn_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the checkpoint contains nested keys
    if 'discriminator' in checkpoint:
        discriminator_weights = checkpoint['discriminator']
    else:
        discriminator_weights = checkpoint
    
    # Load the state dict with flexibility
    discriminator.load_state_dict(discriminator_weights, strict=False)
    discriminator.eval()
    return discriminator


# 1. Data Augmentation
def generate_augmented_data(generator, num_samples=100, latent_dim=128, save_path="augmented_data"):
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        generated_images = generator(z)
        for i, img in enumerate(generated_images):
            save_image(img.cpu(), f"{save_path}/generated_{i}.png")
            wandb.log({"generated_image": [wandb.Image(img, caption=f"Generated Image {i}")]})
    print(f"Augmented data saved to {save_path}")

# 2. Representation Learning
def extract_features(discriminator, dataloader):
    features = []
    labels = []
    discriminator.to(device)
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            feats = discriminator(imgs).detach().cpu().numpy()
            features.append(feats)
            labels.append(lbls.numpy())
            wandb.log({"batch_features": feats.tolist()})
    print("Feature extraction completed.")
    return np.concatenate(features), np.concatenate(labels)

# 3. Transfer Learning
def fine_tune_discriminator(discriminator, dataloader, num_epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    discriminator.to(device)
    discriminator.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = discriminator(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss / len(dataloader)})
        print(f"Epoch {epoch+1}/{num_epochs} completed with average loss {epoch_loss / len(dataloader)}")

# 4. Creative Applications (Style Transfer)
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

def apply_style_transfer(generator, content_image_path, latent_dim=128, save_path="stylized_image.png"):
    content_image = Image.open(content_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    content_tensor = transform(content_image).unsqueeze(0).to(device)

    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        styled_image = generator(z)

    styled_image = styled_image.squeeze(0).cpu()
    styled_image_pil = to_pil_image(styled_image)
    styled_image_pil.save(save_path)
    wandb.log({"styled_image": wandb.Image(styled_image_pil, caption="Stylized Image")})
    print(f"Stylized image saved to {save_path}")

# Main execution
# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    img_size = None

    if args.model == "SVHN":
        img_size = (32, 32, 3)  # SVHN has 3-channel RGB images
    elif args.model == "MNIST":
        img_size = (32, 32, 1)  # MNIST has 1-channel grayscale images
    else:
        raise ValueError(f"Unsupported model {args.model}")

    # Dynamically load the dataloader module if required
    if args.task in ["extract_features", "fine_tune"]:
        if not args.dataloader_path:
            raise ValueError("Dataloader script path is required for extract_features or fine_tune tasks")
        
        # Dynamically import the dataloader module
        import importlib.util
        spec = importlib.util.spec_from_file_location("dataloaders", args.dataloader_path)
        dataloader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataloader_module)
        
        # Choose the appropriate datamodule
        if args.model == "MNIST":
            datamodule = dataloader_module.MNISTDataModule(batch_size=32)
        elif args.model == "SVHN":
            datamodule = dataloader_module.SVHNDataModule(batch_size=32)
        elif args.model == "FMNIST":
            datamodule = dataloader_module.FMNISTDataModule(batch_size=32)
        elif args.model == "Gaussian":
            datamodule = dataloader_module.GaussianDataModule(batch_size=32, m=1024, r=0.025)
        elif args.model == "Sawbridge":
            datamodule = dataloader_module.Sawbridge(batch_size=32)
        else:
            raise ValueError(f"Unsupported model {args.model}")

        # Fetch the appropriate dataloader
        dataloader = datamodule.train_dataloader()

    # Perform task-specific logic
    if args.task == "augment":
        generator = load_generator(PRETRAINED_GAN_MODELS[args.model], img_size=(32, 32, 1), latent_dim=args.latent_dim)
        generate_augmented_data(generator, num_samples=args.num_samples, latent_dim=args.latent_dim, save_path=args.save_path)
    elif args.task == "style_transfer":
        generator = load_generator(PRETRAINED_GAN_MODELS[args.model], img_size=(32, 32, 1), latent_dim=args.latent_dim)
        if not args.content_image:
            raise ValueError("Content image path is required for style transfer task")
        apply_style_transfer(generator, args.content_image, latent_dim=args.latent_dim, save_path=args.save_path)
    elif args.task == "extract_features":

        discriminator = load_discriminator(PRETRAINED_GAN_MODELS[args.model], img_size=img_size)
        features, labels = extract_features(discriminator, dataloader)
    elif args.task == "fine_tune":
        discriminator = load_discriminator(PRETRAINED_GAN_MODELS[args.model], img_size=img_size)
        fine_tune_discriminator(discriminator, dataloader, num_epochs=args.epochs, lr=args.lr)
