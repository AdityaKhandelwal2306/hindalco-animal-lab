"""Train the classifier using torchvision ImageFolder."""
import argparse
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model_utils import build_model, get_transform, log_metric, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--curve_out", default="training_curve.png")
    args = parser.parse_args()
    dataset = ImageFolder(args.train_data, transform=get_transform())
    model = build_model(len(dataset.classes), pretrained=not args.from_scratch, freeze=not args.from_scratch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not args.from_scratch:
        model.features.eval()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        if not args.from_scratch:
            model.features.eval()
        total_loss = correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (output.argmax(1) == labels).sum().item(); total += labels.size(0)
        accuracy = correct / total
        average_loss = total_loss / total
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {average_loss:.4f} - train accuracy: {accuracy:.4f}")
        log_metric("train_loss", average_loss, epoch + 1)
        log_metric("train_accuracy", accuracy, epoch + 1)
    save_model(model.cpu(), dataset.classes, args.model_dir)
    save_loss_chart(epoch_losses, args.curve_out)
    print(f"Saved model to {Path(args.model_dir).resolve()}")


def save_loss_chart(losses, output_path):
    """Save a small loss-versus-epoch chart without adding a plotting dependency."""
    width, height = 640, 420
    left, top, right, bottom = 70, 35, 25, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(losses) if losses else 1.0
    maximum = maximum or 1.0
    points = []
    for index, loss in enumerate(losses):
        x = left if len(losses) == 1 else left + index * plot_width / (len(losses) - 1)
        y = top + plot_height * (1 - loss / maximum)
        points.append((x, y))
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    if len(points) > 1:
        draw.line(points, fill="blue", width=3)
    for point in points:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="blue")
    draw.text((width // 2 - 45, height - 35), "Epoch", fill="black")
    draw.text((8, top), "Loss", fill="black")
    draw.text((width // 2 - 55, 8), "Training Loss", fill="black")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


if __name__ == "__main__":
    main()