import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm

from models.coattention import CoattentionNet
from data.dataset import VQADataset, collate_fn
from utils.metrics import compute_accuracy

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Coattention VQA model")
    parser.add_argument('--img_h5', type=str, default='data/data_img_att.h5', help='Path to image features h5 file')
    parser.add_argument('--ques_h5', type=str, default='data/cocoqa_data_prepro.h5', help='Path to question h5 file')
    parser.add_argument('--json', type=str, default='data/cocoqa_data_prepro.json', help='Path to question vocab json')
    parser.add_argument('--log_dir', type=str, default='logs_att/', help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='model_att/', help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def train():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    logger.info("🔍 Loading dataset...")
    dataset = VQADataset(args.ques_h5, args.img_h5, args.json, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info("🔧 Building Coattention model...")
    model = CoattentionNet(dataset.vocab_size, num_classes=19, embed_dim=args.embed_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("🚀 Starting training...")
    for epoch in range(args.max_iters):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for questions, lengths, images, answers in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            questions, lengths, images, answers = questions.to(device), lengths.to(device), images.to(device), answers.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(questions, lengths, images)
                loss = criterion(outputs, answers)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_acc += compute_accuracy(outputs, answers)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for questions, lengths, images, answers in val_loader:
                questions, lengths, images, answers = questions.to(device), lengths.to(device), images.to(device), answers.to(device)
                with autocast(device_type='cuda'):
                    outputs = model(questions, lengths, images)
                    loss = criterion(outputs, answers)
                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, answers)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.4f} | Val Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.success(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            logger.warning(f"⏳ Early stopping: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.error("Early stopping triggered!")
                break

    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    writer.close()
    logger.success(f"Training complete. Final model saved to {final_path}")

if __name__ == '__main__':
    train()
