import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from utils.metrics import compute_accuracy

def evaluate_model(model, dataset, batch_size=16, collate_fn=None, best_model_path=None):
    logger.info("🔍 Starting evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best model checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss()

    running_test_loss = 0.0
    running_test_acc = 0.0
    all_preds = []
    all_labels = []
    num_valid_samples = 0

    with torch.no_grad():
        for questions, lengths, images, answers in tqdm(test_loader, desc="Evaluating"):
            questions, lengths, images, answers = questions.to(device), lengths.to(device), images.to(device), answers.to(device)

            valid_mask = answers != 19
            if valid_mask.sum() == 0:
                continue

            questions = questions[valid_mask]
            lengths = lengths[valid_mask]
            images = images[valid_mask]
            answers = answers[valid_mask]

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(questions, lengths, images)
                loss = criterion(outputs, answers)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(answers.cpu().tolist())

            running_test_loss += loss.item() * answers.size(0)
            running_test_acc += compute_accuracy(outputs, answers) * answers.size(0)
            num_valid_samples += answers.size(0)

    test_loss = running_test_loss / num_valid_samples
    test_acc = running_test_acc / num_valid_samples

    logger.info(f"✅ Number of valid test samples: {num_valid_samples} / {len(dataset)}")
    logger.info(f"📊 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
