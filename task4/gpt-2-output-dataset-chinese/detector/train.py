import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import os

from .dataset import Corpus, EncodedDataset


def train(model, device, train_loader, optimizer):
    model.train()
    total_loss, total_accuracy, total_count = 0, 0, 0
    for idx, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader)):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()
        total_count += 1

    avg_loss = total_loss / total_count
    avg_accuracy = total_accuracy / total_count
    return avg_loss, avg_accuracy


def validate(model, device, validation_loader):
    model.eval()
    total_loss, total_accuracy, total_count = 0, 0, 0
    with torch.no_grad():
        for idx, (input_ids, attention_mask, labels) in enumerate(tqdm(validation_loader)):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == labels).float().mean()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_count += 1

    avg_loss = total_loss / total_count
    avg_accuracy = total_accuracy / total_count
    return avg_loss, avg_accuracy

def save_model(model, optimizer, epoch, loss, accuracy, save_path):
    """保存模型和优化器的状态字典以及其他训练参数"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    }, save_path)

def main(args):
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(args.bert_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # 加载真实和生成的文本数据集
    real_corpus = Corpus(args.real_dataset, args.data_dir, max_texts=50)
    fake_corpus = Corpus(args.fake_dataset, args.data_dir, max_texts=50)

    # 创建训练数据集和验证数据集
    train_dataset = EncodedDataset(real_corpus.train, fake_corpus.train, tokenizer, max_sequence_length=512)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    validation_dataset = EncodedDataset(real_corpus.valid, fake_corpus.valid, tokenizer, max_sequence_length=512)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    # 设置优化器
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    best_validation_accuracy = 0

    # 进行训练和验证
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer)
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

        validation_loss, validation_accuracy = validate(model, device, validation_loader)
        print(f"Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_path = os.path.join('models', f'model_epoch_{epoch}_val_acc_{validation_accuracy:.4f}.pt')
            save_model(model, optimizer, epoch, validation_loss, validation_accuracy, save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='real_data')
    parser.add_argument('--fake-dataset', type=str, default='fake_data')  # 添加生成数据集的参数
    parser.add_argument('--bert-models', type=str, default='D:/Content_Secu/task4/bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=2e-5)

    args = parser.parse_args()
    main(args)