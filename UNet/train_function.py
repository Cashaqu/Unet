from datetime import datetime
import torch


def training(model, num_epochs, train_loader, valid_loader, device, criterion, optimizer):
    print('Start training...')
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        train_running_loss = 0
        valid_running_loss = 0

        model.train()
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            predict = model(image)
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_running_loss += loss.item()

        model.eval()
        for image, label in valid_loader:
            image, label = image.to(device), label.to(device)

            with torch.no_grad():
                predict = model(image)
                loss = criterion(predict, label)

                valid_losses.append(loss.item())
                valid_running_loss += loss.item()

        train_running_loss = train_running_loss / len(train_loader)
        valid_running_loss = valid_running_loss / len(valid_loader)

        print(f'Epoch: {epoch}; Train Loss: {train_running_loss:.4f}; Validation Loss: {valid_running_loss:.4f}')

    timestemp_for_model = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    torch.save(model, f'./models/model_{timestemp_for_model}.pt')
    print(f'Model saved as ./models/model_{timestemp_for_model}.pt')
    return timestemp_for_model