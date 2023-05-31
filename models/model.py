import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from dataset.dataset_utils import prepare_data_loaders, prepare_dataset


def evaluate_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            labels = labels.view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_total
    average_test_loss = test_loss / len(dataloader)

    return average_test_loss, test_accuracy


num_classes = 1  # Одна классификационная категория: паспорт или не паспорт
resnet18 = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)


# Замена последнего слоя классификатора на слой с одним выходом и сигмоидной активацией
resnet18.fc = nn.Sequential(
    nn.Linear(in_features=512, out_features=num_classes),
    nn.Sigmoid()
)

# Перевод модели в режим обучения
resnet18.train()

criterion = nn.BCELoss()  # Бинарная кросс-энтропия, так как у нас только один класс
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

prepare_dataset()
train_loader, test_loader = prepare_data_loaders()

num_epochs = 10

# Цикл обучения
full_train_timer = time.time()
train_loss_history, train_accuracy_history = [], []
test_loss_history, test_accuracy_history = [], []
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        start = time.time()
        # Получение входных данных и меток классов
        inputs, labels = data

        # Приведение размера целевого тензора
        labels = labels.view(-1, 1)

        # Обнуление градиентов перед обратным распространением
        optimizer.zero_grad()

        # Прямой проход
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels.float())

        # Обратное распространение и оптимизация
        loss.backward()
        optimizer.step()

        # Сбор метрик
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_loss_history.append(round(running_loss / 100, 4))
        train_accuracy_history.append(round(correct / total, 4))

        # Вывод статистики обучения
        print(
            f'[{epoch + 1}, {i + 1}] loss: {round(running_loss / 100, 4)} '
            f'accuracy: {round(correct / total, 4)} time: {round(time.time() - start)}'
        )

    # Расчет и сохранение loss и accuracy на тестовой выборке
    test_timer = time.time()
    test_loss, test_accuracy = evaluate_model(resnet18, test_loader, criterion)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)
    print(
        f'[evaluate] loss: {round(test_loss, 4)} accuracy: {round(test_accuracy, 4)} '
        f'time: {round(time.time() - test_timer)}'
    )


print(f"full train time: {round(time.time() - full_train_timer)}")

# Рисование графиков
plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(train_accuracy_history, label='Train Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
