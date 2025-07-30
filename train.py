import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
import numpy as np
import cnn
from collections import Counter
from torch.utils.data import random_split
from plot import plot_save, plot_bar, plot_graph_save

transform = {
    'train': transforms.Compose([transforms.Resize((128,128)),
                                 transforms.CenterCrop(100),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.RandAugment(2,4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.18300477, 0.18300807, 0.18305317), (0.19476049, 0.19475904, 0.19479235))]),
    'test': transforms.Compose([transforms.Resize((128,128)),
                                transforms.CenterCrop(100),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.18300477, 0.18300807, 0.18305317), (0.19476049, 0.19475904, 0.19479235))])
}

trainset = datasets.ImageFolder('dataset/Training', transform = transform['train'])
testset = datasets.ImageFolder('dataset/Testing', transform = transform['test'])

class_sample_count = np.array([len(np.where(trainset.targets == t)[0]) for t in np.unique(trainset.targets)])
weight = 1. / class_sample_count
print(weight)
sample_weight = np.array([weight[int(t)] for t in trainset.targets])
sample_weight = torch.from_numpy(sample_weight)
sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))

val, test = random_split(testset, [0.4, 0.6])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, sampler = sampler)
valloader = torch.utils.data.DataLoader(val, batch_size=4, shuffle=False)
testloader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=False)


classes = trainset.classes
num_classes = len(classes)

val_samples = {}
train_samples = {}

for idx, c in enumerate(classes):
    val_samples[c] = dict(Counter(testset.targets[i] for i in val.indices))[idx]
    train_samples[c] = dict(Counter(trainset.targets))[idx]

print(val_samples)
print(train_samples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = cnn.CNN(num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005, momentum=0.95)
scheduler = ExponentialLR(optimizer, gamma=0.95)

total_step = len(trainloader)

train_accuracy = []
test_accuracy = []

training_loss = []
validation_loss = []

n_epoch = 50
max_acc = 0

for epoch in range(n_epoch):
    loss = None
    running_loss = 0.0
    correct = 0
    correct_class = {}
    total = 0
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
    accuracy = 100 * correct / total
    train_accuracy.append(accuracy)
    training_loss.append(running_loss / len(trainloader))
    scheduler.step()

    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy on {} train images: {:.2f}%'.format(epoch + 1, n_epoch, running_loss / len(trainloader), total, accuracy))

    with torch.no_grad():
        correct = 0
        correct_class = {}
        val_loss = 0
        for c in classes:
            correct_class[c] = 0
        total = 0
        model.eval()
        for idx, (images, labels) in enumerate(valloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()
            for j in range(len(predicted)):
                prediction = classes[predicted[j]]
                if (predicted == labels).tolist()[j]:
                    correct_class[prediction] += 1

        accuracy = 100 * correct / total
        test_accuracy.append(accuracy)
        validation_loss.append(val_loss / len(valloader))
        print('Epoch [{}/{}], Loss: {:.4f} Accuracy on {} validation images: {:.2f}%'.format(epoch +1, n_epoch, val_loss / len(valloader), total, accuracy))

        if accuracy > max_acc:
            for c in correct_class:
                correct_class[c] = (correct_class[c] / val_samples[c]) * 100
            correct_class['Average'] = accuracy
            print(correct_class)

            print("Accuracy Increased to {}. Model Saved!".format(accuracy))
            max_acc = accuracy
            torch.save(model, 'model/model.pth'.format(accuracy))

            plot_save(correct_class, 'model')

            title = 'Training - Test Accuracy SGD ({} epochs)'.format(epoch + 1)
            plot_graph_save(train_accuracy, test_accuracy, title,'model', 'accuracy')

            title = 'Training - Loss ({} epochs)'.format(epoch + 1)
            plot_graph_save(training_loss, validation_loss, title, 'model', 'loss')

#TEST ON SPLIT SET
model = torch.load('model/model.pth', weights_only=False)
model.eval()

total = 0
correct = 0
correct_class = {}
test_accuracy = []
test_samples = {}

for c in classes:
    correct_class[c] = 0

for idx, c in enumerate(classes):
    test_samples[c] = dict(Counter(testset.targets[i] for i in test.indices))[idx]

for idx, (images, labels) in enumerate(testloader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    for j in range(len(predicted)):
        prediction = classes[predicted[j]]
        if (predicted == labels).tolist()[j]:
            correct_class[prediction] += 1

accuracy = 100 * correct / total
test_accuracy.append(accuracy)
print('\n\nTEST: Accuracy on {} test images: {:.2f}%'.format(total, accuracy))

for c in correct_class:
    correct_class[c] = (correct_class[c] / test_samples[c]) * 100
correct_class['Average'] = accuracy
print(correct_class)

plot_bar(correct_class)