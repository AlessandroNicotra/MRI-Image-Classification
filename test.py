import torch
from torchvision import datasets, transforms
from collections import Counter
from plot import plot_bar

transform = transforms.Compose([transforms.Resize((128,128)),
            transforms.CenterCrop(100),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.18300477, 0.18300807, 0.18305317), (0.19476049, 0.19475904, 0.19479235))])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testset = datasets.ImageFolder('dataset/Testing', transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

classes = testset.classes
num_classes = len(classes)

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
    test_samples[c] = dict(Counter(testset.targets))[idx]

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
print('Accuracy on {} val images: {:.2f}%'.format(total, accuracy))

for c in correct_class:
    correct_class[c] = (correct_class[c] / test_samples[c]) * 100
correct_class['Average'] = accuracy
print(correct_class)

plot_bar(correct_class)