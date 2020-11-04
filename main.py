import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from time import time
import matplotlib.pyplot as plt
import numpy as np

from custom_dataset import CustomDatasetFromImages
from cnn_model import CNNModel


def classify(img, ps):
    ''' 
    Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()/100

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(11), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(11))
    ax2.set_yticklabels(np.arange(11))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
    traindata = CustomDatasetFromImages('train_img_list.csv', './train_data')
    testdata = CustomDatasetFromImages('test_img_list.csv', './test_data')
    

    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    num_epochs = 15
    time0 = time()
    for epoch in range(num_epochs):
        running_loss  = 0
        for i, (images, labels) in enumerate(traindata):
            images = Variable(images)
            images = images.unsqueeze(1)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            
        print ('Epoch : %d/%d,  Loss: %.4f' %(epoch+1, num_epochs, running_loss/traindata.__len__()))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)


    images, labels = next(iter(testdata))
    # replace trainloader to check training accuracy.

    img = images.unsqueeze(1)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logpb = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    pb = torch.exp(logpb)
    probab = list(pb.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    classify(img.view(1, 28, 28), pb)

    # Evaluate model accuracy based on testdata
    correct_count, all_count = 0, 0
    for i, (images,labels) in enumerate(testdata):
        img = images.unsqueeze(1)

        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[0]
        if(true_label == pred_label): correct_count += 1
        all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))

    torch.save(model, 'digit_recog_cnn.pt')
