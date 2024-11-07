from read_dataset import *
from preprop import *
from model import segmentation_model
import torch

def train(img_list, gt_list, model, epoch, learning_rate, optimizer, criterion, data_len):
    
    running_loss = 0.0
    optimizer.zero_grad()

    for i in range(epoch):
        for iter in range(data_len):
            inputs, label = get_data(img_list[iter],gt_list[iter])
            # training loop
            model.train()
            inputs = torch.tensor(inputs).cuda()
            label = torch.tensor(label).long().cuda()

            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (iter % 100 == 0) & (iter != 0):
                print(f'Iteration: {iter+data_len*i}, Loss: {running_loss / (iter+1+data_len*i)}')
        torch.save(model.state_dict(), f'model_state_dict{i}.pth')