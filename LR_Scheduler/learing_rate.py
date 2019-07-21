import torch as t
import matplotlib.pyplot as plt
import math
plt.interactive(False)


def find_lr(net, trn_loader, optimizer, criterion, init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    ii = 0
    for data in trn_loader:
        batch_num += 1
        # t.cuda.empty_cache()
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        inputs, labels = (inputs.cuda()), (labels.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        ii += 1
        if ii % 1000 ==0:
            print(ii)
    return log_lrs, losses


def find_learning_rate(net, train_loader,optimizer=None):
    criterion = t.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = t.optim.SGD(net.parameters(), lr=0.01)
    t.cuda.empty_cache()
    logs, losses = find_lr(net, train_loader, optimizer, criterion, init_value=1e-8, final_value=10., beta=0.98)
    print(logs)
    print(losses)
    # plt.xticks([10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5])
    plt.figure()
    plt.xticks([10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5])
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.plot(logs[10:-5], losses[10:-5])
    plt.show()
    plt.savefig('lr1.png')
