import torch


def evaluate(model, dataloader, topk=(1,5)):
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    model.eval()
    acc = [0.] * len(topk)
    total = 0
    with torch.no_grad():
        maxk = max(topk)
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[0]
            output = model(data)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            total += targets.shape[0]
            for i, k in enumerate(topk):
                correct_k = correct[:k].view(-1).float().sum(0)
                acc[i] += correct_k
        res = [0.] * len(topk)
        for i, k in enumerate(topk):
            res[i] = acc[i] / total
        return res

