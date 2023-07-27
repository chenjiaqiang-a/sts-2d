import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    # pred   : [bs, height, width] -> [bs, width * height]
    # target : [bs, height, width] -> [bs, width * height]
    height, width = pred.size(1), pred.size(2)
    pred, target = pred.view(-1, width * height), target.view(-1, width * height)

    # intersection : [bs,]
    # dice_coef    : [bs,]
    intersection = (pred * target).sum(dim=1)
    dice_coef = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice_coef


def intersection_over_union(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    # pred   : [bs, height, width] -> [bs, width * height]
    # target : [bs, height, width] -> [bs, width * height]
    height, width = pred.size(1), pred.size(2)
    pred, target = pred.view(-1, width * height), target.view(-1, width * height)

    intersection = (pred * target).sum(dim=1)
    union = torch.maximum(pred, target).sum(dim=1)
    iou = (intersection + smooth) / (union + smooth)
    return iou


def hausdorff_distance_2d(pred: torch.Tensor, target: torch.Tensor):
    pred, target = pred.float(), target.float()
    distance_matrix = torch.cdist(pred, target, p=2)

    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    value = torch.cat((value1, value2), dim=1)
    return value.max(1)[0]


if __name__ == '__main__':
    x = torch.zeros((8, 8))
    y = torch.zeros((8, 8))
    x[2:6, 2:6] = 1
    y[3:7, 4:6] = 1
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    print(f"Dice Coefficient: {dice_coefficient(x, y)}")
    print(f"Intersection over Union: {intersection_over_union(x, y)}")
    print(f"2D Hausdorff Distance: {hausdorff_distance_2d(x, y)}")
