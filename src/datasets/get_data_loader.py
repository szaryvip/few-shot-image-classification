import learn2learn as l2l
import torch


def get_data_loader(dataset, way, shot, num_tasks, shuffle):
    dataset = l2l.data.MetaDataset(dataset)
    task_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset, n=way, k=shot+1),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
    ]

    task_dataset = l2l.data.TaskDataset(dataset, task_transforms=task_transforms, num_tasks=num_tasks)
    task_dataloader = torch.utils.data.DataLoader(task_dataset, shuffle=shuffle)
    return task_dataloader
