import torch
import config


def replace_base_fc(trainloader, model):
    # replace fc.weight with the embedding average of train data
    cfg = config.CONFIG()
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for data, label in trainloader:
            data = data[0].cuda()
            label = label.cuda()
            _, _, _, embedding = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    for class_index in range(cfg.DATASET.NUM_CLASS):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    proto_list = torch.stack(proto_list, dim=0)

    model.classifier3.protos.data = proto_list.squeeze(-1).squeeze(-1).cuda()
    model = model.train()

    return model
