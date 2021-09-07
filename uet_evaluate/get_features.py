import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from torchvision.datasets.folder import default_loader
from custom_dataloader import ImageFolderWithPaths

from torchreid.utils import FeatureExtractor
import time
# from IPython import embed


def calculate_features(net, root_dir, batch, save_features = True, save_dir = None, save_name = None):

    query_dir = os.path.join(root_dir, "query")
    gallery_dir = os.path.join(root_dir, "gallery")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 256)),
        torchvision.transforms.ToTensor()
        ])


    queryloader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(query_dir, transform=transform),
        batch_size=batch, shuffle=False, num_workers=8, pin_memory=True
        )

    '''
    try load gallery random
    '''
    galleryloader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(gallery_dir, transform=transform),
        batch_size=batch, shuffle=False, num_workers=8, pin_memory=True
        )

    # compute features
    query_features = torch.tensor([]).float()
    query_labels = torch.tensor([]).long()
    query_paths = []

    gallery_features = torch.tensor([]).float()
    gallery_labels = torch.tensor([]).long()
    gallery_paths = []

    with torch.no_grad():
        print('Calculating embedding for query ...')
        for idx, (inputs, labels, paths) in enumerate(queryloader):        

            inputs = inputs.to(device)

            features = net(inputs).detach().cpu()

            query_features = torch.cat((query_features, features), dim=0)
            query_labels = torch.cat((query_labels, labels))
            query_paths.extend(paths)

            # embed(header='debug model')


        print('Calculating embedding for gallery ...')
        
        start_dataset = time.time()
        for idx, (inputs, labels, paths) in enumerate(galleryloader):
            print("Dataset load time: ", (time.time() - start_dataset) / (idx+1))
            start_infer = time.time()
            if(idx % 10 == 0):
                print(idx)
            inputs = inputs.to(device)

            start = time.time()
            features = net(inputs).cpu()
            print("Model infer time: ", time.time() - start)

            gallery_features = torch.cat((gallery_features, features), dim=0)
            gallery_labels = torch.cat((gallery_labels, labels))
            gallery_paths.extend(paths)
            print("1 batch infer time: ", time.time() - start_infer)

    # save features
    features = {
        "qf": query_features,
        "ql": query_labels,
        "gf": gallery_features,
        "gl": gallery_labels,
        "query_paths": query_paths,
        "gallery_paths": gallery_paths
    }

    if save_features:
        # save_name = os.path.basename(args.ckpt)[:-3]
        save_path = os.path.join(save_dir, 'features_' + save_name + '.pth')
        torch.save(features, save_path)
        print('Save features to {}'.format(save_path))

    return features
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/reid_dataset', type=str)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", default=1, type=int)    
    parser.add_argument("--reid_model_path", default="./checkpoint/ckpt.t7", type=str)
    parser.add_argument("--reid_model_name", default="./checkpoint/ckpt.t7", type=str)

    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--save-dir", default="predicts/", type=str)
    parser.add_argument("--save-name", default="debug", type=str)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True

    assert os.path.isfile(
    args.reid_model_path), "Error: no checkpoint file found!"

    print(device)
    print('Loading from {}'.format(args.reid_model_path))
    feature_extractor = FeatureExtractor(model_name=args.reid_model_name, model_path=args.reid_model_path, device = device)      

    calculate_features(net = feature_extractor, root_dir = args.data_dir, batch = args.batch, save_dir=args.save_dir, save_name=args.save_name)
