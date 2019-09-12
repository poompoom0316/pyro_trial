from __future__ import absolute_import, division, print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer
from pyro.contrib.examples.util import get_data_loader, get_data_directory


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu((self.fc1(x)))
        x = self.fc2(x)
        return x


def train(args, train_loader, gpmodule, optimizer, loss_fn, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.binary:
            target = (target % 2).float()  # convert numbers 0->9 to 0 or 1

        gpmodule.set_data(data, target)
        optimizer.zero_grad()
        loss = loss_fn(gpmodule.model, gpmodule.guide)
        loss.backward()
        optimizer.step()
        batch_idx = batch_idx+1
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {:2d} [{:5d}/{} {:2.0f}%]\tLoss: {:.6f}"
                  .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx /len(train_loader), loss))


def test(args, test_loader, gpmodule, optimizer, loss_fn, epoch):
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.binary:
            target = (target % 2).float()  # convert numbers 0->9 to 0 or 1

        # gpmodelのdataに対する予測値を￥お得る
        f_loc, f_var = gpmodule(data)
        # 尤度を使って予測を実行
        pred = gpmodule.likelihood(f_loc, f_var)
        # 予測値と実測値を比較
        correct += pred.eq(target).long().cpu().sum().item()

    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n"
          .format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return pred, target


def main(args):
    data_dir = args.data_dir if args.data_dir is not None else get_data_directory(__file__)
    train_loader = get_data_loader(dataset_name="MNIST",
                                   data_dir=data_dir,
                                   batch_size=args.batch_size,
                                   dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                   is_training_set=True,
                                   shuffle=True) # ここのマジックナンバーはどこから来ているのか？
    test_loader = get_data_loader(dataset_name="MNIST",
                                  data_dir=data_dir,
                                  batch_size=args.test_batch_size,
                                  dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                  is_training_set=False,
                                  shuffle=False)

    if args.cuda:
        train_loader.num_workers = 1
        test_loader.num_workers = 1

    cnn = CNN()

    # RBFをCNNによってワープさせることで深層カーネルを作成
    # CNNによって高次元の画像を10次元に落とし込み.
    # RBFはCNNへの入力を入力としてCNNの出力間の共分散を算出する
    # on outputs of CNN.

    rbf = gp.kernels.RBF(input_dim=10, lengthscale=torch.ones(10))
    deep_kernel = gp.kernels.Warping(rbf, iwarping_fn=cnn)

    batches = []
    for i, (data, _) in enumerate(train_loader):
        batches.append(data)
        if i>= ((args.num_inducing-1) // args.batch_size):
            break
    Xu = torch.cat(batches)

    if args.binary:
        likelihood = gp.likelihoods.Binary()
        latent_shape = torch.Size([])
    else:
        # multiclass の尤度
        likelihood = gp.likelihoods.MultiClass(num_classes=10)
        # データのクラスが10あるから、各クラスの出力もあるクラスに所属する確率を各クラスに対して出力するモデルが必要
        # よってlatent_shape = [10]
        latent_shape = torch.Size([10])

    gpmodule = gp.models.VariationalSparseGP(X=Xu, y=None, kernel=deep_kernel,
                                             Xu=Xu, likelihood=likelihood, latent_shape=latent_shape,
                                             num_data=60000, whiten=True)
    if args.cuda:
        gpmodule.cuda()

    optimizer = torch.optim.Adam(gpmodule.parameters(), lr=args.lr)

    elbo = infer.JitTraceMeanField_ELBO() if args.jit else infer.TraceMeanField_ELBO() #ここのELBOの選び方は平均場近似というやつか？
    loss_fn = elbo.differentiable_loss

    test_list = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, train_loader, gpmodule, optimizer, loss_fn, epoch)
        with torch.no_grad():
            test_res = test(args, test_loader, gpmodule, optimizer, loss_fn, epoch)
            test_list.append(test_res)
        print("Amount of time spent for epoch {}: {}s\n"
              .format(epoch, int(time.time()-start_time)))


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description='Pyro GP MNIST Example')
    parser.add_argument('--data-dir', type=str, default="data/", metavar='PATH',
                        help='default directory to cache MNIST data')
    parser.add_argument('--num-inducing', type=int, default=70, metavar='N',
                        help='number of inducing input (default: 70)')
    parser.add_argument('--binary', action='store_true', default=False,
                        help='do binary classification')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='enables PyTorch jit')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("-mode", "--mode", default="0", type=str,
                        help="unnecessary")
    parser.add_argument("-port", "--port", default=0, type=int,
                        help="unnecessary")
    args = parser.parse_args()

    pyro.set_rng_seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    main(args)


fig = plt.figure()
count = 0
for i in range(10):
    for j in range(20):
        count +=1
        fig.add_subplot(20, 10, count)
        plt.imshow(aa.cpu().detach().numpy()[j,i,:,:])


