
from torch.autograd import Variable
from utils.metric import compress, calculate_top_map, euclidean_dist
import utils.datasets_nus as datasets_nus
from utils.models import ImgNet, TxtNet, JNet, GCNLI, GCNLT ,GCNet ,GCN_Img ,ImgNet_CLIP
import time
import os
from utils.utils import *
import logging
from utils.models import *
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="DESAH demo")
parser.add_argument('--bits', default='32,64,128', type=str,help='binary code length (default: 128)')
parser.add_argument('--gpu', default='0', type=str,help='selected gpu (default: 0)')
parser.add_argument('--BETA', default=0.9, type=float, help='hyper-parameter: balance parameter')
parser.add_argument('--batch-size', default=16, type=int, help='batch size (default: 32)')
parser.add_argument('--CODE_LEN', default=16, type=int, help='binary code length (default: 16)')
parser.add_argument('--LAMBDA1', default=0.01, type=float, help='hyper-parameter:  (default: )')
parser.add_argument('--LAMBDA2', default=0.01, type=float, help='hyper-parameter: (default: )')
parser.add_argument('--LAMBDA3', default=1, type=float, help='hyper-parameter:  (default: )')
parser.add_argument('--LAMBDA4', default=40, type=float, help='hyper-parameter: (default: )')
parser.add_argument('--LAMBDA5', default=40, type=float, help='hyper-parameter:  (default: )')
parser.add_argument('--kx', default=2, type=float,help='hyper-parameter: balance parameter')
parser.add_argument('--ky', default=2, type=float, help='hyper-parameter: balance parameter')
parser.add_argument('--NUM-EPOCH', default=20, type=int, help='hyper-parameter: EPOCH (default: 40)')
parser.add_argument('--LR-IMG', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR-TXT', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-2)')
parser.add_argument('--MOMENTUM', default=0.9, type=float, help='hyper-parameter: momentum (default: 0.9)')
parser.add_argument('--WEIGHT-DECAY', default=5e-4, type=float, help='hyper-parameter: weight decay (default: )')
parser.add_argument('--NUM-WORKERS', default=0, type=int, help='workers (default: )')
parser.add_argument('--EVAL', default= False, type=bool,help='')
parser.add_argument('--EPOCH-INTERVAL', default=1, type=int, help='INTERVAL (default: 2)')
parser.add_argument('--EVAL-INTERVAL', default=4, type=int, help='evaluation interval (default:)')

class Session:
    def __init__(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)

        self.train_dataset = datasets_nus.NUSWIDE(train=True, transform=datasets_nus.nus_train_transform)
        self.test_dataset = datasets_nus.NUSWIDE(train=False, database=False, transform=datasets_nus.nus_test_transform)
        self.database_dataset = datasets_nus.NUSWIDE(train=False, database=True, transform=datasets_nus.nus_test_transform)
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=args.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=args.NUM_WORKERS)

        self.best_it = 0
        self.best_ti = 0

    def define_model(self, coed_length):

        self.FeatNet_I = ImgNet(code_len=coed_length)
        self.CodeNet_I = ImgNet(code_len=coed_length)
        # self.FeatNet_I = GCN_Img(code_len=coed_length)
        # self.CodeNet_I = GCN_Img(code_len=coed_length)
        # self.FeatNet_I = ImgNet_CLIP(code_len=coed_length)
        # self.CodeNet_I = ImgNet_CLIP(code_len=coed_length)
        txt_feat_len = datasets_nus.txt_feat_len
        self.CodeNet_T = TxtNet(code_len=coed_length, txt_feat_len=txt_feat_len)
        # self.CodeNet_T = GCNet(code_len=coed_length, txt_feat_len=txt_feat_len)

        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=args.LR_IMG, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=args.LR_TXT, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)

    def train(self, epoch, args):
        self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f' % (epoch + 1, args.NUM_EPOCH, self.CodeNet_I.alpha))

        for idx, (img, F_T, labels, _) in enumerate(self.train_loader):
            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            F_I, _, _ = self.FeatNet_I(img)


            _, hid_I, code_I = self.CodeNet_I(img)

            #改成 GCN
            _, hid_T, code_T = self.CodeNet_T(F_T)

            # construct similarity matrix

            F_I = F.normalize(F_I)
            F_T = F.normalize(F_T)

            S_I = F_I.mm(F_I.t())
            S_I= S_I.detach().clone()
            S_I = S_I * (S_I.mm(S_I.t()))
            S_I = S_I * args.kx - 1

            S_T1 = F_T.mm(F_T.t())
            S_T2 = S_T1.detach().clone()
            S_T = S_T1 * (S_T2.mm(S_T2.t()))
            S_T = S_T * args.ky - 1

            # S = args.BETA * S_I + (1 - args.BETA) * S_T
            #以下是修改内容
            S_I_normal = F.normalize(S_I)
            S_T_normal = F.normalize(S_T)
            attention_I = S_I_normal.mm(S_T_normal.t())
            attention_T = S_T_normal.mm(S_I_normal.t())
            S = attention_I * S_I + attention_T * S_T



            # optimize
            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)
            BI = B_I.detach().clone()
            BT = B_T.detach().clone()
            BI =BI.cpu().data.numpy()
            BT = BT.cpu().data.numpy()

            B = (args.LAMBDA1 * S.cpu().detach().numpy().transpose().dot(BI) + args.LAMBDA2 *
                 S.detach().cpu().numpy().transpose().dot(BT) + (args.LAMBDA4 * BI + args.LAMBDA5 * BT)).dot(
                np.linalg.inv(args.LAMBDA1 * BI.transpose().dot(BI) + args.LAMBDA2 * BT.transpose().dot(BT) +
                              (args.LAMBDA4 + args.LAMBDA5) * np.eye(args.CODE_LEN)))

            B = torch.from_numpy(B).type(torch.FloatTensor).cuda()


            B_BI = B.mm(B_I.t())
            B_BT = B.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())

            loss1 = F.mse_loss(B_BI, S)
            loss2 = F.mse_loss(BI_BT, S)
            loss3 = F.mse_loss(B_BT, S)
            loss4 = F.mse_loss(B_I, B)
            loss5 = F.mse_loss(B_T, B)

            # 自编码器损失
            # loss_selfencoder = F.mse_loss(F_I, selfcode_I)
            loss = args.LAMBDA1 * loss1 + args.LAMBDA3 * loss2 + args.LAMBDA2 * loss3 + args.LAMBDA4 * loss4 +args.LAMBDA5 * loss5
                   # args.LAMBDA1 * loss_selfencoder

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(self.train_dataset) // args.batch_size / args.EPOCH_INTERVAL) == 0:
                logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f'
                            % (epoch + 1, args.NUM_EPOCH, idx + 1, len(self.train_dataset) // args.batch_size,
                                loss.item()))

    def eval(self, epoch, bit):
        logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T, self.database_dataset, self.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        if (self.best_it + self.best_ti) < (MAP_I2T + MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I

        logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        logger.info('Best MAP of Image to Text: %.3f, Best MAP of Text to Image: %.3f' % (self.best_it, self.best_ti))
        logger.info('--------------------------------------------------------------------')


    def normalize(self,affnty):
        col_sum = affnty.sum(axis=1)[:, np.newaxis]
        row_sum = affnty.sum(axis=0)

        out_affnty = affnty / col_sum
        in_affnty = (affnty / row_sum).t()

        out_affnty = Variable(torch.Tensor(out_affnty)).cuda()
        in_affnty= Variable(torch.Tensor(in_affnty)).cuda()

        return in_affnty, out_affnty


def mkdir_multi(path):
    # confirm if the path exists
    isExists = os.path.exists(path)

    if not isExists:
        # if not, create path
        os.makedirs(path)
        print('successfully creat path！')
        return True
    else:
        # if exists, notify
        print('path already exists！')
        return False


def _logging():
    global logger
    # logfile = os.path.join(logdir, 'log.log')
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def main():
    global logdir, args

    args = parser.parse_args()

    sess = Session()

    bits = [bit for bit in args.bits.split(',')]
    for bit in bits:
        logdir = './DESAH/nus/'  + bit + '/'
        args.bits = bit
        args.CODE_LEN = int(bit)
        mkdir_multi(logdir)
        _logging()

        if args.EVAL == True:
            sess.load_checkpoints()
        else:
            logger.info("原模型 bits = {}".format(bit))
            logger.info('--------------------------train Stage--------------------------')
            sess.define_model(args.CODE_LEN)
            for epoch in range(args.NUM_EPOCH):
                # train the Model
                sess.train(epoch, args)
                if (epoch + 1) % args.EVAL_INTERVAL == 0:
                    sess.eval(epoch, bit)

if __name__=="__main__":
    main()