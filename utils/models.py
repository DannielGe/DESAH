import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .clip import clip
from .pygcn.layers import  GraphConvolution
import numpy as np
import scipy.sparse as sp



# class ImgNet(nn.Module):
#     def __init__(self, code_len):
#         super(ImgNet, self).__init__()
#         self.alexnet = torchvision.models.alexnet(pretrained=True)
#         self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
#         # self.attention = nn.Sequential(nn.Linear(4096,4096),
#         #                                nn.Sigmoid())
#         self.fc_encode = nn.Linear(4096, code_len)
#         self.alpha = 1.0
#
#     def forward(self, x):
#         x = self.alexnet.features(x)
#         x = x.view(x.size(0), -1)
#         feat = self.alexnet.classifier(x)
#         # att_feat = self.attention(feat)
#         # atten = torch.mul(feat,att_feat) + feat
#         hid = self.fc_encode(feat)
#         code = torch.tanh(self.alpha * hid)
#
#         return feat, hid, code
#
#     def set_alpha(self, epoch):
#         self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)
        # self.vgg_features = self.vgg.features
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        self.fc_encode = nn.Linear(4096, code_len)
        #att
        # self.attention = nn.Sequential(nn.Linear(4096,4096),
        #                                nn.Sigmoid())
        #新增自编码器 -- （两层的全连接）
        # self.selfencoder1 = nn.Linear(code_len,1024)
        # self.selfencoder2 = nn.Linear(1024, 4096)
        # self.selfencoder = nn.Sequential(nn.Linear(code_len, 1024),
        #                                nn.ReLU(),
        #                                nn.Linear(1024,4096)
        #                                )
        self.alpha = 1.0

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        feat = self.vgg.classifier(x)
        #att
        # att_feat = self.attention(feat)
        # atten = torch.mul(feat, att_feat) + feat

        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        # selfencode = self.selfencoder(code)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


#改成resnet
# class ImgNet(nn.Module):
#     def __init__(self, code_len):
#         super(ImgNet, self).__init__()
#         self.resnet = torchvision.models.resnet34(pretrained=True)
#         # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
#         self.fc_encode = nn.Sequential(nn.Linear(512, 4096),
#                                        nn.ReLU(),
#                                        nn.Linear(4096,code_len)
#                                        )
#         self.alpha = 1.0
#
#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)
#
#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)
#
#         x = self.resnet.avgpool(x)
#         feat = x.view(x.size(0), -1)
#         hid = self.fc_encode(feat)
#         code = torch.tanh(self.alpha * hid)
#
#         return feat, hid, code
#
#     def set_alpha(self, epoch):
#         self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class ImgNet_CLIP(nn.Module):
    def __init__(self, code_len):
        super(ImgNet_CLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_image_encode, _= clip.load("ViT-B/16", device=self.device)   #512
        self.clip_image_encode, _= clip.load("RN50x16", device=self.device)   #768
        # self.fc_encode = nn.Linear(512,4096)
        self.hash_layer = nn.Sequential(nn.Linear(768, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, code_len),
                                        )
        self.alpha = 1.0

    def forward(self, x):
        with torch.no_grad():
            feat = self.clip_image_encode.encode_image(x)
            feat = feat.type(torch.float32)
        # feat = self.fc_encode(feat)
        hid = self.hash_layer(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, hid ,code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNLI(nn.Module):
    def __init__(self, code_len):
        super(GCNLI, self).__init__()

        self.gconv1 = nn.Linear(4096, 2048)
        self.BN1 = nn.BatchNorm1d(2048)
        self.act1 = nn.ReLU()

        self.gconv2 = nn.Linear(2048, 2048)
        self.BN2 = nn.BatchNorm1d(2048)
        self.act2 = nn.ReLU()

        self.gconv3 = nn.Linear(2048, code_len)
        self.alpha = 1.0

    def forward(self, x, in_affnty, out_affnty):
        out = self.gconv1(x)
        out = in_affnty.mm(out)
        out = self.BN1(out)
        out = self.act1(out)

        # block 2
        out = self.gconv2(out)
        out = out_affnty.mm(out)
        out = self.BN2(out)
        out = self.act2(out)

        # block 3
        out = self.gconv3(out)
        out = torch.tanh(self.alpha * out)

        return out

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNLT(nn.Module):
    def __init__(self, code_len):
        super(GCNLT, self).__init__()

        self.gconv1 = nn.Linear(4096, 2048)
        self.BN1 = nn.BatchNorm1d(2048)
        self.act1 = nn.ReLU()

        self.gconv2 = nn.Linear(2048, 2048)
        self.BN2 = nn.BatchNorm1d(2048)
        self.act2 = nn.ReLU()

        self.gconv3 = nn.Linear(2048, code_len)
        self.alpha = 1.0

    def forward(self, x, in_affnty, out_affnty):
        out = self.gconv1(x)
        out = in_affnty.mm(out)
        out = self.BN1(out)
        out = self.act1(out)

        # block 2
        out = self.gconv2(out)
        out = out_affnty.mm(out)
        out = self.BN2(out)
        out = self.act2(out)

        # block 3
        out = self.gconv3(out)
        out = torch.tanh(self.alpha * out)

        return out

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.attention = nn.Sequential(nn.Linear(4096, 4096),
                                       nn.Sigmoid())

        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, code_len)
        # #以下是消融部分
        # self.fc = nn.Linear(txt_feat_len, code_len)
        # self.alpha = 1.0

    def forward(self, x):

        feat = self.fc1(x)
        feat = F.relu(self.fc2(feat))
        hid = self.fc3(feat)
        # 以下是消融部分
        # feat = x
        # hid = self.fc(feat)

        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class JNet(nn.Module):
    def __init__(self, code_len):
        super(JNet, self).__init__()
        self.fc_encode = nn.Linear(8192, code_len)
        self.alpha = 1.0

    def forward(self, x):
        hid = self.fc_encode(x)
        code = torch.tanh(self.alpha * hid)

        return hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class GCNet(nn.Module):
    def __init__(self, code_len, txt_feat_len,  dropout=0.5):
        super(GCNet, self).__init__()

        # self.gc1 = GraphConvolution(txt_feat_len, 1024)
        # self.gc2 = GraphConvolution(1024, 1024)
        # self.gc3 = GraphConvolution(1024, code_len)
        # att
        # self.attention = nn.Sequential(nn.Linear(4096, 4096),
        #                                nn.Sigmoid())

        self.gc1 = GraphConvolution(txt_feat_len, 4096)
        self.gc2 = GraphConvolution(4096, 4096)
        self.gc3 = GraphConvolution(4096, code_len)
        self.dropout = dropout
        self.alpha = 1.0

    def forward(self, x):
        x, adj = self.generate_txt_graph(x)
        feat = self.gc1(x, adj)
        feat = F.relu(self.gc2(feat, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        #att
        # att_feat = self.attention(feat)
        # atten = torch.mul(feat, att_feat) + feat

        hid = self.gc3(feat, adj)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code
    #改成 两层的 图卷积
    # def forward(self, x):
    #     x, adj = self.generate_txt_graph(x)
    #     feat = self.gc1(x, adj)
    #     feat = F.relu(feat)
    #     # x = F.dropout(x, self.dropout, training=self.training)
    #     hid = self.gc3(feat, adj)
    #     code = torch.tanh(self.alpha * hid)
    #     return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

        # 改成GCN后加的 两个方法
    def normalize_mx(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))  # 矩阵求和
        r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx  # 返回归一化之后的矩阵

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def generate_txt_graph(self, txt):
        txt_feature = txt
        adj = txt.mm(txt.t())
        adj = torch.sign(adj)
        adj2triad = sp.csr_matrix(adj.cpu().numpy())
        adj2triad = adj2triad + adj2triad.T.multiply(adj2triad.T > adj2triad) - adj2triad.multiply(
            adj2triad.T > adj2triad)
        adj = self.normalize_mx(adj2triad + sp.eye(adj2triad.shape[0]))
        adjacencyMatrix = self.sparse_mx_to_torch_sparse_tensor(adj)
        adjacencyMatrix = adjacencyMatrix.cuda()
        return txt_feature, adjacencyMatrix

class GCN_Img(nn.Module):
    def __init__(self, code_len):
        super(GCN_Img, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.gc1 = GraphConvolution(4096, 4096)
        self.gc2 = GraphConvolution(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        feat_graph, adj = self.generate_img_graph(feat)
        hid = self.gc2( F.relu(self.gc1(feat_graph,adj)) , adj )
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

    # 改成GCN后加的 两个方法
    def normalize_mx(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))  # 矩阵求和
        r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx  # 返回归一化之后的矩阵

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def generate_img_graph(self, img):
        img_feature = img
        adj = img.mm(img.t())
        adj = torch.sign(adj)
        # adj2triad = sp.csr_matrix(adj.cpu().numpy())
        # adj2triad = sp.csr_matrix(adj.detach().numpy())
        adj2triad = sp.csr_matrix(adj.cpu().detach().numpy())
        adj2triad = adj2triad + adj2triad.T.multiply(adj2triad.T > adj2triad) - adj2triad.multiply(
            adj2triad.T > adj2triad)
        adj = self.normalize_mx(adj2triad + sp.eye(adj2triad.shape[0]))
        adjacencyMatrix = self.sparse_mx_to_torch_sparse_tensor(adj)
        adjacencyMatrix = adjacencyMatrix.cuda()
        return img_feature, adjacencyMatrix