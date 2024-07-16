import torch
from torch import nn
import torch.nn.functional as F
from my.algo.fedorcheG.functional import off_diagonal
import contextlib
import numpy as np


class WeightedBCE(nn.Module):

    def __init__(self, num_classes, eps=1e-12, device='cpu'):
        super(WeightedBCE, self).__init__()
        self.eps = eps
        self.one_hot_table = nn.functional.one_hot(torch.arange(0, num_classes), num_classes=num_classes).to(device)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets, weights):
        targets = self.one_hot_table[targets]
        log_probs_pos = torch.log(inputs + self.eps)
        log_probs_neg = torch.log(1 - inputs + self.eps)
        loss1 = - targets * log_probs_pos
        loss2 = -(1 - targets) * log_probs_neg
        loss3 = loss1 + loss2
        loss4 = loss3.mean(1)
        loss5 = weights * loss4
        loss = loss5.mean()

        return loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(models):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    for model in models:
        model.apply(switch_attr)
    yield
    for model in models:
        model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=2, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        tx = x.detach()
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(tx.shape).sub(0.5).to(tx.device)
        d = F.normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(tx + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = F.normalize(d.grad)
                model.zero_grad()
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        return lds


def mixup_data(x1, x2, alpha=1.):
    if alpha > 0:
        lambd = np.random.beta(alpha, alpha)
    else:
        lambd = 1.

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(x1.device)
    mixed_x1 = lambd * x1 + (1 - lambd) * x1[index, :]
    mixed_x2 = lambd * x2 + (1 - lambd) * x2[index, :]
    # y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2


class MixupLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(MixupLoss, self).__init__()
        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss()

    def forward(self, mixed_preds, y1, y2, lambd):
        # mixed_x, y_a, y_b, lambd = mixup_data(x,y, self.alpha)
        return lambd * self.loss(mixed_preds, y1) + (1 - lambd) * self.loss(mixed_preds, y2)


class FedGSim(nn.Module):
    def __init__(self, N_centroids, N_local, encoder, graph_generator, gnn, projector, deg_layer, temperature, device,
                 output_train_gnn, top_k_neighbor=1):
        super(FedGSim, self).__init__()
        self.N_centroids = N_centroids
        self.N_local = N_local
        self.online_encoder = encoder
        self.graph_generator = graph_generator
        self.gnn = gnn
        self.online_projector = projector
        self.deg_layer = deg_layer
        self.device = device
        self.temperature = temperature
        self.L_match = nn.KLDivLoss()
        self.L_pseudo = WeightedBCE(self.N_centroids, device=device)
        self.L_ce = nn.CrossEntropyLoss()
        self.L_kl = KLDivWith2LogSM()
        self.L_vat = VATLoss(xi=0.01, eps=2, ip=10)
        self.bn = nn.BatchNorm1d(512, affine=False)
        self.output_train_gnn = output_train_gnn
        self.top_k_neighbor = top_k_neighbor

    def vat_loss(self, x, logp_pred, xi=0.08, eps=2., ip=1):
        tx = x.detach()
        # prepare random unit tensor
        d = torch.rand(tx.shape).sub(0.5).to(tx.device)
        d = F.normalize(d)

        with _disable_tracking_bn_stats([self.online_encoder, self.online_projector]):
            # calc adversarial direction
            for _ in range(ip):
                d.requires_grad_()
                pred_hat = F.normalize(self.online_projector(self.online_encoder(tx + xi * d)), dim=1)
                logp_hat = F.log_softmax(pred_hat / self.temperature, dim=1)
                adv_distance = F.kl_div(logp_hat, logp_pred, reduction='batchmean')
                adv_distance.backward()
                d = F.normalize(d.grad)
                self.online_encoder.zero_grad()
                self.online_projector.zero_grad()
            # calc LDS
            r_adv = d * eps
            pred_hat = F.normalize(self.online_projector(self.online_encoder(x + r_adv)), dim=1)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, logp_pred, reduction='batchmean')
        return lds

    # Alignment / Uniformity (adapted from Wang and Isola, ICML 2020)
    def hyper_uni_align_loss(self, z1, z2, t=1, temperature=0.1):
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)

        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[~diag].view(2 * N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)  # scalar label per sample
        loss = F.cross_entropy(logits, labels, reduction='sum')

        return loss / (2 * N)

    def info_nce_loss(self, Z1, Z2, n_views=2):
        N_BS = Z1.shape[0]
        features = torch.cat((Z1, Z2))
        labels = torch.cat([torch.arange(N_BS) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def forward_rgc(self, centroids, local_centroids, x1, x2, x3=None, deg_labels=None):
        N_BS = x1.shape[0]
        C = centroids.weight.data.detach().clone().T
        C_local = local_centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        Z1 = self.online_encoder(x1)
        #### note rewrite for graph augmentation #####
        Z1_edge_attr, Z1_edge_index, Z1_aug, Z1_W = self.graph_generator.get_graph(Z1)
        Z1_top_k_neighbors_weights, Z1_top_k_indices = torch.topk(Z1_W, k=self.top_k_neighbor + 1,
                                                                  dim=1)  # 基于similarity matrix 得到最近邻
        Z1_top_k_neighbors_weights, Z1_top_k_indices = Z1_top_k_neighbors_weights[:, 1:], Z1_top_k_indices[:, 1:]
        preds, Z1_aug = self.gnn(Z1_aug, Z1_edge_index, Z1_edge_attr, self.output_train_gnn)
        Z1_aug = Z1_aug[-1]  # careful
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z1 = F.normalize(Z1, dim=1)

        Z2 = self.online_encoder(x2)
        # #### note rewrite for graph augmentation #####
        Z2_edge_attr, Z2_edge_index, Z2_aug, Z2_W = self.graph_generator.get_graph(Z2)
        Z2_top_k_neighbors_weights, Z2_top_k_indices = torch.topk(Z2_W, k=self.top_k_neighbor + 1,
                                                                  dim=1)  # 基于similarity matrix 得到最近邻
        Z2_top_k_neighbors_weights, Z2_top_k_indices = Z2_top_k_neighbors_weights[:, 1:], Z2_top_k_indices[:, 1:]
        preds, Z2_aug = self.gnn(Z2_aug, Z2_edge_index, Z2_edge_attr, self.output_train_gnn)
        Z2_aug = Z2_aug[-1]  # careful
        Z2 = F.normalize(Z2, dim=1)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        # Z1_top_k_features = [Z1[Z1_top_k_indices[:, i]] for i in range(self.top_k_neighbor)]
        # Z2_top_k_features = [Z2[Z2_top_k_indices[:, i]] for i in range(self.top_k_neighbor)]
        # Z_top_k_features =Z1_top_k_features+Z2_top_k_features
        # Z_top_k_features=torch.cat(Z_top_k_features, dim=0)

        # L_match= self.hyper_uni_align_loss(Z1, Z2, temperature=0.2)+ self.hyper_uni_align_loss(Z1_aug, Z2_aug, temperature=0.2)
        # features = torch.cat([Z1, Z2], dim=0)  # N_BS*N_views*F_dim
        # logits_rgc, label_rgc = self.info_nce_loss(features, Z_top_k_features,2)
        # L_rgc= self.L_ce(logits_rgc, label_rgc)

        # Z1_logits, Z1_label = self.info_nce_loss(Z1, Z2_aug, 2)
        # Z2_logits, Z2_label = self.info_nce_loss(Z2, Z1_aug, 2)
        # L_match = self.L_ce(Z1_logits, Z1_label) + self.L_ce(Z2_logits, Z2_label)
        #
        # Z_logits, Z_label = self.info_nce_loss(Z1, Z2, 2)
        Zaug_logits, Zaug_label = self.info_nce_loss(Z1_aug, Z2_aug, 2)
        # L_match = self.L_ce(Z_logits, Z_label) + \
        L_match = self.L_ce(Zaug_logits, Zaug_label)

        Z1_FE, Z2_FE = Z1.clone().detach().cpu(), Z2.clone().detach().cpu()
        with torch.no_grad():
            # Compute target model's assignments
            Z1 = self.online_projector(Z1.clone())
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / self.temperature, dim=1)
            cP1_local = Z1 @ C_local
            tP1_local = F.softmax(cP1_local / self.temperature, dim=1)

            Z2 = self.online_projector(Z2.clone())
            cP2 = Z2 @ C
            tP2 = F.softmax(cP2 / self.temperature, dim=1)
            cP2_local = Z2 @ C_local
            tP2_local = F.softmax(cP2_local / self.temperature, dim=1)

            # vat loss
            logp_pred_Z1 = F.softmax(F.normalize(Z1 / self.temperature, dim=1), dim=1)
            logp_pred_Z2 = F.softmax(F.normalize(Z2 / self.temperature, dim=1), dim=1)

        # vat loss
        L_vat = self.vat_loss(x1, logp_pred_Z1) + self.vat_loss(x2, logp_pred_Z2)

        Z1_aug = self.online_projector(Z1_aug)
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z2_aug = self.online_projector(Z2_aug)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        # SK_Z1_aug_assigns = sknopp(self.local_centroids(Z1_aug), max_iters=10)
        # SK_Z2_aug_assigns = sknopp(self.local_centroids(Z2_aug), max_iters=10)
        # L_match= torch.norm(SK_Z1_aug_assigns-SK_Z2_aug_assigns, p=2, dim=1).mean()
        # L_match= -torch.sum(SK_Z1_aug_assigns*torch.log(tP2), dim=1).mean() - torch.sum(SK_Z2_aug_assigns*torch.log(tP1), dim=1).mean()
        # Convert to log-probabilities
        cZ1_aug = Z1_aug @ C
        cZ2_aug = Z2_aug @ C
        cZ1_aug_local = Z1_aug @ C_local
        cZ2_aug_local = Z2_aug @ C_local
        logpZ1_aug = torch.log(F.softmax(cZ1_aug / self.temperature, dim=1))
        logpZ2_aug = torch.log(F.softmax(cZ2_aug / self.temperature, dim=1))
        logpZ1_aug_local = torch.log(F.softmax(cZ1_aug_local / self.temperature, dim=1))
        logpZ2_aug_local = torch.log(F.softmax(cZ2_aug_local / self.temperature, dim=1))
        # tpZ1_aug = F.softmax(cZ1_aug / self.temperature, dim=1)
        # tpZ2_aug = F.softmax(cZ2_aug / self.temperature, dim=1)
        # tpZ1_aug_local = F.softmax(cZ1_aug_local / self.temperature, dim=1)
        # tpZ2_aug_local = F.softmax(cZ2_aug_local / self.temperature, dim=1)
        # L_cluster = torch.norm(tP2- tpZ1_aug, p=2, dim=1).mean() + torch.norm(tP1-tpZ2_aug, p=2, dim=1).mean()
        L_cluster = -torch.sum(tP2 * logpZ1_aug, dim=1).mean() - torch.sum(tP1 * logpZ2_aug, dim=1).mean() - torch.sum(
            tP2_local * logpZ1_aug_local, dim=1).mean() - torch.sum(tP1_local * logpZ2_aug_local, dim=1).mean()
        # L_cluster =- torch.sum(tP1 * logpZ2_aug, dim=1).mean()
        #
        # Degeneracy regularization
        deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_deg + 5 * L_match + L_cluster + 100 * L_vat  # debug +match loss
        # L= L_match

        return L, Z1_FE, Z2_FE

    def forward_vat(self, centroids, local_centroids, x1, x2, x3=None, deg_labels=None):
        N_BS = x1.shape[0]
        C = centroids.weight.data.detach().clone().T
        C_local = local_centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        Z1 = self.online_encoder(x1)
        #### note rewrite for graph augmentation #####
        Z1_edge_attr, Z1_edge_index, Z1_aug = self.graph_generator.get_graph(Z1)
        preds, Z1_aug = self.gnn(Z1_aug, Z1_edge_index, Z1_edge_attr, self.output_train_gnn)
        Z1_aug = Z1_aug[-1]  # careful
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z1 = F.normalize(Z1, dim=1)

        Z2 = self.online_encoder(x2)
        # #### note rewrite for graph augmentation #####
        Z2_edge_attr, Z2_edge_index, Z2_aug = self.graph_generator.get_graph(Z2)

        preds, Z2_aug = self.gnn(Z2_aug, Z2_edge_index, Z2_edge_attr, self.output_train_gnn)
        Z2_aug = Z2_aug[-1]  # careful
        Z2 = F.normalize(Z2, dim=1)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        # Z1_top_k_features = [Z1[Z1_top_k_indices[:, i]] for i in range(self.top_k_neighbor)]
        # Z2_top_k_features = [Z2[Z2_top_k_indices[:, i]] for i in range(self.top_k_neighbor)]
        # Z_top_k_features =Z1_top_k_features+Z2_top_k_features
        # Z_top_k_features=torch.cat(Z_top_k_features, dim=0)

        # L_match= self.hyper_uni_align_loss(Z1, Z2, temperature=0.2)+ self.hyper_uni_align_loss(Z1_aug, Z2_aug, temperature=0.2)
        # features = torch.cat([Z1, Z2], dim=0)  # N_BS*N_views*F_dim
        # logits_rgc, label_rgc = self.info_nce_loss(features, Z_top_k_features,2)
        # L_rgc= self.L_ce(logits_rgc, label_rgc)

        # Z1_logits, Z1_label = self.info_nce_loss(Z1, Z2_aug, 2)
        # Z2_logits, Z2_label = self.info_nce_loss(Z2, Z1_aug, 2)
        # L_match = self.L_ce(Z1_logits, Z1_label) + self.L_ce(Z2_logits, Z2_label)
        #
        # Z_logits, Z_label = self.info_nce_loss(Z1, Z2, 2)
        Zaug_logits, Zaug_label = self.info_nce_loss(Z1_aug, Z2_aug, 2)
        # L_match = self.L_ce(Z_logits, Z_label) + \
        L_match = self.L_ce(Zaug_logits, Zaug_label)

        Z1_FE, Z2_FE = Z1.clone().detach().cpu(), Z2.clone().detach().cpu()
        with torch.no_grad():
            # Compute target model's assignments
            Z1 = self.online_projector(Z1.clone())
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / self.temperature, dim=1)
            cP1_local = Z1 @ C_local
            tP1_local = F.softmax(cP1_local / self.temperature, dim=1)

            Z2 = self.online_projector(Z2.clone())
            cP2 = Z2 @ C
            tP2 = F.softmax(cP2 / self.temperature, dim=1)
            cP2_local = Z2 @ C_local
            tP2_local = F.softmax(cP2_local / self.temperature, dim=1)

            # vat loss
            logp_pred_Z1 = F.softmax(F.normalize(Z1 / self.temperature, dim=1), dim=1)
            logp_pred_Z2 = F.softmax(F.normalize(Z2 / self.temperature, dim=1), dim=1)

        # vat loss
        L_vat = self.vat_loss(x1, logp_pred_Z1, xi=1e-6, eps=6., ip=1) + self.vat_loss(x2, logp_pred_Z2, xi=1e-6,
                                                                                       eps=6., ip=1)

        Z1_aug = self.online_projector(Z1_aug)
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z2_aug = self.online_projector(Z2_aug)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        # SK_Z1_aug_assigns = sknopp(self.local_centroids(Z1_aug), max_iters=10)
        # SK_Z2_aug_assigns = sknopp(self.local_centroids(Z2_aug), max_iters=10)
        # L_match= torch.norm(SK_Z1_aug_assigns-SK_Z2_aug_assigns, p=2, dim=1).mean()
        # L_match= -torch.sum(SK_Z1_aug_assigns*torch.log(tP2), dim=1).mean() - torch.sum(SK_Z2_aug_assigns*torch.log(tP1), dim=1).mean()
        # Convert to log-probabilities
        cZ1_aug = Z1_aug @ C
        cZ2_aug = Z2_aug @ C
        cZ1_aug_local = Z1_aug @ C_local
        cZ2_aug_local = Z2_aug @ C_local
        logpZ1_aug = torch.log(F.softmax(cZ1_aug / self.temperature, dim=1))
        logpZ2_aug = torch.log(F.softmax(cZ2_aug / self.temperature, dim=1))
        logpZ1_aug_local = torch.log(F.softmax(cZ1_aug_local / self.temperature, dim=1))
        logpZ2_aug_local = torch.log(F.softmax(cZ2_aug_local / self.temperature, dim=1))
        # tpZ1_aug = F.softmax(cZ1_aug / self.temperature, dim=1)
        # tpZ2_aug = F.softmax(cZ2_aug / self.temperature, dim=1)
        # tpZ1_aug_local = F.softmax(cZ1_aug_local / self.temperature, dim=1)
        # tpZ2_aug_local = F.softmax(cZ2_aug_local / self.temperature, dim=1)
        # L_cluster = torch.norm(tP2- tpZ1_aug, p=2, dim=1).mean() + torch.norm(tP1-tpZ2_aug, p=2, dim=1).mean()
        L_cluster = -torch.sum(tP2 * logpZ1_aug, dim=1).mean() - torch.sum(tP1 * logpZ2_aug, dim=1).mean() - torch.sum(
            tP2_local * logpZ1_aug_local, dim=1).mean() - torch.sum(tP1_local * logpZ2_aug_local, dim=1).mean()
        # L_cluster =- torch.sum(tP1 * logpZ2_aug, dim=1).mean()
        #
        # Degeneracy regularization
        deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_deg + 5 * L_match + L_cluster + 100 * L_vat  # debug +match loss
        # L= L_match

        return L, Z1_FE, Z2_FE

    def forward_mixup(self, centroids, local_centroids, x1, x2, x3=None, deg_labels=None):
        x1_mix,x2_mix= mixup_data(x1, x2, alpha=0.5)
        N_BS = x1.shape[0]
        C = centroids.weight.data.detach().clone().T
        C_local = local_centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        Z1 = self.online_encoder(x1)
        #### note rewrite for graph augmentation #####
        Z1_edge_attr, Z1_edge_index, Z1_aug = self.graph_generator.get_graph(Z1)
        preds, Z1_aug = self.gnn(Z1_aug, Z1_edge_index, Z1_edge_attr, self.output_train_gnn)
        Z1_aug = Z1_aug[-1]  # careful
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z1 = F.normalize(Z1, dim=1)

        Z2 = self.online_encoder(x2)
        # #### note rewrite for graph augmentation #####
        Z2_edge_attr, Z2_edge_index, Z2_aug = self.graph_generator.get_graph(Z2)
        preds, Z2_aug = self.gnn(Z2_aug, Z2_edge_index, Z2_edge_attr, self.output_train_gnn)
        Z2_aug = Z2_aug[-1]  # careful
        Z2 = F.normalize(Z2, dim=1)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        # for mixup data
        Z1_mix = self.online_encoder(x1_mix)
        Z1_mix_edge_attr, Z1_mix_edge_index, Z1_mix_aug = self.graph_generator.get_graph(Z1_mix)
        preds, Z1_mix_aug = self.gnn(Z1_mix_aug, Z1_mix_edge_index, Z1_mix_edge_attr, self.output_train_gnn)
        Z1_mix_aug = Z1_mix_aug[-1]  # careful
        Z1_mix = F.normalize(Z1_mix, dim=1)
        Z1_mix_aug = F.normalize(Z1_mix_aug, dim=1)

        Z2_mix = self.online_encoder(x2_mix)
        Z2_mix_edge_attr, Z2_mix_edge_index, Z2_mix_aug = self.graph_generator.get_graph(Z2_mix)
        preds, Z2_mix_aug = self.gnn(Z2_mix_aug, Z2_mix_edge_index, Z2_mix_edge_attr, self.output_train_gnn)
        Z2_mix_aug = Z2_mix_aug[-1]  # careful
        Z2_mix = F.normalize(Z2_mix, dim=1)
        Z2_mix_aug = F.normalize(Z2_mix_aug, dim=1)

        # Z1_top_k_features = [Z1[Z1_top_k_indices[:, i]] for i in range(self.top_k_neighbor)]
        # Z2_top_k_features = [Z2[Z2_top_k_indices[:, i]] for i in range(self.top_k_neighbor)]
        # Z_top_k_features =Z1_top_k_features+Z2_top_k_features
        # Z_top_k_features=torch.cat(Z_top_k_features, dim=0)

        # L_match= self.hyper_uni_align_loss(Z1, Z2, temperature=0.2)+ self.hyper_uni_align_loss(Z1_aug, Z2_aug, temperature=0.2)
        # features = torch.cat([Z1, Z2], dim=0)  # N_BS*N_views*F_dim
        # logits_rgc, label_rgc = self.info_nce_loss(features, Z_top_k_features,2)
        # L_rgc= self.L_ce(logits_rgc, label_rgc)

        # Z1_logits, Z1_label = self.info_nce_loss(Z1, Z2_aug, 2)
        # Z2_logits, Z2_label = self.info_nce_loss(Z2, Z1_aug, 2)
        # L_match = self.L_ce(Z1_logits, Z1_label) + self.L_ce(Z2_logits, Z2_label)
        #
        # Z_logits, Z_label = self.info_nce_loss(Z1, Z2, 2)
        Zaug_logits, Zaug_label = self.info_nce_loss(Z1_aug, Z2_aug, 2)
        Z_mix_aug_logits, Z_mix_aug_label = self.info_nce_loss(Z1_mix_aug, Z2_mix_aug, 2)
        # L_match = self.L_ce(Z_logits, Z_label)
        L_match = self.L_ce(Zaug_logits, Zaug_label) + self.L_ce(Z_mix_aug_logits, Z_mix_aug_label)

        Z1_FE, Z2_FE = Z1.clone().detach().cpu(), Z2.clone().detach().cpu()
        with torch.no_grad():
            # Compute target model's assignments
            Z1 = self.online_projector(Z1.clone())
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / self.temperature, dim=1)
            cP1_local = Z1 @ C_local
            tP1_local = F.softmax(cP1_local / self.temperature, dim=1)

            Z2 = self.online_projector(Z2.clone())
            cP2 = Z2 @ C
            tP2 = F.softmax(cP2 / self.temperature, dim=1)
            cP2_local = Z2 @ C_local
            tP2_local = F.softmax(cP2_local / self.temperature, dim=1)

            # vat loss
            # logp_pred_Z1 = F.softmax(F.normalize(Z1 / self.temperature, dim=1), dim=1)
            # logp_pred_Z2 = F.softmax(F.normalize(Z2 / self.temperature, dim=1), dim=1)

        # vat loss
        # L_vat= self.vat_loss(x1, logp_pred_Z1, xi= 1e-6, eps=6., ip=1) + self.vat_loss(x2, logp_pred_Z2, xi= 1e-6, eps=6., ip=1)

        Z1_aug = self.online_projector(Z1_aug)
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z2_aug = self.online_projector(Z2_aug)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        # SK_Z1_aug_assigns = sknopp(self.local_centroids(Z1_aug), max_iters=10)
        # SK_Z2_aug_assigns = sknopp(self.local_centroids(Z2_aug), max_iters=10)
        # L_match= torch.norm(SK_Z1_aug_assigns-SK_Z2_aug_assigns, p=2, dim=1).mean()
        # L_match= -torch.sum(SK_Z1_aug_assigns*torch.log(tP2), dim=1).mean() - torch.sum(SK_Z2_aug_assigns*torch.log(tP1), dim=1).mean()
        # Convert to log-probabilities
        cZ1_aug = Z1_aug @ C
        cZ2_aug = Z2_aug @ C
        cZ1_aug_local = Z1_aug @ C_local
        cZ2_aug_local = Z2_aug @ C_local
        logpZ1_aug = torch.log(F.softmax(cZ1_aug / self.temperature, dim=1))
        logpZ2_aug = torch.log(F.softmax(cZ2_aug / self.temperature, dim=1))
        logpZ1_aug_local = torch.log(F.softmax(cZ1_aug_local / self.temperature, dim=1))
        logpZ2_aug_local = torch.log(F.softmax(cZ2_aug_local / self.temperature, dim=1))
        # tpZ1_aug = F.softmax(cZ1_aug / self.temperature, dim=1)
        # tpZ2_aug = F.softmax(cZ2_aug / self.temperature, dim=1)
        # tpZ1_aug_local = F.softmax(cZ1_aug_local / self.temperature, dim=1)
        # tpZ2_aug_local = F.softmax(cZ2_aug_local / self.temperature, dim=1)
        # L_cluster = torch.norm(tP2- tpZ1_aug, p=2, dim=1).mean() + torch.norm(tP1-tpZ2_aug, p=2, dim=1).mean()
        L_cluster = -torch.sum(tP2 * logpZ1_aug, dim=1).mean() - torch.sum(tP1 * logpZ2_aug, dim=1).mean() - torch.sum(
            tP2_local * logpZ1_aug_local, dim=1).mean() - torch.sum(tP1_local * logpZ2_aug_local, dim=1).mean()
        # L_cluster =- torch.sum(tP1 * logpZ2_aug, dim=1).mean()
        #
        # Degeneracy regularization
        deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_deg + 5 * L_match + L_cluster #+ 100 * L_vat  # debug +match loss
        # L= L_match

        return L, Z1_FE, Z2_FE

    def forward(self, centroids, x1, x2, x3=None, deg_labels=None):
        N_BS = x1.shape[0]
        C = centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        Z1 = self.online_encoder(x1)
        #### note rewrite for graph augmentation #####
        edge_attr, edge_index, Z1_aug = self.graph_generator.get_graph(Z1)
        preds, Z1_aug = self.gnn(Z1_aug, edge_index, edge_attr, self.output_train_gnn)
        Z1_aug = Z1_aug[-1]  # careful
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z1 = F.normalize(Z1, dim=1)

        Z2 = self.online_encoder(x2)
        # #### note rewrite for graph augmentation #####
        edge_attr, edge_index, Z2_aug = self.graph_generator.get_graph(Z2)
        preds, Z2_aug = self.gnn(Z2_aug, edge_index, edge_attr, self.output_train_gnn)
        Z2_aug = Z2_aug[-1]  # careful
        Z2 = F.normalize(Z2, dim=1)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        logits, label = self.info_nce_loss(self.online_projector(Z1_aug), self.online_projector(Z2_aug), 2)
        L_match = self.L_ce(logits, label)

        with torch.no_grad():
            # Compute target model's assignments
            Z1 = self.online_projector(Z1)
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / self.temperature, dim=1)

            Z2 = self.online_projector(Z2)
            cP2 = Z2 @ C
            tP2 = F.softmax(cP2 / self.temperature, dim=1)

        Z1_aug = self.online_projector(Z1_aug)
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z2_aug = self.online_projector(Z2_aug)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        # SK_Z1_aug_assigns = sknopp(self.local_centroids(Z1_aug), max_iters=10)
        # SK_Z2_aug_assigns = sknopp(self.local_centroids(Z2_aug), max_iters=10)
        # L_match= torch.norm(SK_Z1_aug_assigns-SK_Z2_aug_assigns, p=2, dim=1).mean()
        # L_match= -torch.sum(SK_Z1_aug_assigns*torch.log(tP2), dim=1).mean() - torch.sum(SK_Z2_aug_assigns*torch.log(tP1), dim=1).mean()
        # Convert to log-probabilities
        cZ1_aug = Z1_aug @ C
        cZ2_aug = Z2_aug @ C
        logpZ1_aug = torch.log(F.softmax(cZ1_aug / self.temperature, dim=1))
        logpZ2_aug = torch.log(F.softmax(cZ2_aug / self.temperature, dim=1))
        L_cluster = -torch.sum(tP2 * logpZ1_aug, dim=1).mean() - torch.sum(tP1 * logpZ2_aug, dim=1).mean()

        # Degeneracy regularization
        # deg_preds1 = self.deg_layer(self.online_projector(Z1_aug))
        # deg_preds2 = self.deg_layer(self.online_projector(Z2_aug))
        deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_deg + L_cluster + L_match  # debug +match loss

        return L, Z1

    def forward_swav(self, centroids, x1, x2, x3=None, deg_labels=None):
        N_BS = x1.shape[0]
        C = centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        Z1 = self.online_encoder(x1)
        #### note rewrite for graph augmentation #####
        edge_attr, edge_index, Z1_aug = self.graph_generator.get_graph(Z1)
        preds, Z1_aug = self.gnn(Z1_aug, edge_index, edge_attr, self.output_train_gnn)
        Z1_aug = Z1_aug[-1]  # careful
        Z1_aug = F.normalize(Z1_aug, dim=1)
        Z1 = F.normalize(Z1, dim=1)

        Z2 = self.online_encoder(x2)
        # #### note rewrite for graph augmentation #####
        edge_attr, edge_index, Z2_aug = self.graph_generator.get_graph(Z2)
        preds, Z2_aug = self.gnn(Z2_aug, edge_index, edge_attr, self.output_train_gnn)
        Z2_aug = Z2_aug[-1]  # careful
        Z2 = F.normalize(Z2, dim=1)
        Z2_aug = F.normalize(Z2_aug, dim=1)

        with torch.no_grad():
            # Compute target model's assignments
            Z1 = self.online_projector(Z1)
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / self.temperature, dim=1)
            Z2 = self.online_projector(Z2)
            cP2 = Z2 @ C
            tP2 = F.softmax(cP2 / self.temperature, dim=1)

        Z1_aug = self.online_projector(Z1_aug)
        Z2_aug = self.online_projector(Z2_aug)
        SK_Z1_aug_assigns = sknopp(self.local_centroids(Z1_aug), max_iters=10)
        SK_Z2_aug_assigns = sknopp(self.local_centroids(Z2_aug), max_iters=10)
        L_match = torch.norm(SK_Z1_aug_assigns - SK_Z2_aug_assigns, p=2, dim=1).mean()
        # L_match= -torch.sum(SK_Z1_aug_assigns*torch.log(tP2), dim=1).mean() - torch.sum(SK_Z2_aug_assigns*torch.log(tP1), dim=1).mean()
        # Convert to log-probabilities
        cZ1_aug = Z1_aug @ C
        cZ2_aug = Z2_aug @ C
        logpZ1_aug = torch.log(F.softmax(cZ1_aug / self.temperature, dim=1))
        logpZ2_aug = torch.log(F.softmax(cZ2_aug / self.temperature, dim=1))
        L_cluster = -torch.sum(tP2 * logpZ1_aug, dim=1).mean() - torch.sum(tP1 * logpZ2_aug, dim=1).mean()

        # Degeneracy regularization
        # deg_preds1 = self.deg_layer(self.online_projector(Z1_aug))
        # deg_preds2 = self.deg_layer(self.online_projector(Z2_aug))
        deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_deg + L_cluster + L_match  # debug +match loss

        return L, Z1

    def forward_raw(self, centroids, x1, x2, x3=None, deg_labels=None):
        N_BS = x1.shape[0]
        C = centroids.weight.data.detach().clone().T

        # # Online model's outputs [bsize, D]
        # # Z1 = F.normalize(self.online_projector(self.online_encoder(x1)), dim=1)
        # Z1 = self.online_encoder(x1)
        # Z1 = F.normalize(self.online_projector(Z1), dim=1)
        #
        #
        # #### note rewrite for graph augmentation #####
        # edge_attr, edge_index, Z1_aug = self.graph_generator.get_graph(Z1)
        # preds, Z1_aug = self.gnn(Z1_aug, edge_index, edge_attr, self.output_train_gnn)
        # Z1_aug = Z1_aug[-1]  # careful
        # Z1_aug = F.normalize(self.online_projector(Z1_aug), dim=1)

        # Z2 = F.normalize(self.online_projector(self.online_encoder(x2)), dim=1)
        Z2 = self.online_encoder(x2)

        # Z2 = F.normalize(Z2, dim=1)
        Z2_logits = self.centroids(Z2)
        Z2_logits = F.softmax(Z2_logits, dim=1)  # Z2@self.d2hg.means.T
        weights, pseudo_label = torch.max(Z2_logits, dim=1)
        thres = 0.5
        weights = weights.ge(thres)

        # #### note rewrite for graph augmentation #####
        edge_attr, edge_index, Z2_aug = self.graph_generator.get_graph(Z2)
        preds, Z2_aug = self.gnn(Z2_aug, edge_index, edge_attr, self.output_train_gnn)
        preds = F.softmax(preds[-1], dim=1)
        L_pseudo = self.L_pseudo(preds, pseudo_label, weights)
        # L_pseudo = self.L_pseudo(preds[0], pseudo_label, weights)+self.L_pseudo(preds[-1], pseudo_label, weights)

        Z2_aug = Z2_aug[-1]  # careful
        # Compute online model's assignments
        Z2 = F.normalize(Z2, dim=1)
        Z2_aug = F.normalize(Z2_aug, dim=1)
        # label_Z2 = torch.argmax(self.centroids(Z2), dim=1)
        # label_Z2_aug = torch.argmax(self.centroids(Z2), dim=1)

        # # SK_Z2_assigns = dasknopp(Z2, label_Z2, C.T, reg_e=1, max_iters=10)
        # # SK_Z2_aug_assigns = dasknopp(Z2_aug, label_Z2_aug, C.T, reg_e=1, max_iters=10)
        # SK_Z2_assigns = sknopp(self.local_centroids(Z2), max_iters=10)
        # SK_Z2_aug_assigns = sknopp(self.local_centroids(Z2_aug), max_iters=10)
        # # L_match = torch.norm(SK_Z2_assigns - SK_Z2_aug_assigns, p=2, dim=-1).mean()
        # # L_match =  self.L_match(SK_Z2_assigns.log(), SK_Z2_assigns) + self.L_match(SK_Z2_aug_assigns.log(), SK_Z2_assigns)
        # L_match = torch.norm(SK_Z2_assigns - SK_Z2_aug_assigns, p=2, dim=-1).mean()

        # cZ2 = Z2 @ C
        # cZ2_aug = Z2_aug @ C
        # L_match= self.L_match(cZ2, cZ2_aug) + self.L_match(cZ2, cZ2_aug)

        Z2_aug = F.normalize(self.online_projector(Z2_aug), dim=1)
        Z2 = F.normalize(self.online_projector(Z2), dim=1)

        # Compute online model's assignments
        cZ2 = Z2 @ C
        cZ2_aug = Z2_aug @ C

        # Convert to log-probabilities
        logpZ2 = torch.log(F.softmax(cZ2 / self.temperature, dim=1))
        logpZ2_aug = torch.log(F.softmax(cZ2_aug / self.temperature, dim=1))

        # # Target outputs [bsize, D]
        Z1 = self.online_encoder(x1)
        c = self.bn(Z1).T @ self.bn(Z2)
        c.div_(N_BS)
        # torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).mean()
        L_BT = on_diag + 2. * off_diag

        with torch.no_grad():
            # self.ema_update()
            tZ1 = F.normalize(self.online_projector(Z1), dim=1)
            # tZ1 = F.normalize(self.target_projector(Z1), dim=1)

            # Compute target model's assignments
            cP1 = tZ1 @ C
            tP1 = F.softmax(cP1 / self.temperature, dim=1)

        # Clustering loss
        L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean() - torch.sum(tP1 * logpZ2_aug, dim=1).mean() + L_BT.mean()

        # Degeneracy regularization
        deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_cluster + L_deg + L_pseudo  # + L_match

        return L, Z1


# Projector
class ProjectionMLPOrchestra(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(ProjectionMLPOrchestra, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x)))
        x = self.layer2_bn(self.layer2(x))
        return x


class KLDivWith2LogSM(torch.nn.Module):
    def __init__(self):
        super(KLDivWith2LogSM, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.KLDiv = torch.nn.KLDivLoss()

    def forward(self, sources, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes)
        """
        log_probs_s = self.logsoftmax(sources)
        log_probs_t = self.logsoftmax(targets)
        loss = self.KLDiv(log_probs_s, log_probs_t)
        return loss


@torch.no_grad()
def model_divergence(model1: nn.Module, model2: nn.Module) -> float:
    """Compute the divergence between two models."""
    dict1 = dict(model1.named_parameters())
    dict2 = dict(model2.named_parameters())
    total = 0.0
    count = 0
    for name in dict1.keys():
        if 'conv' in name and 'weight' in name:  # note: only consider conv weights
            total += torch.dist(dict1[name], dict2[name], p=2).cpu()
            count += 1
    # note: The implementation in the author's source code is slightly different from his paper.
    divergence = total / count
    return float(divergence)


############ Orchestra model class ############
# Sinkhorn Knopp
def sknopp(cZ, lamd=25, max_iters=100):
    '''

    Args:
        cZ: cost matrix = feature @ centroids, e.g., Z @centroids.T
        lamd: multiplier for ??? # careful
        max_iters: convex optimization iteration

    Returns:

    '''
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape  # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T  # probs should be [N_centroids, N_samples] note inited prob

        r = torch.ones((N_centroids, 1),
                       device=probs.device) / N_centroids  # desired row sum vector # note inited marginal
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples  # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # note marginal distri (probs @ c): shape= N_centroids *1
            # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T
            # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations.
        probs *= c.squeeze()
        probs = probs.T  # [N_samples, N_centroids]
        probs *= r.squeeze()  # note shape: N_samples * N_centroids

        return probs * N_samples  # Soft assignments
