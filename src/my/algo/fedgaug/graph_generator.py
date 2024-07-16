import torch
import torch.nn.functional as F


class GraphGenerator():
    def __init__(self,thresh=0, sim_type='correlation', set_negative='hard'):
        '''
        construct graph by similarity
        Args:
            thresh: similarity threshold for constructing edges of graph
            sim_type: similarity measure
            set_negative: method of handling negative similarity
        '''
        self.thresh = thresh
        self.sim = sim_type
        self.set_negative  = set_negative

    @staticmethod
    def set_negative_to_zero(W):
        return F.relu(W)

    def set_negative_to_zero_soft(self, W):
        """ It shifts the negative probabilities towards the positive regime """
        n = W.shape[0]
        minimum = torch.min(W)
        W = W - minimum
        W = W * (torch.ones((n, n)).to(W) - torch.eye(n).to(W))
        return W

    def _get_A(self, W):
        if self.thresh != 'no':
            W  = torch.where(W > self.thresh, W, torch.tensor(0).float().to(W))
            A = torch.ones_like(W).where(W > self.thresh, torch.tensor(0).float().to(W))
        else:
            A = torch.ones_like(W)

        return W, A

    def _get_W(self, x):
        if self.sim == 'correlation':
            x = (x - x.mean(dim=1).unsqueeze(1))
            norms = x.norm(dim=1)
            W = torch.mm(x, x.t()) / torch.ger(norms, norms)
        elif self.sim == 'cosine':
            W = torch.mm(x, x.t())
        elif self.sim == 'learnt':
            n = x.shape[0]
            W = torch.zeros(n, n)
            for i, xi in enumerate(x):
                for j, xj in enumerate(x[(i + 1):], i + 1):
                    W[i, j] = W[j, i] = self.sim(xi, xj) + 1e-8
            W = W.cuda()

        if self.set_negative == 'hard':
            W = self.set_negative_to_zero(W.to(x))
        else:
            W = self.set_negative_to_zero_soft(W)

        return W

    def get_graph(self, x):
        W = self._get_W(x) # sim matrix
        W, A = self._get_A(W) # adj matrix

        A = torch.nonzero(A)
        W_line = W[A[:, 0], A[:, 1]]

        return W_line, A, x#,W
