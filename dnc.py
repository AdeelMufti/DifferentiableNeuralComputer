import os
import re
import argparse
import time

import numpy as np
from chainer import functions as F
from chainer import links as L
from chainer import \
    Chain, Variable, optimizers, serializers
try:
    import cupy as cp
except Exception as e:
    None

xp = np


def overlap(u, v):  # u, v: (1 * -) -> (1 * 1)
    if v.shape[0] != 1:
        u_tup = ()
        for i in range(v.shape[0]):
            u_tup = u_tup + (u,)
        u_repeated = xp.vstack(u_tup)
        v_repeated = xp.repeat(v, u.shape[0], 0)
        denominator = xp.sqrt(xp.sum(u_repeated * u_repeated, 1) * xp.sum(v_repeated * v_repeated, 1))
        denominator = denominator.reshape(v.shape[0],u.shape[0]).T
    else:
        denominator = xp.sqrt(xp.sum(u * u, 1) * xp.sum(v * v, 1)).reshape(-1,1)
    denominator[denominator == 0.] = 1.
    return xp.dot(u, v.T) / denominator


def C(M, k, beta):
    ret_list = overlap(M, k) * beta
    ret_list = ret_list.T
    if ret_list.shape[0] != 1:
        softmax = xp.exp(ret_list - xp.max(ret_list,1).reshape(-1,1))
        softmax = softmax / softmax.sum(1).reshape(-1,1)
    else:
        softmax = xp.exp(ret_list - xp.max(ret_list))
        softmax = softmax / softmax.sum()
    ret_list = softmax.T
    return ret_list


def u2a(u):  # u, a: (N * 1)
    N = len(u)
    phi = xp.argsort(xp.reshape(u,N))  # u[phi]: ascending
    cumprod = xp.cumprod(u[phi])
    cumprod[-1] = 1.
    cumprod = xp.roll(cumprod, 1)
    cumprod = cumprod.reshape(-1,1)
    a_list = xp.zeros((N,1)).astype(xp.float32)
    a_list[phi] = (cumprod * (1.0 - u[phi]))
    return a_list


class DeepLSTM(Chain):
    def __init__(self, lstm_hidden_dim, d_out, gpu_for_nn_only=False):
        self.gpu_for_nn_only = gpu_for_nn_only
        super(DeepLSTM, self).__init__(
            l1=L.LSTM(None, lstm_hidden_dim),
            l2=L.Linear(lstm_hidden_dim, d_out)
        )

    def __call__(self, x):
        if self._device_id is not None and self.gpu_for_nn_only:
            x = F.copy(x, self._device_id)
        y = self.l2(self.l1(x))
        if self._device_id is not None and self.gpu_for_nn_only:
            y = F.copy(y, -1)
        return y

    def reset_state(self):
        self.l1.reset_state()

    def get_h(self):
        return self.l1.h

    def get_c(self):
        return self.l1.c


class Linear(Chain):
    def __init__(self, d_out, gpu_for_nn_only=False):
        self.gpu_for_nn_only = gpu_for_nn_only
        super(Linear, self).__init__(
            l=L.Linear(None, d_out)
        )

    def __call__(self, x):
        if self._device_id is not None and self.gpu_for_nn_only:
            x = F.copy(x, self._device_id)
        y = self.l(x)
        if self._device_id is not None and self.gpu_for_nn_only:
            y = F.copy(y, -1)
        return y


class DNC(Chain):
    def __init__(self, X, Y, N, W, R, lstm_hidden_dim, K=8, gpu_for_nn_only=False):
        self.X = X  # input dimension
        self.Y = Y  # output dimension
        self.N = N  # number of memory slot
        self.W = W  # dimension of one memory slot
        self.R = R  # number of read heads
        self.K = K  # Described under **Methods** of DNC paper, in the *Sparse link matrix* section

        self.xi_split_indices = xp.cumsum(xp.array([self.W * self.R, self.R, self.W, 1, self.W, self.W, self.R, 1, 1])).tolist()

        self.controller = DeepLSTM(lstm_hidden_dim, Y + W * R + 3 * W + 5 * R + 3, gpu_for_nn_only)
        self.linear = Linear(self.Y, gpu_for_nn_only)

        super(DNC, self).__init__(
            l_dl=self.controller,
            l_Wr=self.linear
        )

    def __call__(self, x):
        self.chi = F.concat((x, self.r))
        (self.nu, self.xi) = F.split_axis(self.l_dl(self.chi), [self.Y], 1)

        (self.kr, self.betar, self.kw, self.betaw, self.e, self.v, self.f, self.ga, self.gw, self.pi) = \
            F.split_axis(self.xi, self.xi_split_indices, 1)

        self.kr = F.reshape(self.kr, (self.R, self.W))  # R * W
        self.betar = 1 + F.softplus(self.betar)  # 1 * R
        # self.kw: 1 * W
        self.betaw = 1 + F.softplus(self.betaw)  # 1 * 1
        self.e = F.sigmoid(self.e)  # 1 * W
        # self.v : 1 * W
        self.f = F.sigmoid(self.f)  # 1 * R
        self.ga = F.sigmoid(self.ga)  # 1 * 1
        self.gw = F.sigmoid(self.gw)  # 1 * 1
        self.pi = F.softmax(F.reshape(self.pi, (self.R, 3)))  # R * 3 (softmax for 3)

        # self.wr : N * R
        self.psi_mat = 1 - F.broadcast_to(self.f,(self.N,self.R)) * self.wr  # N x R
        self.psi = F.prod(self.psi_mat, 1).reshape(self.N, 1) # N x 1

        # self.ww, self.u : N * 1
        self.u = (self.u + self.ww - (self.u * self.ww)) * self.psi

        self.a = u2a(self.u.data)  # N * 1
        self.cw = C(self.M.data, self.kw.data, self.betaw.data)  # N * 1
        self.ww = F.matmul(F.matmul(self.a, self.ga) + F.matmul(self.cw, 1.0 - self.ga), self.gw)  # N * 1
        self.M = self.M * (xp.ones((self.N, self.W)).astype(xp.float32) - F.matmul(self.ww, self.e)) + F.matmul(self.ww,
                                                                                                                self.v)  # N * W
        if self.K > 0:
            self.p = (1.0 - F.matmul(Variable(xp.ones((self.N, 1)).astype(xp.float32)), F.reshape(F.sum(self.ww), (1, 1)))) \
                     * self.p + self.ww  # N * 1
            self.p.data = xp.sort(self.p.data,0)
            self.p.data[0:-self.K] = 0.
            self.p.data[-self.K:] = self.p.data[-self.K:]/xp.sum(self.p.data[-self.K:])
            self.ww.data = xp.sort(self.ww.data,0)
            self.ww.data[0:-self.K] = 0.
            self.ww.data[-self.K:] = self.ww[-self.K:].data/xp.sum(self.ww.data[-self.K:])
            self.wwrep = F.matmul(self.ww, Variable(xp.ones((1, self.N)).astype(xp.float32)))  # N * N
            self.ww_p_product = xp.zeros((self.N,self.N)).astype(xp.float32)
            self.ww_p_product[-self.K:,-self.K:] = F.matmul(self.ww[-self.K:,-self.K:], F.transpose(self.p[-self.K:,-self.K:])).data
            self.L = (1.0 - self.wwrep - F.transpose(self.wwrep)) * self.L + self.ww_p_product  # N * N
            self.L = self.L * (xp.ones((self.N, self.N)) - xp.eye(self.N))  # force L[i,i] == 0
            self.L.data[self.L.data < 1/self.K] = 0.
        else:
            self.p = (1.0 - F.matmul(Variable(xp.ones((self.N, 1)).astype(xp.float32)),
                                     F.reshape(F.sum(self.ww), (1, 1)))) \
                     * self.p + self.ww  # N * 1
            self.wwrep = F.matmul(self.ww, Variable(xp.ones((1, self.N)).astype(xp.float32)))  # N * N
            self.L = (1.0 - self.wwrep - F.transpose(self.wwrep)) * self.L + F.matmul(self.ww,
                                                                                      F.transpose(self.p))  # N * N
            self.L = self.L * (xp.ones((self.N, self.N)) - xp.eye(self.N))  # force L[i,i] == 0
        self.fo = F.matmul(self.L, self.wr)  # N * R
        self.ba = F.matmul(F.transpose(self.L), self.wr)  # N * R

        self.cr = C(self.M.data, self.kr.data, self.betar.data)

        self.bacrfo = F.concat((F.reshape(F.transpose(self.ba), (self.R, self.N, 1)),
                                F.reshape(F.transpose(self.cr), (self.R, self.N, 1)),
                                F.reshape(F.transpose(self.fo), (self.R, self.N, 1)),), 2)  # R * N * 3
        self.pi = F.reshape(self.pi, (self.R, 3, 1))  # R * 3 * 1
        self.wr = F.transpose(F.reshape(F.batch_matmul(self.bacrfo, self.pi), (self.R, self.N)))  # N * R

        self.r = F.reshape(F.matmul(F.transpose(self.M), self.wr), (1, self.R * self.W))  # W * R (-> 1 * RW)

        self.y = self.l_Wr(self.r) + self.nu  # 1 * Y
        return self.y

    def reset_state(self):
        self.l_dl.reset_state()
        self.u = Variable(xp.zeros((self.N, 1)).astype(xp.float32))
        self.p = Variable(xp.zeros((self.N, 1)).astype(xp.float32))
        self.L = Variable(xp.zeros((self.N, self.N)).astype(xp.float32))
        self.M = Variable(xp.zeros((self.N, self.W)).astype(xp.float32))
        self.r = Variable(xp.zeros((1, self.R * self.W)).astype(xp.float32))
        self.wr = Variable(xp.zeros((self.N, self.R)).astype(xp.float32))
        self.ww = Variable(xp.zeros((self.N, 1)).astype(xp.float32))

    def to_gpu(self, device=None):
        global xp
        xp = cp
        if device is not None:
            xp.cuda.Device(device).use()
        self.l_dl.to_gpu(device)
        self.l_Wr.to_gpu(device)

    def to_cpu(self):
        global xp
        xp = np
        self.l_dl.to_cpu()
        self.l_Wr.to_cpu()

    def get_h(self):
        return self.l_dl.get_h()

    def get_c(self):
        return self.l_dl.get_c()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized Chainer DNC')
    parser.add_argument('--hps', default="256,64,4,256,8",
                        help='DNC hyperparams: N,W,R,H,K. K=0 disable sparse Link matrix. H is hidden dim for LSTM')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gpu_for_nn_only', action='store_true',
                        help='If using GPU, use GPU/CuPy for LSTM/Linear layers, and CPU/NumPy for DNC core')
    parser.add_argument('--lstm_only', action='store_true', help='Use vanilla LSTMs instead of DNC')
    parser.add_argument('--task', default="addition", help="Which task to test: sum, repeat, priority_sort")
    parser.add_argument('--max_seq_len', default=12, type=int,
                        help="Max sequence length to train or test on (picked randomly starting from 0")
    parser.add_argument('--max_seq_wid', default=6, type=int, help="Width of each sequence. Always the same.")
    parser.add_argument('--test_seq_len', default=0, type=int,
                        help="Test longer sequence length than trained on to check generalization. 0 = off")
    parser.add_argument('--iterations', default=100000, type=int,
                        help="How many iterations of training sequences fed to network")
    args = parser.parse_args()
    print("args = " + str(vars(args))+"\n")

    N, W, R, H, K = args.hps.split(",")
    N, W, R, H, K = int(N), int(W), int(R), int(H), int(K)

    X = args.max_seq_wid
    if args.task == "addition":
        Y = 1
    elif args.task == "repeat":
        Y = args.max_seq_wid
    elif args.task == "priority_sort":
        Y = args.max_seq_wid-2
    else:
        print("Unknown task: "+args.task)
        exit()

    if args.lstm_only:
        model = DeepLSTM(H, Y, True)
        dnc_or_lstm = "lstm"
    else:
        model = DNC(X, Y, N, W, R, H, K, args.gpu_for_nn_only)
        dnc_or_lstm = "dnc"

    if not os.path.exists("result"):
        os.makedirs("result")

    max_iter = 0
    auto_resume_file = None
    files = os.listdir("result")
    for file in files:
        pattern = re.compile("^"+dnc_or_lstm+"_"+args.task+"_iter_")
        if pattern.match(file):
            iter = int(re.search(r'\d+', file).group())
            if (iter > max_iter):
                max_iter = iter
                auto_resume_file = os.path.join("result", file)
    if auto_resume_file is not None:
        print("Resuming from saved model: "+auto_resume_file+"\n")
        serializers.load_npz(auto_resume_file, model)

    if args.test_seq_len > 0:
        if not auto_resume_file:
            print("No saved model found to resume from for testing.")
            exit()
        args.test = True
        if max_iter == args.iterations:
            max_iter -= 1
    else:
        args.test = False

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    opt = optimizers.Adam(alpha=0.0001)
    opt.setup(model)
    start = time.time()
    for i in range(max_iter+1, args.iterations+1):
        model.reset_state()
        loss = 0
        outputs = []

        if args.task == "addition":
            def onehot(x, n):
                ret = xp.zeros(n).astype(xp.float32)
                ret[x] = 1.0
                return ret
            def generate_data():
                if args.test:
                    length = args.test_seq_len
                else:
                    length = int(xp.random.randint(2, (args.max_seq_len) + 1))
                content = xp.random.randint(0, args.max_seq_wid - 1, length)
                seq_length = length + 1
                input_data = xp.zeros((seq_length, args.max_seq_wid)).astype(xp.float32)
                target_data = 0.0
                sums_text = ""
                for i in range(seq_length):
                    if i < length:
                        input_data[i] = onehot(content[i], args.max_seq_wid)
                        target_data += content[i]
                        sums_text += str(content[i]) + " + "
                    else:
                        input_data[i] = onehot(args.max_seq_wid - 1, args.max_seq_wid)
                input_data = input_data.reshape((1,) + input_data.shape)
                target_data = xp.array(target_data).astype(xp.float32)
                target_data = target_data.reshape(1, 1, 1)
                return input_data, target_data, sums_text
            input_data, target_data, sums_text = generate_data()
            input_data = input_data[0]
            target_data = target_data[0]
            target_data = xp.vstack((xp.zeros(input_data.shape[0]-1).astype(xp.float32).reshape(-1, 1), target_data))

        elif args.task == "repeat":
            def generate_data():
                if args.test:
                    length = args.test_seq_len//2+1
                else:
                    length = int(xp.random.randint(1, args.max_seq_len//2+1))
                input_data = xp.zeros((2 * length + 1, args.max_seq_wid), dtype=xp.float32)
                target_data = xp.zeros((2 * length + 1, args.max_seq_wid), dtype=xp.int)
                sequence = xp.random.randint(0, 2, (length, args.max_seq_wid - 1))
                input_data[:length, :args.max_seq_wid - 1] = sequence
                input_data[length, -1] = 1
                target_data[length + 1:, :args.max_seq_wid - 1] = sequence
                return input_data, target_data
            input_data, target_data = generate_data()

        elif args.task == "priority_sort":
            def generate_data():
                if args.test:
                    length = args.test_seq_len//2+1
                else:
                    length = int(xp.random.randint(2, args.max_seq_len//2+1))
                input_data = xp.random.randint(0,2,(length, args.max_seq_wid)).astype(xp.float32)
                input_data[:,0] *= 0.
                input_data[-1] *= 0.
                input_data[:,-1] *= 0.
                input_data[-1,-1] = 1.
                priority_sort_index = xp.random.uniform(-1,1,(length-1,1))
                input_data[0:-1, 0:1] = priority_sort_index
                internal_sort_index = xp.argsort(priority_sort_index.reshape(-1))
                target_data = input_data[internal_sort_index]
                input_data = xp.vstack((input_data, xp.zeros((length, args.max_seq_wid)).astype(xp.float32)))
                target_data = xp.concatenate((xp.zeros((length+1, args.max_seq_wid)).astype(xp.float32),target_data))
                target_data = target_data[:,1:-1]
                return input_data, target_data
            input_data, target_data = generate_data()

        for j in range(input_data.shape[0]):
            output_data = model(F.expand_dims(input_data[j],0))
            outputs.append(output_data[0])
            #loss += F.sigmoid_cross_entropy(output_data[0], target_data[j], reduce="no")
            loss += (output_data[0] - target_data[j]) ** 2
        loss = F.mean(loss)

        if not args.test:
            model.cleargrads()
            loss.backward()
            opt.update()
            loss.unchain_backward()

        if not args.test and i == max_iter+1:
            print("\nTime \t\t Iter \t\t Loss")
            print("------------------------------------------")
        if not args.test and i % 10 == 0:
            print("{:.2f}s".format(time.time()-start), "\t\t", i, "\t\t", loss.data)
        if args.test or i % 500 == 0:
            print("---Sample Training Output---")
            print("Input Data:")
            print(input_data)
            if args.task == "addition":
                print(sums_text)
            print("Target Data:")
            print(target_data)
            if args.task == "addition":
                for row in outputs:
                    print("Output: {:.5f}".format(float(row.data)))
            else:
                for row in outputs:
                    print("Output: ", end="")
                    for col in row.data:
                        print("{:.5f} ".format(float(col)), end="")
                    print()
            print("----------------------------")
        if not args.test and i % 5000 == 0:
            filename = "result/"+dnc_or_lstm +"_" + args.task +"_iter_" + str(i) +".model"
            print("Saving model to: "+filename)
            serializers.save_npz(filename, model)
        if args.test:
            break