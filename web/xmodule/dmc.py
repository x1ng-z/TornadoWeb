import numpy as np
from scipy import signal
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import Bounds
# import time
# import matplotlib.pyplot as plt
import json
import sys


class dmcsolver:
    '''
    采用qp求解还是满秩矩阵的逆矩阵求解两种方式
    '''

    def init(self, N, R, Q, M, P, m, p, v, alphe, alphemethod, integrationInc_mv, B_resp_array, A_resp_matrix,
             runStyle):
        '''
        初始化求解器
        :param N:int 序列数目
        :param R:size=[m,1]系数r
        :param Q:size=[p,1]系数q
        :param M:控制域
        :param P:预测域
        :param m:mv个数
        :param p:pv个数
        :param v:前馈数量
        :param alphe:柔化系数 size=[p,1]
        :param alphemethod:size=[p,1]
        :param integrationInc_mv:mv引起的积分环节增量size=[p,m]
        :param B_resp_array:ff对pv响应向量size=[N*p,v]
        :param A_resp_matrix:mv对pv的响应矩阵size=[p*P,m*M]
        :param runStyle 误差是否需要进行归一化
        '''
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.p = p
        self.M = M
        self.m = m
        self.v = v
        self.alphe = alphe
        self.alphemethod = alphemethod
        self.runStyle = runStyle

        self.param_interatInc_matrix_PM = np.zeros((P, M))
        interatInc_sequence = np.arange(1, P + 1)
        for indexMi in range(M):
            self.param_interatInc_matrix_PM[indexMi:P * (0 + 1), indexMi] = interatInc_sequence[0: P - indexMi]

        self.integrationInc_mv = integrationInc_mv

        self.A_resp_matrix = A_resp_matrix
        self.B_resp_array = B_resp_array
        self.mv_array = []
        self.dmv_array = []
        self.ff_array = []
        self.dff_array = []
        self.yk_c = []
        # self.spk=[]
        self.spk_vector = []
        self.yk_lastdelta = []
        # self.interatInc_matrix_PM = []

        self.param_A = self.param_interatInc_matrix_PM  # np.tril(np.ones((self.P, self.M)), 0)
        self.plus_interagate = np.kron(self.integrationInc_mv, self.param_A)

        self.param_Q = np.kron(np.diagflat(self.Q), np.eye(self.P))

        self.param_R = np.kron(np.diagflat(self.R), np.eye(self.M))

        self.initreversiblemethod()
        self.initqpmethod()

    def reversiblemethod(self):
        '''
        直接用可逆矩阵法求解dmv
        :return:
        '''
        # ff和mv历史增量部分叠加
        # yk_lastdelta = self.predict(self.N, self.p, self.m, self.v, self.B_resp_array, self.mv_array, self.dmv_array, self.ff_array, self.dff_array,np.zeros((self.p, 1)))
        param_yk_p_N = np.kron(np.eye(self.p), np.eye(self.P, self.N))  # 提取前P个
        # param_yk_c_matrix = np.kron(np.eye(self.p), np.ones(self.P).reshape(-1, 1))
        # yk_c_vector = np.dot(param_yk_c_matrix, self.yk_c)  # yk时刻的矫正向量
        Ek = self.spk_vector - (np.dot(param_yk_p_N, self.yk_p_N))

        for index_alf, alf in enumerate(self.alphe.reshape(-1).tolist()):
            #构建柔和系数矩阵
            param_alphe = np.array([1 - alf ** i for i in range(1, self.P + 1)]).reshape(-1, 1) if self.alphemethod[
                                                                                                       index_alf, 0] == 'before' else np.flipud(
                np.array([1 - alf ** i for i in range(1, self.P + 1)]).reshape(-1, 1))
            if(self.runStyle==1):
                # 归一化
                # 最大的残差
                #_abs_residual=np.abs(Ek[index_alf * self.P:(index_alf + 1) * self.P])
                _max_residual=np.max(np.abs(Ek[index_alf * self.P:(index_alf + 1) * self.P]))
                Ek[index_alf * self.P:(index_alf + 1) * self.P]=Ek[index_alf * self.P:(index_alf + 1) * self.P]/_max_residual if _max_residual>0 else Ek[index_alf * self.P:(index_alf + 1) * self.P]
            #柔化
            Ek[index_alf * self.P:(index_alf + 1) * self.P] = Ek[index_alf * self.P:(index_alf + 1) * self.P] * param_alphe


        E_ = Ek  # - np.dot(param_yk_lastdelta, yk_lastdelta)
        dmv = np.dot(self.K_, E_)  # np.dot(self.L, np.dot(self.K_, E_))
        # print('inv total', np.dot(self.K_, E_))
        return dmv

    def qpmethod(self):
        bounds = Bounds(-1 * self.dmvmax.reshape(-1), self.dmvmax.reshape(-1))
        ##lb <= A.dot(x) <= ub #累计增量+mv初始值不能超过umax umin
        linear_constraintu = LinearConstraint(self.param_linelimit, (self.mvmin - self.mv0).reshape(-1),
                                              (self.mvmax - self.mv0).reshape(-1))
        # linear_constrainty = LinearConstraint(self.A,(self.Ymin-self.y0).transpose()[0,:],(self.Ymax-self.y0).transpose()[0,:])
        x0 = np.zeros(self.M * self.m)
        # res = minimize(self.costfunction, x0, method='trust-constr',
        #                jac=self.gradientcost,
        #                hess=self.hessioncost,
        #                # jac="2-point",#在求一次导比较难时使用
        #                # hess=SR1(),#在求二阶倒数比较难时使用
        #                constraints=[linear_constraintu],
        #                # constraints=[linear_constraintu,linear_constrainty],
        #                options={'verbose': 0, 'disp': False,'factorization_method':'SVDFactorization'},
        #                bounds=bounds,
        #                )
        ineq_cons = [{
            'type': 'ineq',
            'fun': lambda x: (np.dot(self.param_linelimit, x.reshape(-1, 1)) + self.mv0 - self.mvmin).reshape(-1),
            'jac': lambda x: self.param_linelimit
        },
            {
                'type': 'ineq',
                'fun': lambda x: (self.mvmax - (np.dot(self.param_linelimit, x.reshape(-1, 1)) + self.mv0)).reshape(-1),
                'jac': lambda x: -self.param_linelimit
            }]
        res = minimize(self.costfunction, x0, jac=self.gradientcost, hess=self.hessioncost,
                       bounds=bounds, constraints=ineq_cons, options={'disp': False})
        sys.stdout.write("qp=%s\n" % res.x)
        sys.stdout.flush()
        dmv = res.x.reshape((self.m * self.M, 1))  # np.dot(self.L, res.x)
        return dmv

    def initreversiblemethod(self):
        constant_inverse = np.linalg.pinv(
            np.dot(np.dot((self.A_resp_matrix + self.plus_interagate).transpose(), self.param_Q),
                   (self.A_resp_matrix + self.plus_interagate)) + self.param_R)
        self.K_ = np.dot(constant_inverse,
                         np.dot((self.A_resp_matrix + self.plus_interagate).transpose(), self.param_Q))
        cellk = np.zeros(self.M)
        cellk[0] = 1
        self.L = np.kron(np.eye(self.m), cellk)

    def initqpmethod(self):
        self.param_linelimit = np.kron(np.eye(self.m), np.tril(np.ones((self.M, self.M)), k=0))
        self.dmvmax = []
        self.mvmin = []
        self.mvmax = []
        self.mv0 = []

    def decode(self):
        return {
            'N': self.N,
            'Q': self.narrayconvert(self.Q),
            'R': self.narrayconvert(self.R),
            'P': self.P,
            'p': self.p,
            'M': self.M,
            'm': self.m,
            'v': self.v,
            'alphe': self.narrayconvert(self.alphe),
            'alphemethod': self.narrayconvert(self.alphemethod),
            'param_interatInc_matrix_PM': self.narrayconvert(self.param_interatInc_matrix_PM),
            'integrationInc_mv': self.narrayconvert(self.integrationInc_mv),
            'A_resp_matrix': self.narrayconvert(self.A_resp_matrix),
            'B_resp_array': self.narrayconvert(self.B_resp_array),
            'mv_array': self.narrayconvert(self.mv_array),
            'dmv_array': self.narrayconvert(self.dmv_array),
            'ff_array': self.narrayconvert(self.ff_array),
            'dff_array': self.narrayconvert(self.dff_array),
            'yk_c': self.narrayconvert(self.yk_c),
            'spk_vector': self.narrayconvert(self.spk_vector),
            "yk_lastdelta": self.narrayconvert(self.yk_lastdelta),
            'param_A': self.narrayconvert(self.param_A),
            'plus_interagate': self.narrayconvert(self.plus_interagate),
            'param_Q': self.narrayconvert(self.param_Q),
            'param_R': self.narrayconvert(self.param_R),
            'K_': self.narrayconvert(self.K_),
            'L': self.narrayconvert(self.L),
            'param_linelimit': self.narrayconvert(self.param_linelimit),
            'dmvmax': self.narrayconvert(self.dmvmax),
            'mvmin': self.narrayconvert(self.mvmin),
            'mvmax': self.narrayconvert(self.mvmax),
            'mv0': self.narrayconvert(self.mv0),
            'runStyle': self.runStyle
        }

    def encode(self, properties):
        self.N = properties['N']
        self.Q = self.listConvert(properties['Q'])
        self.R = self.listConvert(properties['R'])
        self.P = properties['P']
        self.p = properties['p']
        self.M = properties['M']
        self.m = properties['m']
        self.v = properties['v']
        self.alphe = self.listConvert(properties['alphe'])
        self.alphemethod = self.listConvert(properties['alphemethod'])
        self.param_interatInc_matrix_PM = self.listConvert(properties['param_interatInc_matrix_PM'])
        self.integrationInc_mv = self.listConvert(properties['integrationInc_mv'])
        self.A_resp_matrix = self.listConvert(properties['A_resp_matrix'])
        self.B_resp_array = self.listConvert(properties['B_resp_array'])
        self.mv_array = self.listConvert(properties['mv_array'])
        self.dmv_array = self.listConvert(properties['dmv_array'])
        self.ff_array = self.listConvert(properties['ff_array'])
        self.dff_array = self.listConvert(properties['dff_array'])
        self.yk_c = self.listConvert(properties['yk_c'])
        self.spk_vector = self.listConvert(properties['spk_vector'])
        self.yk_lastdelta = self.listConvert(properties["yk_lastdelta"])
        self.param_A = self.listConvert(properties['param_A'])
        self.plus_interagate = self.listConvert(properties['plus_interagate'])
        self.param_Q = self.listConvert(properties['param_Q'])
        self.param_R = self.listConvert(properties['param_R'])
        self.K_ = self.listConvert(properties['K_'])
        self.L = self.listConvert(properties['L'])
        self.param_linelimit = self.listConvert(properties['param_linelimit'])
        self.dmvmax = self.listConvert(properties['dmvmax'])
        self.mvmin = self.listConvert(properties['mvmin'])
        self.mvmax = self.listConvert(properties['mvmax'])
        self.mv0 = self.listConvert(properties['mv0'])
        self.runStyle = properties['runStyle']

    def costfunction(self, dmv):
        dmv = dmv.reshape((self.m * self.M, 1))

        # param_yk_c_matrix = np.kron(np.eye(self.p), np.ones(self.P).reshape(-1, 1))
        # yk_c_vector = np.dot(param_yk_c_matrix, self.yk_c)  # yk时刻的矫正向量
        param_yk_p_N = np.kron(np.eye(self.p), np.eye(self.P, self.N))  # 提取前P个

        Ek = self.spk_vector - (np.dot(param_yk_p_N, self.yk_p_N))
        for index_alf, alf in enumerate(self.alphe.reshape(-1).tolist()):
            param_alphe = np.array([1 - alf ** i for i in range(1, self.P + 1)]).reshape(-1, 1) if self.alphemethod[
                                                                                                       index_alf, 0] == 'before' else np.flipud(
                np.array([1 - alf ** i for i in range(1, self.P + 1)]).reshape(-1, 1))

            if (self.runStyle == 1):
                # 归一化
                # 最大的残差
                _max_residual = np.max(np.abs(Ek[index_alf * self.P:(index_alf + 1) * self.P]))
                Ek[index_alf * self.P:(index_alf + 1) * self.P] = Ek[index_alf * self.P:( index_alf + 1) * self.P] / _max_residual if _max_residual > 0 else Ek[index_alf * self.P:(index_alf + 1) * self.P]
            #乘上柔化矩阵
            Ek[index_alf * self.P:(index_alf + 1) * self.P] = Ek[
                                                              index_alf * self.P:(index_alf + 1) * self.P] * param_alphe
        E_ = np.dot(-1 * (self.A_resp_matrix + self.plus_interagate), dmv) + Ek
        cost = np.dot(np.dot(E_.transpose(), self.param_Q), E_) + np.dot(np.dot(dmv.transpose(), self.param_R), dmv)
        return cost[0][0]

    def gradientcost(self, dmv):
        dmv = dmv.reshape((self.m * self.M, 1))

        param_yk_p_N = np.kron(np.eye(self.p), np.eye(self.P, self.N))  # 提取前P个

        # param_yk_p_N = np.kron(np.eye(self.p), np.ones(self.P).reshape(-1, 1))
        # yk_c_vector = np.dot(param_yk_c_matrix, self.yk_c)  # yk时刻的矫正向量

        # E_tao = self.spk_vector - (yk_c_vector + np.dot(param_yk_lastdelta, self.yk_lastdelta))
        Ek = self.spk_vector - (np.dot(param_yk_p_N, self.yk_p_N))
        for index_alf, alf in enumerate(self.alphe.reshape(-1).tolist()):
            param_alphe = np.array([1 - alf ** i for i in range(1, self.P + 1)]).reshape(-1, 1) if self.alphemethod[
                                                                                                       index_alf, 0] == 'before' else np.flipud(
                np.array([1 - alf ** i for i in range(1, self.P + 1)]).reshape(-1, 1))
            Ek[index_alf * self.P:(index_alf + 1) * self.P] = Ek[
                                                              index_alf * self.P:(index_alf + 1) * self.P] * param_alphe
        E_ = np.dot(-1 * (self.A_resp_matrix + self.plus_interagate), dmv) + Ek
        gradient = np.dot(np.dot(-1 * (self.A_resp_matrix + self.plus_interagate).transpose(), self.param_Q),
                          E_) + np.dot(
            self.param_R, dmv)
        return 2 * gradient.reshape(-1)

    def hessioncost(self, dmv):
        # dmv = dmv.reshape((self.m * self.M, 1))

        k = np.dot(np.dot((self.A_resp_matrix + self.plus_interagate).transpose(), self.param_Q),
                   (self.A_resp_matrix + self.plus_interagate)) + self.param_R
        # hess = np.dot(k, dmv)
        return 2 * k
        # justQ = np.dot(self.Q, self.alphediag_2)
        # H=(2 * np.dot(np.dot(self.dynamix_matrix.transpose(), justQ), self.dynamix_matrix) + 2 * self.R)
        # # print(H)
        # return H.transpose()

    def setmv_array(self, mv_array):
        '''
        :param mv_array:mv队列size=[m,N]
        :return:
        '''
        self.mv_array = mv_array

    def setdmv_array(self, dmv_array):
        '''
        :param dmv_array:dmv队列 size=[m,N]
        :return:
        '''
        self.dmv_array = dmv_array

    def setff_array(self, ff_array):
        '''
        :param ff_array: 前馈队列size=[v,N]
        :return:
        '''
        self.ff_array = ff_array

    def setdff_array(self, dff_array):
        '''
        :param dff_array: dff队列size=[v,N]
        :return:
        '''
        self.dff_array = dff_array

    def setyk_c(self, yk_c):
        '''
        :param yk_c: pv测量值size=[p,1]
        :return:
        '''
        self.yk_c = yk_c

    def setspk_vector(self, spk_vector):
        '''
        :param spk_vector: 未来N步的给定值 size=[p*N,1]
        :return:
        '''
        self.spk_vector = spk_vector

    def setdmvmax(self, dmvmax):
        '''
        :param dmvmax: dmv的最大值 size=[m*M,1]
        :return:
        '''
        self.dmvmax = dmvmax

    def setmvmin(self, mvmin):
        '''
        :param mvmin:mv最小值size=[m*M,1]
        :return:
        '''
        self.mvmin = mvmin

    def setmvmax(self, mvmax):
        '''
        :param mvmax:mv最大值size=[m*M,1]
        :return:
        '''
        self.mvmax = mvmax

    def setmv0(self, mv0):
        '''
        :param mv0: mv的初始值size=[m*M,1]
        :return:
        '''
        self.mv0 = mv0

    def setyk_p_N(self, yk_p_N):
        '''
        :param yk_lastdelta:k时刻的根据dff和dmv的队列中的数据得到基于yk的k+1至k+N的时刻y的预测值size=[p*N,1]
        :return:
        '''
        self.yk_p_N = yk_p_N

    def narrayconvert(self, value):
        return value.tolist() if (type(value) == np.ndarray or type(value) == np.matrix) else value

    def listConvert(self, value):
        return np.array(value) if type(value) == list else value


class dmc:
    def init(self, P, p, M, m, N, outStep, feedforwardNum, A, B, qi, ri, alphe, funneltype, alphemethod, runStyle,
             integrationInc_mv, integrationInc_ff, DEBUG):
        '''
                    function:
                        预测控制
                    Args:
                           :param P 预测时域长度
                           :param p PV数量
                           :param M mv计算后续输出几步
                           :param m mv数量
                           :param N 阶跃响应序列个数
                           :param outStep 输出间隔
                           :param feedforwardNum 前馈数量
                           :param A mv对pv的阶跃响应
                           :param B ff对pv的阶跃响应
                           :param qi 优化控制域矩阵，用于调整sp与预测的差值，在滚动优化部分
                           :param ri 优化时间域矩阵,用于约束调整dmv的大小，在滚动优化部分
                           :param pvusemv 一个矩阵，标记pv用了哪些mv
                           :param runStyle 运行方式，现在定义为误差是否归一化
                           :param alphe 柔化系数
                           :param alphemethod 柔化系数方法 目前支持before after两种
                           :param funneltype 漏斗类型shape=(pv数量，2)：如pv数量为2 [[0,0],[1,0],[0,1]],[0,0]全漏斗，[1,0]下漏斗，[0,1]上漏斗
                           :param integrationInc_mv mv积分环节增量shape=[p,m]
                           :param integrationInc_ff ff积分环节增量shape=[p,v]
                    '''

        '''预测时域长度'''
        self.P = P

        '''输出个数'''
        self.p = p

        '''控制时域长度'''
        self.M = M

        '''输入个数'''
        self.m = m

        '''建模时域'''
        self.N = N

        '''输出间隔'''
        self.outStep = outStep

        '''前馈数量'''
        self.v = feedforwardNum

        '''pv 使用mv的标记矩阵'''
        # self.pvusemv = pvusemv

        self.alphe = alphe

        self.alphemethod = alphemethod

        self.runStyle = runStyle

        '''mv 对 pv 的阶跃响应'''
        self.A_step_response_sequence = np.zeros((p * N, m))
        for loop_outi in range(p):
            for loop_ini in range(m):
                self.A_step_response_sequence[N * loop_outi:N * (loop_outi + 1), loop_ini] = A[loop_outi, loop_ini, :]

        '''ff 对 pv 的阶跃响应'''
        self.B_step_response_sequence = []

        '''前馈数量为0 则不需要初始化前馈响应B_step_response_sequence'''
        if feedforwardNum != 0:
            self.B_step_response_sequence = np.zeros((p * N, feedforwardNum))
            for outi in range(p):
                for ini in range(feedforwardNum):
                    self.B_step_response_sequence[outi * N:(outi + 1) * N, ini] = B[outi, ini]
        '''算法运行时间'''
        self.costtime = 0
        self.PINF = 2 ** 200  # 正无穷
        self.NINF = -2 ** 200  # 负无穷

        self.funneltype = funneltype
        self.dynamic_matrix_PM, dynamic_matrix_NN, param_interatInc_matrix_PM = self.responmatrix(
            self.A_step_response_sequence)
        self.dmcsolver = dmcsolver()
        self.dmcsolver.init(N, ri.reshape(m, 1),
                            qi.reshape(p, 1),
                            M,
                            P,
                            m,
                            p,
                            feedforwardNum,
                            alphe.reshape(p, 1),
                            alphemethod.reshape(p, 1),
                            integrationInc_mv,
                            self.B_step_response_sequence,
                            self.dynamic_matrix_PM, runStyle)
        '''积分增量'''
        self.integrationInc_mv = integrationInc_mv
        self.integrationInc_ff = integrationInc_ff
        '''
        用于计算响应序列的差值向量hi
        [0,0,0,0
         1,0,0,0
         0,1,0,0
         0,0,1,0
        ]
        '''
        params_hi = np.kron(np.eye(p), np.eye(N, k=-1))
        self.hi_a = self.A_step_response_sequence - np.dot(params_hi, self.A_step_response_sequence)  # mv对pv响应ai的增量
        if feedforwardNum != 0:
            self.hi_b = self.B_step_response_sequence - np.dot(params_hi, self.B_step_response_sequence)
        else:
            self.hi_b = []
        self._DEBUG = DEBUG
        pass

    def Pj(self, hi, ff, dff, sigma):
        '''
            :return Pj返回的数据是某一个前馈Pj向量数据[P1,P2,P3...PN]
           :param hi 响应增量向量
           :param ff ff/mv向量，[ff_N-1...ff_k-1,ff_k],一共有N个数据，初始化时全部初始化成ff_k的数值，也就是说全部初始化成刚进来的数据
           :param dff ff/mv增量,[dff_N-1...dff_k-1,dff_k]，一共有N个数据，初始化的时候全部初始化成0，认为以前的ff或者mv是没有变化的
           :param sigma 积分环节的比例系数
        '''
        masknew = np.ones(len(hi))
        masknew[-1] = 0  # 最新的dffk=0
        # print(masknew)
        param_Sm = np.ones((len(hi), len(hi)))
        # print(param_Sm)
        param_Sm = np.tril(param_Sm, 0)
        # print(param_Sm)
        # print('pickdff', dff)
        # print('pickhi', hi)
        maskdff = dff * masknew
        conv = signal.convolve(maskdff, hi)

        # print('conv', conv[len(hi) - 1:])
        Sm = conv[self.N - 1:] + ff[-2] * sigma
        Pj = np.dot(param_Sm, Sm.reshape(-1, 1))
        # print('Pj', Pj)
        return Pj

    def predict(self, mv_array, dmv_array, ff_array, dff_array, yk_c):
        '''
        :param N:序列个数
        :param P:预测域
        :param p:pv个数
        :param M:控制域
        :param m:mv个数
        :param v:前馈ff数量
        :param A_resp_array:mv对pv的响应序列,size=[p*N,m]
        :param A_resp_matrix:mv对pv的响应序列,size=A[p*N,m*N]
        :param B_resp_array:ff对pv的响应序列,size=[p*N,m]
        :param B_resp_matrix:ff对pv的响应序列B,size=[p*N,m]
        :param mv_array:mv的数据数组[[mv_k-N+1,...,mv_k-2,mv_k-1,mv_k],..],size=[m,N]
        :param dmv_array:dmv的数据数组[[dmv_k-N+1,...,dmv_k-2,dmv_k-1,dmv_k],..],size=[m,N]
        :param ff_array:ff的数据数组[[ff_k-N+1,...,ff_k-2,ff_k-1,ff_k],..],size=[v,N]
        :param dff_array:ff的数据数组[[dff_k-N+1,...,dff_k-2,dff_k-1,dff_k],..],size=[v,N'
        :param yk_c:k时刻的矫正值(真实值),size=[p,1]
        :return:
        '''

        '''预测值的叠加分为两个部分，
        一个是mv的叠加，这部分叠加的时候建立的假设是当前k时刻读到三个数，pv值Y,mv值x.这里需要理解一下
        mv在k时刻到的是mv(k-1)时刻的值，因为k时刻 mv需要根据我们计算出来的dmv进行变化了，而ff不同，因为ff不可控性
        我们在k时刻读到的ff就是已经叠加了dff(k)的值了，所以他就是ff(k),
        因此，在预测的时候，可以认为当前的dmv(k)==0,mv(k)==mv(k-1),也就是基于过去的dmv和mv进行预测
                Yk+1_c=Yk_c+A_vector*dmv(k)+P_a_j
        把A_vector)*dmv(k)+P_a_j部分计作ΔA
                Yk+1_c=Yk_c+ΔA
        mv and dmv向量：
                mv=[mv(k-N+1)...,mv(k-2),mv(k-1),mv(k)],mv(k)其实是在预测的时候还是未知，dmv(k)还没算出来。在预测时候mv(k)=mv(k-1)
                dmv=[dmv(k-N+1)...,dmv(k-2),dmv(k-1),dmv(k)]，dmv(k)未知,在预测的时候认为0
        另外一部分是前馈的叠加:
                  Yk+1_c=Yk_c+ΔA+B_vector*dff(k)+P_b_j
        把B_vector*dff(k)+P_b_j部分计作ΔB
                  Yk+1_c=Yk_c+ΔA+ΔB
         ff and dff向量：
                ff=[ff(k-N+1)...,ff(k-2),ff(k-1),ff(k)],ff(k)，ff(k)已知
                dff=[dff(k-N+1)...,dff(k-2),dff(k-1),dff(k)]，dff(k)已知
        '''
        param_yk_c_matrix = np.kron(np.eye(self.p), np.ones(self.N).reshape(-1, 1))
        yk_c_vector = np.dot(param_yk_c_matrix, yk_c)  # yk时刻的矫正向量

        for index_p in range(self.p):
            # 计算mv部分的预测增量，由于是k时是基于过去来预测未来，并且dmv(k)=0，所以只要计算P_a_j部分就行了
            for index_m in range(self.m):
                yk_c_vector[index_p * self.N:(index_p + 1) * self.N] = yk_c_vector[index_p * self.N:(
                                                                                                            index_p + 1) * self.N] + self.Pj(
                    self.hi_a[index_p * self.N:(index_p + 1) * self.N, index_m], mv_array[index_m, :],
                    dmv_array[index_m, :], self.integrationInc_mv[index_p, index_m])

            if self.v != 0:
                for index_v in range(self.v):
                    yk_c_vector[index_p * self.N:(index_p + 1) * self.N] = yk_c_vector[index_p * self.N:(
                                                                                                                index_p + 1) * self.N] + np.dot(
                        (self.B_step_response_sequence[index_p * self.N:(index_p + 1) * self.N, index_v] +
                         self.integrationInc_ff[index_p, index_v] * np.arange(1, self.N + 1)),
                        dff_array[index_v, -1]).reshape(-1, 1) + self.Pj(
                        self.hi_b[index_p * self.N:(index_p + 1) * self.N, index_v], ff_array[index_v, :],
                        dff_array[index_v, :], self.integrationInc_ff[index_p, index_v])
        return yk_c_vector

    def rolloptimization(self, spk_vector, yk_c, mv0, dmvmax, dmvmin, mvmax, mvmin, yk_p_N):
        '''
        :param yk_c pv初始值 size=[p,1]
        :param spk_vector k时刻的以后k+1至k+P设定值，size=[P*p,1]，这里传入的已经是前P个数据了
        :param mv0 mv0初始值 size=[m,1]
        :param dmvmax dmv最大值 size=[m,1]
        :param dmvmin,dmv最小值 size=[m,1]
        :param mvmax,mv最大值 size=[m,1]
        :param mvmin mv最小值 size=[m,1]
        :param mv_array:mv的数据数组[[mv_k-N+1,...,mv_k-2,mv_k-1,mv_k],..],size=[m,N]
        :param dmv_array:dmv的数据数组[[dmv_k-N+1,...,dmv_k-2,dmv_k-1,dmv_k],..],size=[m,N]
        :param ff_array:ff的数据数组[[ff_k-N+1,...,ff_k-2,ff_k-1,ff_k],..],size=[v,N]
        :param dff_array:ff的数据数组[[dff_k-N+1,...,dff_k-2,dff_k-1,dff_k],..],size=[v,N'
        :param yk_p_N:k时刻的k+1~k+N预测值(真实值),size=[p*N,1]
        :return:
        '''

        self.dmcsolver.setyk_p_N(yk_p_N)
        self.dmcsolver.setspk_vector(spk_vector)
        self.dmcsolver.setyk_c(yk_c)
        dmv = self.dmcsolver.reversiblemethod()
        sys.stdout.write(" inv =%s\n" % dmv.tolist())
        sys.stdout.flush()
        if self._DEBUG:
            print("inv", dmv)

        if (not self.checklimit(mv0, dmv, mvmax, mvmin, dmvmax)) or self._DEBUG:
            self.dmcsolver.setdmvmax(np.kron(dmvmax, np.ones((self.M, 1))))
            self.dmcsolver.setmvmax(np.kron(mvmax, np.ones((self.M, 1))))
            self.dmcsolver.setmvmin(np.kron(mvmin, np.ones((self.M, 1))))
            self.dmcsolver.setmv0(np.kron(mv0, np.ones((self.M, 1))))
            dmv = self.dmcsolver.qpmethod()
            if self._DEBUG:
                print("qp", dmv)

        return dmv

        # #满秩矩阵
        # param_A = np.tril(np.ones((self.P, self.M)), 0)
        # plus_interagate=np.kron(self.integrationInc_mv,param_A)
        #
        # param_Q=np.kron(np.diagflat(self.Q),np.eye(self.P))
        #
        # param_R=np.kron(np.diagflat(self.R),np.eye(self.M))
        # constant_inverse=np.linalg.pinv(np.dot(np.dot((A_resp_matrix+plus_interagate).transpose(),param_Q),(A_resp_matrix+plus_interagate))+param_R)
        # K_=np.dot(constant_inverse,np.dot((A_resp_matrix+plus_interagate).transpose(),param_Q))
        #
        # #ff和mv历史增量部分叠加
        # yk_lastdelta=self.predict(self.N,self.p,self.m,self.v,B_resp_array,mv_array,dmv_array,ff_array,dff_array.np.zeros((self.p,1)))
        # #提取前P个
        # param_yk_lastdelta=np.kron(np.eye(self.p),np.eye(self.P,self.N))
        #
        #
        # param_yk_c_matrix = np.kron(np.eye(self.p), np.ones(self.P).reshape(-1, 1))
        # yk_c_vector = np.dot(param_yk_c_matrix, yk_c)  # yk时刻的矫正向量
        # #param_yk_c_matrix+=yk_lastdelta
        # spk_vector=np.dot(param_yk_c_matrix, self.spk)
        # Ek=spk_vector-yk_c_vector
        #
        # for index_alf,alf in enumerate(self.alphe.tolist()):
        #     Ek[index_alf * self.P:(index_alf + 1) * self.P]=Ek[index_alf*self.P:(index_alf+1)*self.P]*np.array([1 - alf ** i for i in range(1, self.P + 1)]).reshape(-1,1)
        # E_=Ek-np.dot(param_yk_lastdelta,yk_lastdelta)
        # cellk=np.zeros(self.M)
        # cellk[0]=1
        # K=np.kron(np.eye(self.m),cellk)
        # deltau=np.dot(K,np.dot(K_,E_))
        #
        # #迭代法求
        pass

    def responmatrix(self, A_step_response_sequence):
        dynamic_matrix_PM = np.zeros((self.P * self.p, self.M * self.m))  # P预测域内的响应矩阵动态矩阵
        dynamic_matrix_NN = np.zeros((self.N * self.p, self.N * self.m))  # N全部响应序列的响应动态矩阵
        interatInc_matrix_PM = np.zeros((self.P, self.M))
        interatInc_sequence = np.arange(1, self.P + 1)
        for indexMi in range(self.M):
            interatInc_matrix_PM[indexMi:self.P * (0 + 1), indexMi] = interatInc_sequence[0: self.P - indexMi]
        for indexpi in range(self.p):
            for indexmi in range(self.m):
                for indexMi in range(self.M):
                    dynamic_matrix_PM[self.P * indexpi + indexMi:self.P * (indexpi + 1),
                    self.M * indexmi + indexMi] = A_step_response_sequence[
                                                  indexpi * self.N:indexpi * self.N + self.P - indexMi, indexmi]

                for indexNi in range(self.N):
                    dynamic_matrix_NN[self.N * indexpi + indexNi:self.N * (indexpi + 1),
                    self.N * indexmi + indexNi] = A_step_response_sequence[
                                                  indexpi * self.N:indexpi * self.N + self.N - indexNi, indexmi]

        return dynamic_matrix_PM, dynamic_matrix_NN, interatInc_matrix_PM

    def buildfunel(self, sp, deadZones, funelInitValues):
        '''
                            function:
                                  构建漏斗
                            Args:
                                :param sp sp值
                                :param deadZones 死区
                                :param funelInitValues 漏斗初始值
                                :param N 阶跃响应数据点数量
                                :param p pv数量
                                :param funneltype漏斗类型
                                :param maxfunnelvale上漏斗最大值 近似正无穷
                                ;:param minfunnelvale 下漏斗最小值 近似负无穷
                            Returns:
                                :return originalfunnels 原始全漏斗数据为shape为(2,p*N)
                                funels=[ up1  up2..upN  pv的高限制都在这一行
                                donw1 donw2...donwN   pv的低限制都在这一行 ]

                                 :return decoratefunnels 根据漏斗类型修饰的漏斗数据为shape为(p*N,2)
                                funels=[ up1  up2..upN  pv的高限制都在这一行
                                donw1 donw2...donwN   pv的低限制都在这一行 ]

                 '''
        funnels = np.zeros((self.p * self.N, 2))
        funnelswithtype = np.zeros((self.p * self.N, 2))

        leftUpPointsY = sp + deadZones + funelInitValues
        leftDownPointsY = sp - deadZones - funelInitValues

        rightUpProintsY = sp + deadZones
        rightdDownProintsY = sp - deadZones

        for indexp in range(self.p):
            upki = np.true_divide(leftUpPointsY[indexp, 0] - rightUpProintsY[indexp, 0], (1 - self.N))
            upbi = leftUpPointsY[indexp, 0]

            downki = np.true_divide(leftDownPointsY[indexp, 0] - rightdDownProintsY[indexp, 0], (1 - self.N))
            downbi = leftDownPointsY[indexp, 0]

            funnels[indexp * self.N:(indexp + 1) * self.N, 0] = upki * np.arange(self.N) + upbi
            funnels[indexp * self.N:(indexp + 1) * self.N, 1] = downki * np.arange(self.N) + downbi

            funnelswithtype[indexp * self.N:(indexp + 1) * self.N, 0] = upki * np.arange(self.N) + upbi + \
                                                                        self.funneltype[indexp, 0] * self.PINF
            funnelswithtype[indexp * self.N:(indexp + 1) * self.N, 1] = downki * np.arange(self.N) + downbi + \
                                                                        self.funneltype[indexp, 1] * self.NINF
        return funnels, funnelswithtype

    def biuldspkvectorwhithFunel(self, ykp_c_N, funels):
        '''
        :param ykp_c_N 预测值[p*N,1]
        :param funels: shape(2,N*p)[ up1  up2..upN  pv的高限制都在这一行
                         donw1 donw2...donwN   pv的低限制都在这一行 ]
        :return: W_i shape=(p * P, 1)
        '''
        param_ykp_c = np.kron(np.eye(self.p), np.eye(self.P, self.N))
        ykp_c_p = np.dot(param_ykp_c, ykp_c_N)
        funels_p = np.dot(param_ykp_c, funels)
        midpart = ykp_c_p.copy()
        uppart = funels_p[:, 0].reshape((self.p * self.P, 1)).copy()
        downpart = funels_p[:, 1].reshape((self.p * self.P, 1)).copy()
        uppart[ykp_c_p < funels_p[:, 0].reshape((self.p * self.P, 1))] = 0
        downpart[ykp_c_p > funels_p[:, 1].reshape((self.p * self.P, 1))] = 0
        midpart[(midpart > funels_p[:, 0].reshape((self.p * self.P, 1))) + (
                midpart < funels_p[:, 1].reshape((self.p * self.P, 1)))] = 0
        spkvector = midpart + uppart + downpart
        return spkvector

        #
        # for indexp in range(p):
        #     '''这里，响应和漏斗都是N的，要截取为P的'''
        #     upfunel = funels[0, indexp * N:(indexp) * N + P].copy()
        #     downfunel = funels[1, indexp * N:(indexp) * N + P].copy()
        #     # 判断超过上funel，超过则获取 funle,不然则获取y0的
        #     y0P_test = y0[indexp * N:(indexp) * N + P, 0].copy()
        #     y0P_rec = y0[indexp * N:(indexp) * N + P, 0].copy()
        #
        #     y0P_rec[((y0P_test <= downfunel) + (y0P_test >= upfunel))] = 0  # 截取y0部分
        #     downfunel[downfunel < y0P_test] = 0  # 截取下漏斗部分
        #     upfunel[y0P_test < upfunel] = 0  # 截取上漏斗部分
        #     W_i[indexp * P:(indexp + 1) * P, 0] = y0P_rec + downfunel + upfunel
        # return W_i

    def mvconstraint(self, mv0, dmv, mvmax, mvmin, dmvmax, dmvmin):
        '''
        :param mv0:
        :param dmv:
        :param mvmax:
        :param mvmin:
        :param dmvmax:
        :param dmvmin:
        :return:
        '''
        '''判断计算出来的值是否为nan，如果是，则替换为0'''
        dmv[np.isnan(dmv)] = 0
        '''dmv累加矩阵'''

        '''L矩阵 只取即时控制增量'''
        cellk = np.zeros(self.M)
        cellk[0] = 1
        L = np.kron(np.eye(self.m), cellk)
        '''本次要输出的dmv'''
        dmv_k = np.dot(L, dmv)
        for index, needcheckdmv in np.ndenumerate(dmv_k):
            '''检查下dmv是否在限制之内'''
            if (np.abs(needcheckdmv) > dmvmax[index]):
                dmv_k[index] = dmvmax[index] if (dmv_k[index] > 0) else (
                        -1 * dmvmax[index])
            '''dmv是否小于最小调节量，如果小于，则不进行调节'''
            if (np.abs(needcheckdmv) < dmvmin[index]):
                dmv_k[index] = 0
            '''mv叠加dmv完成以后是否大于mvmax'''
            if ((mv0[index] + dmv_k[index]) >= mvmax[index]):
                dmv_k[index] = mvmax[index] - mv0[index]
            '''mv叠加dmv完成以后是否小于于mvmmin'''
            if ((mv0[index] + dmv_k[index[0]]) <= mvmin[index]):
                dmv_k[index] = mvmin[index] - mv0[index]

        return dmv_k

    def checklimit(self, mv0, dmv, mvmax, mvmin, dmvmax):
        '''
        :param mv0:
        :param dmv:
        :param mvmax:
        :param mvmin:
        :param dmvmax:
        :param dmvmin:
        :return:
        '''
        param_dmv = np.kron(np.eye(self.m), np.tril(np.ones((self.M, self.M)), k=0))
        dmv_vector = np.dot(param_dmv, dmv.reshape(self.m * self.M, 1))

        '''叠加了增量后的mv'''
        param_mv0 = np.kron(np.eye(self.m), np.ones(self.M).reshape(-1, 1))
        mv0_vector = np.dot(param_mv0, mv0.reshape((self.m, 1)))

        accummv = mv0_vector + dmv_vector

        mvmin_vector = np.dot(param_mv0, mvmin.reshape((self.m, 1)))

        mvmax_vector = np.dot(param_mv0, mvmax.reshape((self.m, 1)))

        dmvmax_vector = np.dot(param_mv0, dmvmax.reshape((self.m, 1)))
        '''分解为mvmin和mvmax'''

        '''检查增量下界上界'''
        '''检查mv上下限'''
        if (((mvmin_vector) <= accummv).all() and (mvmax_vector >= accummv).all() and (
                np.abs(dmv) <= dmvmax_vector).all()):
            return True
        else:
            return False

    def decode(self):
        '''
        将对象序列化成json
        :return:
        '''
        return {
            'P': self.P,
            'p': self.p,
            'M': self.M,
            'm': self.m,
            'N': self.N,
            'outStep': self.outStep,
            'v': self.v,
            'alphe': self.narrayConvert(self.alphe),
            'alphemethod': self.narrayConvert(self.alphemethod),
            'A_step_response_sequence': self.narrayConvert(self.A_step_response_sequence),
            'B_step_response_sequence': self.narrayConvert(self.B_step_response_sequence),
            'costtime': self.narrayConvert(self.costtime),
            'PINF': self.PINF,
            'NINF': self.NINF,  # 负无穷
            'funneltype': self.narrayConvert(self.funneltype),
            'dynamic_matrix_PM': self.narrayConvert(self.dynamic_matrix_PM),
            'dmcsolver': self.narrayConvert(self.dmcsolver.decode()),
            'integrationInc_mv': self.narrayConvert(self.integrationInc_mv),
            'integrationInc_ff': self.narrayConvert(self.integrationInc_ff),
            'hi_a': self.narrayConvert(self.hi_a),  # mv对pv响应ai的增量
            'hi_b': self.narrayConvert(self.hi_b),
            'DEBUG': self.narrayConvert(self._DEBUG),
            'runStyle': self.runStyle
        }

    def encode(self, properties):
        '''
                    function:将json反序列化成对象
                        预测控制
                    Args:
                           :param P 预测时域长度
                           :param p PV数量
                           :param M mv计算后续输出几步
                           :param m mv数量
                           :param N 阶跃响应序列个数
                           :param outStep 输出间隔
                           :param feedforwardNum 前馈数量
                           :param A mv对pv的阶跃响应
                           :param B ff对pv的阶跃响应
                           :param qi 优化控制域矩阵，用于调整sp与预测的差值，在滚动优化部分
                           :param ri 优化时间域矩阵,用于约束调整dmv的大小，在滚动优化部分
                           :param pvusemv 一个矩阵，标记pv用了哪些mv
                           :param alphe 柔化系数
                           :param alphemethod 柔化系数方法 目前支持before after两种
                           :param funneltype 漏斗类型shape=(pv数量，2)：如pv数量为2 [[0,0],[1,0],[0,1]],[0,0]全漏斗，[1,0]下漏斗，[0,1]上漏斗
                           :param integrationInc_mv mv积分环节增量shape=[p,m]
                           :param integrationInc_ff ff积分环节增量shape=[p,v]
                    '''

        '''预测时域长度'''
        self.P = properties['P']

        '''输出个数'''
        self.p = properties['p']

        '''控制时域长度'''
        self.M = properties['M']

        '''输入个数'''
        self.m = properties['m']

        '''建模时域'''
        self.N = properties['N']

        '''输出间隔'''
        self.outStep = properties['outStep']

        '''前馈数量'''
        self.v = properties['v']

        '''pv 使用mv的标记矩阵'''
        # self.pvusemv = pvusemv

        self.alphe = self.listConvert(properties['alphe'])

        self.alphemethod = self.listConvert(properties['alphemethod'])

        '''mv 对 pv 的阶跃响应'''
        self.A_step_response_sequence = self.listConvert(properties['A_step_response_sequence'])

        '''ff 对 pv 的阶跃响应'''
        self.B_step_response_sequence = self.listConvert(properties['B_step_response_sequence'])

        '''前馈数量为0 则不需要初始化前馈响应B_step_response_sequence'''

        '''算法运行时间'''
        self.costtime = self.listConvert(properties['costtime'])
        self.PINF = self.listConvert(properties['PINF'])  # 正无穷
        self.NINF = self.listConvert(properties['NINF'])  # 负无穷

        self.funneltype = self.listConvert(properties['funneltype'])
        self.dynamic_matrix_PM = self.listConvert(properties['dynamic_matrix_PM'])
        self.dmcsolver = dmcsolver()
        self.dmcsolver.encode(properties['dmcsolver'])
        '''积分增量'''
        self.integrationInc_mv = self.listConvert(properties['integrationInc_mv'])
        self.integrationInc_ff = self.listConvert(properties['integrationInc_ff'])
        '''
        用于计算响应序列的差值向量hi
        [0,0,0,0
         1,0,0,0
         0,1,0,0
         0,0,1,0
        ]
        '''

        self.hi_a = self.listConvert(properties['hi_a'])
        self.hi_b = self.listConvert(properties['hi_b'])
        self._DEBUG = self.listConvert(properties['DEBUG'])
        self.runStyle = properties['runStyle']
        pass

    def narrayConvert(self, value):
        return value.tolist() if (type(value) == np.ndarray or type(value) == np.matrix) else value

    def listConvert(self, value):
        return np.array(value) if type(value) == list else value


def main(input_data, context):
    # start=time.time()
    OUT = {}
    __dmc = None
    if (input_data['step'] == 'build'):
        if 'dmc' not in context:
            '''模型参数'''
            p = input_data["p"]  # 2
            m = input_data["m"]  # 2
            N = input_data["N"]  # 4
            P = input_data["P"]  # 3
            M = input_data["M"]  # 2
            v = input_data["fnum"]  # 2
            A_step_response_sequence = np.array(
                input_data[
                    "A"])  # np.kron(np.ones((p, m)), np.arange(1, N + 1).reshape((-1, 1))).transpose().reshape((p, m, N))
            B_step_response_sequence = np.array(input_data["B"]) if (
                    "B" in input_data) else []  # np.kron(np.ones((p, m)), np.arange(2, N + 2).reshape((-1, 1))).transpose().reshape((p, v, N))
            Q = np.array(input_data["Q"])  # np.ones(p) * 2
            R = np.array(input_data["R"])  # np.ones(m) * 3
            alphe = np.array(input_data['alphe'])  # np.ones(p) * 0.3
            alphemethod = np.array(input_data['alphemethod'])  # np.array(['after', 'before'])
            #积分参数以60秒为单位的增量数据
            integrationInc_mv = np.array(input_data['integrationInc_mv']) * (input_data["APCOutCycle"] / 60)
            integrationInc_ff = np.array(input_data['integrationInc_ff']) * (input_data["APCOutCycle"] / 60) if (
                    'integrationInc_ff' in input_data) else []  # np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])  # [p,m].[p,v]

            # pvusemv = np.array([[1, 1], [1, 1]])
            funneltype = np.array(input_data['funneltype'])  # np.array([[0, 0], [0, 0]])
            runStyle = input_data["runStyle"]
            __dmc = dmc()
            __dmc.init(P, p, M, m, N, input_data["APCOutCycle"], v, A_step_response_sequence,
                       B_step_response_sequence, Q, R, alphe,
                       funneltype, alphemethod, runStyle, integrationInc_mv, integrationInc_ff, False)
            context['dmc'] = __dmc.decode()

        for key, value in context.items():
            context[key] = __dmc.narrayConvert(value)
        OUT = {
            'step': input_data['step']
        }

    elif (input_data['step'] == 'compute'):
        __dmc = dmc()
        __dmc.encode(context['dmc'])
        # 将context内容转换为np
        for key, value in context.items():
            context[key] = __dmc.listConvert(value)
        yk_c = np.array(input_data['y0']).reshape((__dmc.p, 1))  # np.ones((p, 1)) * 2
        spk = np.array(input_data['wi']).reshape((__dmc.p, 1))  # np.ones((p, 1)) * 3
        # param_yk_c_matrix = np.kron(np.eye(p), np.ones(P).reshape(-1, 1))
        # spk_vector = np.dot(param_yk_c_matrix, spk)

        mv0 = np.array(input_data['U']).reshape((__dmc.m, 1))  # np.array([[0], [0]])
        dmvmax = np.array(input_data['limitDU'])[:, 1].reshape((__dmc.m, 1))  # np.array([[2], [10]])
        dmvmin = np.array(input_data['limitDU'])[:, 0].reshape((__dmc.m, 1))  # np.array([[0], [0]])
        mvmax = np.array(input_data['limitU'])[:, 1].reshape((__dmc.m, 1))  # np.array([[100], [100]])
        mvmin = np.array(input_data['limitU'])[:, 0].reshape((__dmc.m, 1))  # np.array([[-100], [-100]])
        deadZones = np.array(input_data['deadZones']).reshape((__dmc.p, 1))  # np.array([[2], [2]])
        funelInitValues = np.array(input_data['funelInitValues']).reshape((__dmc.p, 1))  # np.array([[7], [7]])
        '''计算过程参数'''
        '''
           ff_k最新k时刻是已知的，且ff全部初始化成ff_k,dff_k在初始值的时候=0，更新时，将ff向左平移，把ff_k填进去
           ff=[ff_k-n+1,..ff_k-1,ff_k]
           dff=[dff_k-n+1,..dff_k-1,dff_k]

           mv_k最新k时刻是未知的，读到的只是mv_k-1的值，dmv_k还未计算出来，可以暂时认为是0，在初始值的时,mv全部初始化成mv_k-1,dmv全部初始化成0，更新将k时候的值填写到mv_k-1的位置上去
           mv=[mv_k-n+1,..mv_k-1,mv_k]
           dmv=[dmv_k-n+1,..dmv_k-1,dmv_k]

        '''

        if 'ff' not in context:
            context['ff'] = np.dot(np.array(input_data['FF']).reshape((__dmc.v, 1)), np.ones((1, __dmc.N))) if (
                    'FF' in input_data) else []  # 前馈值
            context['dff'] = np.zeros((__dmc.v, __dmc.N)) if (
                    'FF' in input_data) else []  # np.array([[0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.3]]), np.array([[0.1, 0.1, 0.1, 0.2], [0.1, 0.1, 0.1, 0.3]])
        if 'mvfb' not in context:
            context['mvfb'] = np.dot(np.array(input_data['UFB']).reshape((__dmc.m, 1)), np.ones((1, __dmc.N)))
            context['dmv'] = np.zeros((__dmc.m, __dmc.N))
        if 'yk_1_p' not in context:
            context['yk_1_p'] = yk_c

        leftmove_1 = np.eye(__dmc.N, k=-1)
        leftmove_1[-1, -1] = 1
        leftmove_0 = np.eye(__dmc.N, k=-1)

        if __dmc.v != 0:
            context['ff'] = np.dot(context['ff'], leftmove_1)
            context['dff'] = np.dot(context['dff'], leftmove_1)
            context['ff'][:, -1] = np.array(input_data['FF']).reshape(-1)  # (__dmc.v, 1) modify by 2021:3:22
            context['dff'][:, -1] = (context['ff'][:, -1] - context['ff'][:, -2]).reshape(-1)

        context['mvfb'][:, -1] = np.array(input_data['UFB'])
        context['dmv'][:, -1] = context['mvfb'][:, -1] - context['mvfb'][:, -2]

        context['mvfb'] = np.dot(context['mvfb'], leftmove_1)
        context['dmv'] = np.dot(context['dmv'], leftmove_0)

        e = context['yk_1_p'] - yk_c
        yk_p_N = __dmc.predict(context['mvfb'], context['dmv'], context['ff'], context['dff'], yk_c)
        cellk = np.zeros(__dmc.N)
        cellk[0] = 1
        L = np.kron(np.eye(__dmc.p), cellk)
        context['yk_1_p'] = np.dot(L, yk_p_N)

        funnels, funnelswithtype = __dmc.buildfunel(spk, deadZones, funelInitValues)
        spk_vector = __dmc.biuldspkvectorwhithFunel(yk_p_N, funnelswithtype)

        re_dmv = __dmc.rolloptimization(spk_vector, yk_c, mv0, dmvmax, dmvmin, mvmax, mvmin, yk_p_N)

        dmv_constraint = __dmc.mvconstraint(mv0, re_dmv, mvmax, mvmin, dmvmax, dmvmin)
        writemv = mv0 + dmv_constraint
        if __dmc._DEBUG:
            # fig, ax = plt.subplots()
            # X = np.arange(0, __dmc.N, 1)
            # plt.plot( yk_p_N[:, 0], 'g',label="yk_p_N")
            # plt.plot(funnels[:,0], 'r',label="funnelsup")
            # plt.plot(funnels[:,1], 'b',label="funnelsdown")
            # plt.plot(spk_vector, 'k', label="spk_vector")
            # plt.show()
            pass
        # print(context['dff'][:, -1])
        # print(type(context['dff'][:, -1]))
        OUT = {
            'mv': writemv.reshape(-1).tolist(),
            'dmv': dmv_constraint.reshape(-1).tolist(),
            'e': e.reshape(-1).tolist(),
            'dff': context['dff'][:, -1].reshape(-1).tolist() if len(context['dff']) != 0 else [],
            'predict': yk_p_N.reshape(-1).tolist(),
            'funelupAnddown': funnels.transpose().tolist(),
            'step': input_data['step']
        }
        # write_resp = requests.post("http://localhost:8080/AILab/python/updateModleData.do", data=payload)
    # print('cost='+str(time.time()-start))

    # f = open("E:\\project\\2021_Project\\algorithmplatform\\Dlls\\model_Weight.txt", 'a')
    # f.write('cost='+str(time.time()-start))
    # f.close()
    #
    for key, value in context.items():
        context[key] = __dmc.narrayConvert(value)

    return OUT

#
# if __name__ == '__main__':
#     context={}
#     input_databuild={"A":[[[0.0,0.0,-0.0,-1.6212159319211483,-3.1548204939117985,-4.605548250930241,-5.977877909530064,-7.276046144584225,-8.504060678805592,-9.66571265544328,-10.764588342351878,-11.804080203566578,-12.787397372564484,-13.71757555954536,-14.597486423317607,-15.429846436722581,-16.21722527296668,-16.962053738751564,-17.666631278693824,-18.33313307420181,-18.963616758725408,-19.560028770110137,-20.12421035966667,-20.657903276507007,-21.16275514469618,-21.64032454981974,-22.092085850670486,-22.51943373090907,-22.923687504750475,-23.30609518996885,-23.667837360795055,-24.01003079260155,-24.3337319096266,-24.639940046381593,-24.929600532810216,-25.20360761272397,-25.462807204523877,-25.707999512731334,-25.939941498390443,-26.159349215968486,-26.366900023969066,-26.563234676082523,-26.748959299329496,-26.92464726530453,-27.090840960296717,-27.24805345975203,-27.396770112246873,-27.537450037862826,-27.670527545588445,-27.796413474123895,-27.9154964602278,-28.02814413852196,-28.134704276457967,-28.23550584794966,-28.33086004898585,-28.421061258358858,-28.506387946474728,-28.587103535050883,-28.663457210355237,-28.735684692497447,-28.80400896314728,-28.868640953926633,-28.92978019760054,-28.987615444077456,-29.04232524312054,-29.09407849556893,-29.143034974770718,-29.189345819837417,-29.23315400224268,-29.274594767205823,-29.313796051222653,-29.35087887703282,-29.38595772724285,-29.419140897758417,-29.450530832116932,-29.480224437752646,-29.508313385170606,-29.534884390953067,-29.560019485472136,-29.583796266135046]],[[0.0,0.0,-0.0,-1.6212159319211483,-3.1548204939117985,-4.605548250930241,-5.977877909530064,-7.276046144584225,-8.504060678805592,-9.66571265544328,-10.764588342351878,-11.804080203566578,-12.787397372564484,-13.71757555954536,-14.597486423317607,-15.429846436722581,-16.21722527296668,-16.962053738751564,-17.666631278693824,-18.33313307420181,-18.963616758725408,-19.560028770110137,-20.12421035966667,-20.657903276507007,-21.16275514469618,-21.64032454981974,-22.092085850670486,-22.51943373090907,-22.923687504750475,-23.30609518996885,-23.667837360795055,-24.01003079260155,-24.3337319096266,-24.639940046381593,-24.929600532810216,-25.20360761272397,-25.462807204523877,-25.707999512731334,-25.939941498390443,-26.159349215968486,-26.366900023969066,-26.563234676082523,-26.748959299329496,-26.92464726530453,-27.090840960296717,-27.24805345975203,-27.396770112246873,-27.537450037862826,-27.670527545588445,-27.796413474123895,-27.9154964602278,-28.02814413852196,-28.134704276457967,-28.23550584794966,-28.33086004898585,-28.421061258358858,-28.506387946474728,-28.587103535050883,-28.663457210355237,-28.735684692497447,-28.80400896314728,-28.868640953926633,-28.92978019760054,-28.987615444077456,-29.04232524312054,-29.09407849556893,-29.143034974770718,-29.189345819837417,-29.23315400224268,-29.274594767205823,-29.313796051222653,-29.35087887703282,-29.38595772724285,-29.419140897758417,-29.450530832116932,-29.480224437752646,-29.508313385170606,-29.534884390953067,-29.560019485472136,-29.583796266135046]]],"B":[[[0.03997779265339241,0.07675913749590951,0.11059960838317251,0.14173434461358714,0.17037968478532667,0.19673467001732275,0.2209824269793429,0.24329144034108813,0.2638167234818781,0.2827008955955588,0.30007517267485595,0.3160602792609958,0.3307672872938494,0.3442983878913246,0.35674760142068374,0.3682014307956938,0.37873946253902085,0.3884349197863287,0.3973551710746112,0.4055621984500555,0.4131130281480673,0.4200601268380227,0.42645176618603875,0.43233235826891425,0.437742764169853,0.44272057790023495,0.4473003876202561,0.45151401597351914,0.45539074120553186,0.4589575006025454,0.46223907766432226,0.4652582743113954,0.46803606932339553,0.47059176410934894,0.47294311682282597,0.4751064657538342,0.4770968428548409,0.47892807818975536,0.48061289603163226,0.4821630032768266,0.4835891707899431,0.4849013082448029,0.4861085329814564,0.48721923335769396,0.48824112703524897,0.48918131460569375,0.49004632892864547,0.49084218052510686,0.49157439934135577,0.49224807317357894,0.4928678830202408,0.4934381356078339,0.4939627933160133,0.4944455017100495,0.494889614871908,0.4952982187059677,0.4956741523813165,0.4960200280596163,0.4963382490456143,0.4966310264864199,0.49690039473558123,0.49714822548871784,0.49737624078893033,0.4975860249923555,0.49777903577700766,0.4979566142714025,0.49811999437334,0.49827031132359967,0.49840860959411976,0.4985358501454735,0.498652917104069,0.49876062390546994,0.4988597189465233,0.4989508907855683,0.49903477292685894,0.49911194822244587,0.49918295292210296,0.4992482803994394,0.4993083845800888,0.499363683095795]],[[0.07995558530678482,0.15351827499181903,0.22119921676634502,0.2834686892271743,0.34075936957065334,0.3934693400346455,0.4419648539586858,0.48658288068217626,0.5276334469637562,0.5654017911911176,0.6001503453497119,0.6321205585219916,0.6615345745876988,0.6885967757826492,0.7134952028413675,0.7364028615913876,0.7574789250780417,0.7768698395726574,0.7947103421492224,0.811124396900111,0.8262260562961345,0.8401202536760454,0.8529035323720775,0.8646647165378285,0.875485528339706,0.8854411558004699,0.8946007752405122,0.9030280319470383,0.9107814824110637,0.9179150012050908,0.9244781553286445,0.9305165486227908,0.9360721386467911,0.9411835282186979,0.9458862336456519,0.9502129315076684,0.9541936857096818,0.9578561563795107,0.9612257920632645,0.9643260065536532,0.9671783415798862,0.9698026164896058,0.9722170659629128,0.9744384667153879,0.9764822540704979,0.9783626292113875,0.9800926578572909,0.9816843610502137,0.9831487986827115,0.9844961463471579,0.9857357660404816,0.9868762712156678,0.9879255866320266,0.988891003420099,0.989779229743816,0.9905964374119354,0.991348304762633,0.9920400561192326,0.9926764980912286,0.9932620529728398,0.9938007894711625,0.9942964509774357,0.9947524815778607,0.995172049984711,0.9955580715540153,0.995913228542805,0.99623998874668,0.9965406226471993,0.9968172191882395,0.997071700290947,0.997305834208138,0.9975212478109399,0.9977194378930466,0.9979017815711366,0.9980695458537179,0.9982238964448917,0.9983659058442059,0.9984965607988788,0.9986167691601776,0.99872736619159]]],"integrationInc_mv":[[0.0],[0.0]],"funneltype":[[0.0,0.0],[0.0,1.0]],"alphemethod":["after","after"],"alphe":[0.3,0.3],"m":1,"M":2,"integrationInc_ff":[[0.0],[0.0]],"N":80,"p":2,"P":75,"fnum":1,"Q":[0.002,0.002],"APCOutCycle":10,"R":[1.0],"pvusemv":[[1],[1]],"enable":1,"msgtype":"build"}
#
#     print(main(input_databuild,context))
#     input_datacompute=[{"UFB":[37.00229],"FF":[170.0],"funelInitValues":[10.0,10.0],"wi":[170.0,170.0],"limitDU":[[0.0,0.5]],"U":[37.00229],"enable":1,"y0":[154.09036,154.09036],"deadZones":[10.0,0.0],"limitU":[[36.5,42.0]],"msgtype":"compute"},{"UFB":[37.006],"FF":[170.0],"funelInitValues":[10.0,10.0],"wi":[170.0,170.0],"limitDU":[[0.0,0.5]],"U":[37.006],"enable":1,"y0":[154.09036,154.09036],"deadZones":[10.0,0.0],"limitU":[[36.5,42.0]],"msgtype":"compute"}]
#     for testindex in range(0,2):
#         print('testindex:',testindex)
#
#         print(main(input_datacompute[testindex],context))
#
