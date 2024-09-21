from src.model import *

class Estimator_R(Estimator):
    def __init__(self, M=1e6, *inputs):
        self.M = M
        super().__init__(*inputs)

    def train(self, value=1, *inputs):
        self.n[inputs] = min(self.n[inputs]+1, self.M)
        self.v[inputs] += self.delta(value, *inputs)

    def e(self, value=1, *inputs):
        """
        $e^R_{m} = 1-2(Z_R \Delta\hat{R}_m^2)$ 
        """
        return 1-2 * (self.z(*inputs) * self.delta(value, *inputs)**2)

    def z(self, *inputs):
        """
        $Z_R= (R_{\max}-R_{\min})^{-1} = (0 - (-1))^{-1} = 1$  
        """
        return 1

    def delta(self, value, *inputs):
        """
        $\Delta \hat{R}_m = \frac{r- \hat{R}_m(s,a) }{N_m(s,a)+1}$  
        """
        return (value-self.v[inputs]) / (np.sum(self.n[inputs[:-1]])+1)
 
class Estimator_T(Estimator):
    def __init__(self, M=1e6, *inputs):
        self.M = M
        super().__init__(*inputs)

    def train(self, value=1, *inputs):
        S = self.n.shape[0]
        s,a,s_ = inputs

        self.n[inputs] = min(self.n[inputs]+1, self.M)
        self.v[s,a] += self.delta(value, *inputs)
        
    def e(self, value=1, *inputs):
        """
        $e^T_{m} = 1-2(Z_T \sum_{k \in S}\Delta\hat{T}_m(k)^2)$ 
        """
        return 1-2*  self.z(*inputs)  * np.sum(self.delta(value, *inputs)**2)
    
    def z(self, *inputs):
        """
        $Z_T= \frac{1}{2}(N(s,a)+1)^2$
        """
        s,a,_ = inputs
        return (1/2)*(np.sum(self.n[s,a])+1)**2    

    def delta(self, value=1, *inputs):
        """
        $
            \Delta \hat{T}_m(k) = 
                \begin{cases}
                    \frac{1- \hat{T}_m(s,a,k) }{N_m(s,a)+1} , & \text{if } k = s'\\
                    \\
                    \frac{0- \hat{T}_m(s,a,k) }{N_m(s,a)+1}, & \text{if } k \neq s'
                \end{cases} 
        $  
        """
        S = self.n.shape[0]
        s,a,s_ = inputs
        d = lambda k: (
                (value-self.v[s,a,k])/(np.sum(self.n[s,a])+1) 
                    if k==s_ else 
                (0-self.v[s,a,k])/(np.sum(self.n[s,a])+1)
            )
        return np.array([d(k) for k in range(S)])

class Model(Model):
    def __init__(self, S,A, Omega=.5, M=1e2, rho=.9):
        r = Estimator_R(M, len(S), len(A) ,len(S))
        t = Estimator_T(M, len(S), len(A) ,len(S))
        super().__init__(S,A, t,r)

        self.Omega = Omega
        self.M = M
        self.rho = rho
        self._E = 0 

    def e(self, s,a,s_,r):
        nm = min(np.sum(self.t.n[s,a])+1, self.t.M)
        cm = nm/self.t.M
        return cm*(self.Omega * self.r.e(r, s,a,s_) + (1-self.Omega) * self.t.e(1, s,a,s_))
    
    def E(self, s,a,s_,r):
        self._E += self.rho*(self.e(s,a,s_,r) - self._E)
        return self._E



class RLCD:
    def __init__(self, S,A, Emin=-0.01, Omega=0, rho=.9, M=1e2):
        self.S = S
        self.A = A
        self.Models = []
        self.current_model = None
        self.Emin = Emin
        self.Omega = Omega
        self.rho = rho
        self.M = M
        self.new_model()

    def new_model(self, M=1e2):
        self.current_model = Model(self.S, self.A, self.Omega, self.M, self.rho)
        self.Models.append(self.current_model)

    def learn(self, s,a,s_,r, log=False):
        E = [m.E(s,a,s_,r) for m in self.Models]
        self.current_model = self.Models[ np.argmax(E) ]
        new_model = self.current_model._E < self.Emin
        
        if log:
            print('E: ', E, new_model)
        if new_model:
            self.new_model()

        self.current_model.learn(s,a,s_,r)
    
    def sample(self, s, a):
        sims = [m.sample(s,a) for m in self.Models]
        E = [m.E(s,a,s_,r) for m, (s_,r) in zip(self.Models, sims)]
        return sims[ np.argmax(E) ]