from numpy.typing import NDArray
import numpy as np

class Population:
    def __init__(self,
                 populations : list,
                 tau: float,
                 G: dict,
                 E: dict,
                 gamma: dict,
                 V : float,
                 dV : float,
                 type: str = "excitatory",
                 sampling_rate: float = 100.0,
                 Q_max: float = 5.0,
                 C : float = 1.0,
                 theta : float = -40.0,
                 sigma : float = 5.0,
                 phi_stim : float = 0.0,
                 phi_sigma : float = 0.001,)-> None:
        
        # Constants
        self.tau: float = tau
        self.G = G
        self.E = E
        self.gamma = gamma
        self.sampling_rate = sampling_rate
        self.dt = 1/sampling_rate
        self.type = type # ["excitatory","inhibitory"] pas plus
        self.Q_max = 5.0
        self.C = 1.0
        self.theta = -40.0
        self.sigma = 5.0
        self.phi_stim = 0.0
        self.phi_sigma = 0.001


        # Generated variables 
        N_excitatory = len([pop for pop, strenght in populations if strenght > 0])
        N_inhibitory = len([pop for pop, strenght in populations if strenght < 0])

        # --- Excitatory ---
        self.Se = np.zeros(N_excitatory)
        #[[]*N_excitatory]
        self.dSe = np.zeros(N_excitatory)
        self.d2Se = np.zeros(N_excitatory)
        self.Ne = np.zeros(N_excitatory)

        # --- Inhibitory ---
        self.Si = np.zeros(N_inhibitory)
        self.dSi = np.zeros(N_inhibitory)
        self.d2Si = np.zeros(N_inhibitory)
        self.Ni = np.zeros(N_inhibitory)

        # --- Polarization/Depolarization ---
        self.V = V
        self.dV = dV
    
    def compute_S(self,
                 populations : list[Population]):

        for e,population in enumerate(populations):
            self.d2Se[e] = self.gamma["e"]**2 # Facteur initial
            self.d2Se[e] *= self.Ne[e]*(population.get_Q("excitatory")+self.phi()-self.Se[e]) # Activité excitatrice
            self.d2Se[e] -= 2*self.gamma["e"]*self.dSe[e] # Modulation par la vitesse d'ouverture

            self.d2Si[e] = self.gamma["i"]**2 # Facteur initial
            self.d2Si[e] *= self.Ni[e]*(population.get_Q("inhibitory")-self.Si[e]) # Activité inhibitrice
            self.d2Si[e] -= 2*self.gamma["i"]*self.dSi[e] # Modulation par la vitesse d'ouverture

        self.dSe += self.d2Se*self.dt
        self.Se += self.dSe*self.dt
        self.Se = np.clip(self.Se, 0, 1)

        
        self.dSi += self.d2Si*self.dt
        self.Si += self.dSi*self.dt
        self.Si = np.clip(self.Si,0,1)
        
    def get_Q(self,
             type: float = "excitatory"):
        if self.type == type:
            return self.Q_max / (1+ np.exp((-self.C)*(self.V - self.theta)/self.sigma))
        else:
            return 0

    def phi(self):
        return np.random.normal(self.phi_stim, self.phi_sigma)

    def get_V(self):
        self.dV = (-self.I_l() - self.I_ampa() - self.I_gaba()) / self.tau
        self.V += self.dV * self.dt
    
    def I_l(self)->float:
        return self.G["l"]*(self.V-self.E["l"])

    def I_ampa(self)->float:
        return np.sum(self.G["ampa"]*self.Se*(self.V-self.E["ampa"]))

    def I_gaba(self)->float:
        return np.sum(self.G["gaba"]*self.Si*(self.V-self.E["gaba"]))
    
    def update_states(self,
                      populations : list[Population],
                      connections : list[float]) -> None:
        excitatory_signal = [(pop.V, strenght) for pop, strenght in zip(populations, connections) if strenght > 0]
        inhibitory_signal = [(pop.V, strenght) for pop, strenght in zip(populations, connections) if strenght < 0]

        d2Sek = self.update_S_ek(excitatory_signal)
        d2Sik = self.update_S_ik(inhibitory_signal)


    def update_S_ek(self,
                    excitatory_signal : list[tuple[float]])->float:

        d2S = (self.gamma["e"]**2) # float

        excitatory_signal = [(self.Q(v), strenght) for v,strenght in excitatory_signal]
        input_signal = np.array([np.prod(signal)+self.noise()-self.Sek[-1] for signal in excitatory_signal])
        d2S = d2S*input_signal # 1D NDArray

        d2S -= len(d2S)*(2*self.gamma["e"]*self.dS[-1]) # 1D NDArray
        d2S = np.sum(d2S)

        self.d2S.append(d2S)
        return d2S

    def dS_ek2(self)->float:
        return (gamma["e"]**2)*(N)
    
    def get_S_ik(self):
        pass