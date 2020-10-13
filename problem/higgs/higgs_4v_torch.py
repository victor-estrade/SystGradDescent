import sys
import collections
import torch
import numpy as np



def normalize_weight(batch, background_luminosity=410999.84732187376, 
                              signal_luminosity=691.9886077135781):
    """Normalizes weight inplace"""
    w = batch['Weight']
    y = batch['Label']
    batch['Weight'] = compute_normalized_weight(w, y, background_luminosity=background_luminosity, signal_luminosity=signal_luminosity)


def compute_normalized_weight(w, y, 
                              background_luminosity=410999.84732187376, 
                              signal_luminosity=691.9886077135781):
    """Normalize the given weight to assert that the luminosity is the same as the nominal.
    Returns the normalized weight vector/Series
    """
    background_weight_sum = w[y==0].sum()
    signal_weight_sum = w[y==1].sum()
    w_new = w.clone().detach()
    w_new[y==0] = w[y==0] * ( background_luminosity / background_weight_sum )
    w_new[y==1] = w[y==1] * ( signal_luminosity / signal_weight_sum )
    return w_new


def split_data_label_weights(batch, feature_names):
    X = torch.cat([batch[name].view(-1, 1) for name in feature_names], axis=1)
    y = batch["Label"]
    W = batch["Weight"]
    return X, y, W


def asinh(x):
    """
    $$ asinh(x) = ln(x + \sqrt{ x^2 +1 }) $$
    """
    return torch.log(x + torch.sqrt( (x*x) + torch.ones_like(x)))


class V4:
    """
    A simple 4-vector class to ease calculation
    """
    px=0
    py=0
    pz=0
    e=0
    def __init__(self, apx=0., apy=0., apz=0., ae=0.):
        """
        Constructor with 4 coordinates
        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
#         if self.e + 1e-3 < self.p():
#             raise ValueError("Energy is too small! Energy: {}, p: {}".format(self.e, self.p()))

    def copy(self):
        new_v4 = V4()
        new_v4.px = self.px.clone().detach()
        new_v4.py = self.py.clone().detach()
        new_v4.pz = self.pz.clone().detach()
        new_v4.e = self.e.clone().detach()
        return new_v4
    
    def p2(self):
        return self.px**2 + self.py**2 + self.pz**2
    
    def p(self):
        return torch.sqrt(self.p2())
    
    def pt2(self):
        return self.px**2 + self.py**2
    
    def pt(self):
        return torch.sqrt(self.pt2())
    
    def m(self):
        return torch.sqrt( torch.abs( self.e**2 - self.p2() ) + sys.float_info.epsilon ) # abs and epsilon are needed for protection
    
    def eta(self):
        return asinh( self.pz/self.pt() )
    
    def phi(self):
        return torch.atan2(self.py, self.px)
    
    def deltaPhi(self, v):
        """delta phi with another v"""
        return (self.phi() - v.phi() + 3*np.pi) % (2*np.pi) - np.pi
    
    def deltaEta(self,v):
        """delta eta with another v"""
        return self.eta()-v.eta()
    
    def deltaR(self,v):
        """delta R with another v"""
        return torch.sqrt(self.deltaPhi(v)**2+self.deltaEta(v)**2 )

    def eWithM(self,m=0.):
        """recompute e given m"""
        return torch.sqrt(self.p2()+m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):
        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (self.px, self.py,self.pz,self.e,self.pt(),self.eta(),self.phi(),self.m())

    def scale(self,factor=1.): # scale
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = torch.abs( factor*self.e )
    
    def scaleFixedM(self,factor=1.): 
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)
    
    def setPtEtaPhiM(self, pt=0., eta=0., phi=0., m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt*torch.cos(phi)
        self.py = pt*torch.sin(phi)
        self.pz = pt*torch.sinh(eta)
        self.e = self.eWithM(m)
    
    def sum(self, v):
        """Add another V4 into self"""
        self.px += v.px
        self.py += v.py
        self.pz += v.pz
        self.e += v.e
    
    def __iadd__(self, other):
        """Add another V4 into self"""
        try:
            self.px += other.px
            self.py += other.py
            self.pz += other.pz
            self.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self
    
    def __add__(self, other):
        """Add 2 V4 vectors : v3 = v1 + v2 = v1.__add__(v2)"""
        copy = V4()
        try:
            copy.px = self.px + other.px
            copy.py = self.py + other.py
            copy.pz = self.pz + other.pz
            copy.e = self.e + other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy

    def __sub__(self, other):
        """sub 2 V4 vectors : v3 = v1 - v2 = v1.__sub__(v2)"""
        copy = V4()
        try:
            copy.px = self.px - other.px
            copy.py = self.py - other.py
            copy.pz = self.pz - other.pz
            copy.e = self.e - other.e
        except AttributeError:
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy

    def __isub__(self, other):
        """Sub another V4 into self"""
        try:
            self.px -= other.px
            self.py -= other.py
            self.pz -= other.pz
            self.e -= other.e
        except AttributeError:
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self


def safe_division(x, y):
    """ x/y """
    y_ok = (y != 0.0)
    y_ok = y_ok.type(y.type())
    safe_y = y_ok * y + (1-y_ok)
    unsafe_op = x / safe_y
    return unsafe_op


# magic variable
# FIXME : does it really returns sqrt(2) if in dead center ?
def METphi_centrality(aPhi, bPhi, cPhi):
    x = torch.sin(bPhi - aPhi)    
    caPhi = torch.sin(cPhi - aPhi)
    bcPhi = torch.sin(bPhi - cPhi)
    A = safe_division(caPhi, x)
    B = safe_division(bcPhi, x)
    res = (A+B) / torch.sqrt(A**2 + B**2)
    return res


# another magic variable
def eta_centrality(eta, etaJ1, etaJ2):
    """
    Calculate the eta centrality score for an object to be between two other objects in eta
    Returns 1 if in dead center
    Returns value smaller than 1/e if object is not between
    """
    center = (etaJ1 + etaJ2) / 2.
    x = etaJ1 - center
    width = safe_division(torch.ones_like(x), (x*x))
    return torch.exp(-width * (eta - center)**2)

# ==================================================================================
# V4 vector constructors for particles
# ==================================================================================

def V4_tau(batch):
    vtau = V4() # tau 4-vector
    vtau.setPtEtaPhiM(batch["PRI_tau_pt"], batch["PRI_tau_eta"], 
                      batch["PRI_tau_phi"],torch.tensor(0.8, requires_grad=True))
    # tau mass 0.8 like in original
    return vtau

def V4_lep(batch):
    vlep = V4() # lepton 4-vector
    vlep.setPtEtaPhiM(batch["PRI_lep_pt"], batch["PRI_lep_eta"], 
                      batch["PRI_lep_phi"], torch.tensor(0., requires_grad=True))
    # lep mass 0 (either 0.106 or 0.0005 but info is lost)
    return vlep

def V4_met(batch):
    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(batch["PRI_met"], 
                      torch.tensor(0., requires_grad=True),
                      batch["PRI_met_phi"],
                      torch.tensor(0., requires_grad=True)) # met mass zero
    return vmet

def V4_leading_jet(batch):
    zeros_batch = torch.zeros_like(batch["PRI_tau_pt"])
    vj1 = V4()
    print(batch["PRI_jet_num"].dtype, "PRI_jet_num")
    print(zeros_batch.dtype, "zeros_batch")
    print(batch["PRI_jet_leading_pt"].dtype, "PRI_jet_leading_pt")
    print(batch["PRI_jet_leading_eta"].dtype, "PRI_jet_leading_eta")
    print(batch["PRI_jet_leading_phi"].dtype, "PRI_jet_leading_phi")
    vj1.setPtEtaPhiM(torch.where(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_pt"], zeros_batch),
                     torch.where(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_eta"], zeros_batch),
                     torch.where(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_phi"], zeros_batch),
                         torch.tensor(0., requires_grad=True)) # zero mass
    return vj1

def V4_subleading_jet(batch):
    zeros_batch = torch.zeros_like(batch["PRI_tau_pt"])
    vj2 = V4()
    vj2.setPtEtaPhiM(torch.where(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_pt"], zeros_batch),
                     torch.where(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_eta"], zeros_batch),
                     torch.where(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_phi"], zeros_batch),
                     torch.tensor(0., requires_grad=True)) # zero mass
    return vj2

# ==================================================================================
# Update data batch
# ==================================================================================

def update_met(batch, vmet):
    batch["PRI_met"] = vmet.pt()
    batch["PRI_met_phi"] = vmet.phi()

def update_jet(batch, vj1, vj2, missing_value_batch):
    vjsum = vj1 + vj2
    batch["DER_deltaeta_jet_jet"] = torch.where(batch["PRI_jet_num"] > 1, vj1.deltaEta(vj2), missing_value_batch )
    batch["DER_mass_jet_jet"] = torch.where(batch["PRI_jet_num"] > 1, vjsum.m(), missing_value_batch )
    batch["DER_prodeta_jet_jet"] = torch.where(batch["PRI_jet_num"] > 1, vj1.eta() * vj2.eta(), missing_value_batch )

def update_eta_centrality(batch, missing_value_batch):
    eta_centrality_tmp = eta_centrality(batch["PRI_lep_eta"],batch["PRI_jet_leading_eta"],batch["PRI_jet_subleading_eta"])                       
    batch["DER_lep_eta_centrality"] = torch.where(batch["PRI_jet_num"] > 1, eta_centrality_tmp, missing_value_batch )

def update_transverse_met_lep(batch, vlep, vmet):
    vtransverse = V4()
    vtransverse.setPtEtaPhiM(vlep.pt(), torch.tensor(0., requires_grad=True), 
                            vlep.phi(), torch.tensor(0., requires_grad=True)) # just the transverse component of the lepton
    vtransverse += vmet
    batch["DER_mass_transverse_met_lep"] = vtransverse.m()

def update_mass_vis(batch, vlep, vtau):
    vltau = vlep + vtau # lep + tau
    batch["DER_mass_vis"] = vltau.m()

def update_pt_h(batch, vlep, vmet, vtau):
    vltaumet = vlep + vtau + vmet # lep + tau + met
    batch["DER_pt_h"] = vltaumet.pt()

def update_detla_R_tau_lep(batch, vlep, vtau):
    batch["DER_deltar_tau_lep"] = vtau.deltaR(vlep)

def update_pt_tot(batch, vj1, vj2, vlep, vmet, vtau):
    vtot = vlep + vtau + vmet + vj1 + vj2
    batch["DER_pt_tot"] = vtot.pt()

def update_sum_pt(batch, vlep, vtau):
    batch["DER_sum_pt"] = vlep.pt() + vtau.pt() + batch["PRI_jet_all_pt"] # sum_pt is the scalar sum

def update_pt_ratio_lep_tau(batch, vlep, vtau):
    batch["DER_pt_ratio_lep_tau"] = vlep.pt()/vtau.pt()

def update_met_phi_centrality(batch):
    batch["DER_met_phi_centrality"] = METphi_centrality(batch["PRI_lep_phi"], batch["PRI_tau_phi"], batch["PRI_met_phi"])

def update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_batch):
    update_met(batch, vmet)
    update_jet(batch, vj1, vj2, missing_value_batch)
    update_eta_centrality(batch, missing_value_batch)
    update_transverse_met_lep(batch, vlep, vmet)
    update_mass_vis(batch, vlep, vtau)
    update_pt_h(batch, vlep, vmet, vtau)
    update_detla_R_tau_lep(batch, vlep, vtau)
    update_pt_tot(batch, vj1, vj2, vlep, vmet, vtau)
    update_sum_pt(batch, vlep, vtau)
    update_pt_ratio_lep_tau(batch, vlep, vtau)
    update_met_phi_centrality(batch)


# ==================================================================================
# Mu reweighting
# ==================================================================================
def mu_reweighting(data, mu=1.0):
    """
    Update signal weights inplace to correspond to the new value of mu
    """
    y = data['Label']
    w = data['Weight']
    w_new = w.clone().detach()
    w_new[y==1] = mu * w[y==1]
    data['Weight'] = w_new


# ==================================================================================
# TES : Tau Energy Scale
# ==================================================================================
def tau_energy_scale(batch, scale=1.0, missing_value=0.0):
    """
    Manipulate one primary input : the PRI_tau_pt and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_tau_pt <-- PRI_tau_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.

    Notes :
    -------
        Recompute :
            - PRI_tau_pt
            - PRI_met
            - PRI_met_phi
            - DER_deltaeta_jet_jet
            - DER_mass_jet_jet
            - DER_prodeta_jet_jet
            - DER_lep_eta_centrality
            - DER_mass_transverse_met_lep
            - DER_mass_vis
            - DER_pt_h
            - DER_deltar_tau_lep
            - DER_pt_tot
            - DER_sum_pt
            - DER_pt_ratio_lep_tau
            - DER_met_phi_centrality
            - DER_mass_MMC
    """
    zeros_batch = torch.zeros_like(batch["PRI_tau_pt"])
    missing_value_batch = zeros_batch + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    # scale tau energy scale, arbitrary but reasonable value
    vtau_original = V4_tau(batch) # tau 4-vector
    batch["PRI_tau_pt"] = batch["PRI_tau_pt"] * scale

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # fix MET according to tau pt change
    vtau_original.scaleFixedM( scale - 1.0 )
    vmet = vmet - vtau_original
    vmet.pz = zeros_batch
    vmet.e = vmet.eWithM(0.)

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_batch)
    return batch


# ==================================================================================
# JES : Jet Energy Scale
# ==================================================================================
def jet_energy_scale(batch, scale=1.0, missing_value=0.0):
    """
    Manipulate jet primaries input and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_jet_pt <-- PRI_jet_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.
    """
    zeros_batch = torch.zeros_like(batch["PRI_tau_pt"])
    missing_value_batch = zeros_batch + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    vj1_original = V4_leading_jet(batch) # first jet if it exists
    vj2_original = V4_subleading_jet(batch) # second jet if it exists
    # scale jet energy, arbitrary but reasonable value
    batch["PRI_jet_leading_pt"] = scale * batch["PRI_jet_leading_pt"]
    batch["PRI_jet_subleading_pt"] = scale * batch["PRI_jet_subleading_pt"]
    batch["PRI_jet_all_pt"] = scale * batch["PRI_jet_all_pt"]

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # fix MET according to jet pt change
    vj1_original.scaleFixedM( scale - 1.0 )
    vj2_original.scaleFixedM( scale - 1.0 )
    vmet = vmet - (vj1_original + vj2_original)
    vmet.pz = zeros_batch
    vmet.e = vmet.eWithM(0.)

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_batch)
    return batch


# ==================================================================================
# LES : Lep Energy Scale
# ==================================================================================
def lep_energy_scale(batch, scale=1.0, missing_value=0.0):
    """
    Manipulate one primary input : the PRI_lep_pt and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_jet_pt <-- PRI_jet_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.
    """
    zeros_batch = torch.zeros_like(batch["PRI_tau_pt"])
    missing_value_batch = zeros_batch + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    vlep_original = V4_lep(batch) # lepton 4-vector
    # scale jet energy, arbitrary but reasonable value
    batch["PRI_lep_pt"] = scale * batch["PRI_lep_pt"]

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # fix MET according to lep pt change
    vlep_original.scaleFixedM( scale - 1.0 )
    vmet = vmet - vlep_original
    vmet.pz = zeros_batch
    vmet.e = vmet.eWithM(0.)

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_batch)
    return batch


# ==================================================================================
# Soft term
# ==================================================================================
def soft_term(batch, sigma_met=3.0, missing_value=0.0):
    """
    Manipulate MET primaries input and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        sigma_met : the mean energy (default = 3 GeV) of the missing v4.
        missing_value : (default=0.0) the value used to code missing value.
            This is not used to find missing values but to write them in feature column that have some.
    """
    zeros_batch = torch.zeros_like(batch["PRI_tau_pt"])
    missing_value_batch = zeros_batch + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # Compute the missing v4 vector
    normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([sigma_met]))
    v4_soft_term = V4()
    v4_soft_term.px = normal.rsample(zeros_batch.size()).type(zeros_batch.type()).view(-1)
    v4_soft_term.py = normal.rsample(zeros_batch.size()).type(zeros_batch.type()).view(-1)
    v4_soft_term.pz = zeros_batch
    v4_soft_term.e = v4_soft_term.eWithM(0.)

    # fix MET according to soft term
    vmet = vmet + v4_soft_term

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_batch)
    return batch


# ==================================================================================
# Nasty Background
# ==================================================================================
def nasty_background(batch, scale=1.5):
    """Apply a scaling to the weight.

    TODO maybe explain why it scales only some backgrounds.
    """
    # element of detail_label_num are not nasty backgrounds
    detail_label_num={
        57207:0, # Signal
        4613:1,
        8145:2,
        4610:3,
        917703: 105, #Z
        5127399:111,
        4435976:112,
        4187604:113,
        2407146:114,
        1307751:115,
        944596:122,
        936590:123,
        1093224:124,
        225326:132,
        217575:133,
        195328:134,
        254338:135,
        2268701:300 #T
        }
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    W = batch["Weight"]
    iWeight = torch.floor(1e7 * W + 0.5).type(torch.ByteTensor)
    is_not_nasty = torch.zeros(iWeight.size(), dtype=torch.uint8)
    for is_recognized in [torch.tensor(k, dtype=torch.uint8) == iWeight for k in detail_label_num]:
        is_not_nasty = is_not_nasty | is_recognized
    is_nasty_bkg = ~is_not_nasty

    # FIXME : For some reason if_then_else function does not work here
    # batch["Weight"] = if_then_else(is_nasty_bkg, scale * W, lambda x : x + 0 )
    # I used another trick to apply conditional assignment
    W_nasty = (is_nasty_bkg.type(W.type()) * scale) * W
    W_original = is_not_nasty.type(W.type()) * W
    batch["Weight"] = W_nasty + W_original
    return batch



def syst_effect(batch, tes=1.0, jes=1.0, les=1.0, missing_value=0.0):
    """
    Manipulate primaries input and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        tes : the tau energy factor applied : PRI_tau_pt <-- PRI_tau_pt * scale
        jes : the jet energy factor applied : PRI_jet_pt <-- PRI_jet_pt * scale
        les : the lep energy factor applied : PRI_lep_pt <-- PRI_lep_pt * scale
        missing_value : (default=0.0) the value used to code missing value.
            This is not used to find missing values but to write them in feature column that have some.

    """
    vtau_original = V4_tau(batch).copy() # tau 4-vector
    vj1_original = V4_leading_jet(batch).copy() # first jet if it exists
    vj2_original = V4_subleading_jet(batch).copy() # second jet if it exists
    vlep_original = V4_lep(batch).copy() # lepton 4-vector

    # scale tau energy scale, arbitrary but reasonable value
    batch["PRI_tau_pt"] = batch["PRI_tau_pt"] * tes
    # scale jet energy, arbitrary but reasonable value
    batch["PRI_jet_leading_pt"] = batch["PRI_jet_leading_pt"] * jes
    batch["PRI_jet_subleading_pt"] = batch["PRI_jet_subleading_pt"] * jes
    batch["PRI_jet_all_pt"] = batch["PRI_jet_all_pt"] * jes
    # scale jet energy, arbitrary but reasonable value
    batch["PRI_lep_pt"] = batch["PRI_lep_pt"] * les

    # build new 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # fix MET according to tau pt change
    vtau_original.scaleFixedM( tes - 1.0 )
    vmet -= vtau_original
    # fix MET according to jet pt change
    vj1_original.scaleFixedM( jes - 1.0 )
    vj2_original.scaleFixedM( jes - 1.0 )
    vmet -= vj1_original + vj2_original
    # fix MET according to jet pt change
    vlep_original.scaleFixedM( les - 1.0 )
    vmet -= vlep_original
    # Fix MET pz to 0 and update e accordingly
    vmet.pz = 0.
    vmet.e = vmet.eWithM(0.)

    zeros_batch = torch.zeros_like(batch["PRI_tau_pt"])
    missing_value_batch = zeros_batch + missing_value

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_batch)
