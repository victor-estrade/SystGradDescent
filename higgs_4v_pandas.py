import sys
import collections
import numpy as np
import copy

# ==================================================================================
#  V4 Class and physic computations
# ==================================================================================

class V4:
    """
    A simple 4-vector class to ease calculation
    """
    px=0
    py=0
    pz=0
    e=0
    def __init__(self,apx=0., apy=0., apz=0., ae=0.):
        """
        Constructor with 4 coordinates
        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
        if self.e + 1e-3 < self.p():
            raise ValueError("Energy is too small! Energy: {}, p: {}".format(self.e, self.p()))

    def copy(self):
        return copy.deepcopy(self)
    
    def p2(self):
        return self.px**2 + self.py**2 + self.pz**2
    
    def p(self):
        return np.sqrt(self.p2())
    
    def pt2(self):
        return self.px**2 + self.py**2
    
    def pt(self):
        return np.sqrt(self.pt2())
    
    def m(self):
        return np.sqrt( np.abs( self.e**2 - self.p2() ) ) # abs is needed for protection
    
    def eta(self):
        return np.arcsinh( self.pz/self.pt() )
    
    def phi(self):
        return np.arctan2(self.py, self.px)
    
    def deltaPhi(self, v):
        """delta phi with another v"""
        return (self.phi() - v.phi() + 3*np.pi) % (2*np.pi) - np.pi
    
    def deltaEta(self,v):
        """delta eta with another v"""
        return self.eta()-v.eta()
    
    def deltaR(self,v):
        """delta R with another v"""
        return np.sqrt(self.deltaPhi(v)**2+self.deltaEta(v)**2 )

    def eWithM(self,m=0.):
        """recompute e given m"""
        return np.sqrt(self.p2()+m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):
        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (self.px, self.py,self.pz,self.e,self.pt(),self.eta(),self.phi(),self.m())

    def scale(self,factor=1.): # scale
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = np.abs( factor*self.e )
    
    def scaleFixedM(self,factor=1.): 
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)
    
    def setPtEtaPhiM(self, pt=0., eta=0., phi=0., m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt*np.cos(phi)
        self.py = pt*np.sin(phi)
        self.pz = pt*np.sinh(eta)
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
        copy = self.copy()
        try:
            copy.px += other.px
            copy.py += other.py
            copy.pz += other.pz
            copy.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy

    def __sub__(self, other):
        """sub 2 V4 vectors : v3 = v1 - v2 = v1.__sub__(v2)"""
        copy = self.copy()
        try:
            copy.px -= other.px
            copy.py -= other.py
            copy.pz -= other.pz
            copy.e -= other.e
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

# magic variable
# FIXME : does it really returns sqrt(2) if in dead center ?
def METphi_centrality(aPhi, bPhi, cPhi):
    """
    Calculate the phi centrality score for an object to be between two other objects in phi
    Returns sqrt(2) if in dead center
    Returns smaller than 1 if an object is not between
    a and b are the bounds, c is the vector to be tested
    """
    # Safely compute and set to zeros results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        
        A = np.true_divide(np.sin(cPhi - aPhi), np.sin(bPhi - aPhi))
        A[A == np.inf] = 0
        A = np.nan_to_num(A)
        
        B = np.true_divide(np.sin(bPhi - cPhi), np.sin(bPhi - aPhi))
        B[B == np.inf] = 0
        B = np.nan_to_num(B)
        
        res = (A+B)/np.sqrt(A**2 + B**2)
        res[res == np.inf] = 0
        res = np.nan_to_num(res)
    return res


# another magic variable
def eta_centrality(eta, etaJ1, etaJ2):
    """
    Calculate the eta centrality score for an object to be between two other objects in eta
    Returns 1 if in dead center
    Returns value smaller than 1/e if object is not between
    """
    center = (etaJ1 + etaJ2) / 2.
    
    # Safely compute and set to zeros results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore'):
        width  = 1. / (etaJ1 - center)**2
        width[width == np.inf] = 0
        width = np.nan_to_num(width)
        
    return np.exp(-width * (eta - center)**2)


# ==================================================================================
#  Now we enter in the manipulation procedures (everything works on data inplace)
# ==================================================================================

def label_to_float(data):
    """
    Transform the string labels to float values.
    s -> 1.0
    b -> 0.0

    Works inplace on the given data !

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
    """
    if data['Label'].dtype == object:
        #copy entry in human usable form
        data["Label"] = (data["Label"] == 's').astype("float")
    else:
        pass

# ==================================================================================
def getDetailLabel(origWeight, Label, num=True):
    """
    Given original weight and label, 
    return more precise label specifying the original simulation type.
    
    Args
    ----
        origWeight: the original weight of the event
        Label : the label of the event (can be {"b", "s"} or {0,1})
        num: (default=True) if True use the numeric detail labels
                else use the string detail labels. You should prefer numeric labels.

    Return
    ------
        detailLabel: the corresponding detail label ("W" is the default if not found)

    Note : Could be better optimized but this is fast enough.
    """
    # prefer numeric detail label
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
    # complementary for W detaillabeldict=200
    #previous alphanumeric detail label    
    detail_label_str={
       57207:"S0",
       4613:"S1",
       8145:"S2",
       4610:"S3",
       917703:"Z05",
       5127399:"Z11",
       4435976:"Z12",
       4187604:"Z13",
       2407146:"Z14",
       1307751:"Z15",
       944596:"Z22",
       936590:"Z23",
       1093224:"Z24",
       225326:"Z32",
       217575:"Z33",
       195328:"Z34",
       254338:"Z35",
       2268701:"T"
    }

    if num:
        detailLabelDict = detail_label_num
    else:
        detailLabelDict = detail_label_str

    iWeight=int(1e7*origWeight+0.5)
    detailLabel = detailLabelDict.get(iWeight, "W") # "W" is the default value if not found
    if detailLabel == "W" and (Label != 0 and Label != 'b') :
        raise ValueError("ERROR! if not in detailLabelDict sould have Label==1 ({}, {})".format(iWeight,Label))
    return detailLabel

def add_detail_label(data, num=True):
    """
    Add a 'detailLabel' column with the detailed labels.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        num: (default=True) if True use the numeric detail labels
                else use the string detail labels. You should prefer numeric labels.
    """
    if "origWeight" in data.columns:
        detailLabel = [getDetailLabel(w, label, num=num) for w, label in zip(data["origWeight"], data["Label"])]
    else:
        detailLabel = [getDetailLabel(w, label, num=num) for w, label in zip(data["Weight"], data["Label"])]
    data["detailLabel"] = detailLabel

# ==================================================================================

def rounding(data, DECIMALS=3):
    """Fix precision to 3 decimals (per default)"""
    columns = [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_deltar_tau_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_tau",
        "DER_met_phi_centrality",
        "DER_lep_eta_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_met_sumet",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_jet_all_pt",
    ]

    for c in columns:
        if c in data:
            data[c] = data[c].round(decimals=DECIMALS)

# ==================================================================================
# Nasty backgrounds
# ==================================================================================

def nasty_background(data, systBkgNorm):
    """
    Apply a scaling to the weight.
    Keeps the previous weights in the 'origWeight' columns

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    TODO maybe explain why it scales only the background.
    """
    # only a weight manipulation
    if not "origWeight" in data.columns:
        data["origWeight"] = data["Weight"]
    if not "detailLabel" in data.columns:
        add_detail_label(data)
    # scale the weight, arbitrary but reasonable value
    data["Weight"] = ( data["Weight"]*systBkgNorm ).where(data["detailLabel"] == "W", other=data["origWeight"])
    # data.drop(["origWeight", "detailLabel"], axis=1, inplace=True)


# ==================================================================================
# V4 vector constructors for particles
# ==================================================================================

def V4_tau(data):
    vtau = V4() # tau 4-vector
    vtau.setPtEtaPhiM(data["PRI_tau_pt"], data["PRI_tau_eta"], data["PRI_tau_phi"], 0.8)
    # tau mass 0.8 like in original
    return vtau

def V4_lep(data):
    vlep = V4() # lepton 4-vector
    vlep.setPtEtaPhiM(data["PRI_lep_pt"], data["PRI_lep_eta"], data["PRI_lep_phi"], 0.)
    # lep mass 0 (either 0.106 or 0.0005 but info is lost)
    return vlep
    
def V4_met(data):
    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(data["PRI_met"], 0., data["PRI_met_phi"], 0.) # met mass zero
    return vmet

def V4_leading_jet(data):
    vj1 = V4()
    vj1.setPtEtaPhiM(data["PRI_jet_leading_pt"].where( data["PRI_jet_num"] > 0, other=0 ),
                         data["PRI_jet_leading_eta"].where( data["PRI_jet_num"] > 0, other=0 ),
                         data["PRI_jet_leading_phi"].where( data["PRI_jet_num"] > 0, other=0 ),
                         0.) # zero mass
    return vj1

def V4_subleading_jet(data):
    vj2 = V4()
    vj2.setPtEtaPhiM(data["PRI_jet_subleading_pt"].where( data["PRI_jet_num"] > 1, other=0 ),
                     data["PRI_jet_subleading_eta"].where( data["PRI_jet_num"] > 1, other=0 ),
                     data["PRI_jet_subleading_phi"].where( data["PRI_jet_num"] > 1, other=0 ),
                     0.) # zero mass
    return vj2


# ==================================================================================
# Update data
# ==================================================================================

def update_met(data, vmet):
    data["PRI_met"] = vmet.pt()
    data["PRI_met_phi"] = vmet.phi()

def update_jet(data, vj1, vj2, missing_value):
    vjsum = vj1 + vj2
    data["DER_deltaeta_jet_jet"] = vj1.deltaEta(vj2).where(data["PRI_jet_num"] > 1, other=missing_value)
    data["DER_mass_jet_jet"] = vjsum.m().where(data["PRI_jet_num"] > 1, other=missing_value)
    data["DER_prodeta_jet_jet"] = ( vj1.eta() * vj2.eta() ).where(data["PRI_jet_num"] > 1, other=missing_value)

def update_eta_centrality(data, missing_value):
    eta_centrality_tmp = eta_centrality(data["PRI_lep_eta"],
                                        data["PRI_jet_leading_eta"],
                                        data["PRI_jet_subleading_eta"])
    data["DER_lep_eta_centrality"] = eta_centrality_tmp.where(data["PRI_jet_num"] > 1, other=missing_value)

def update_transverse_met_lep(data, vlep, vmet):
    vtransverse = V4()
    vtransverse.setPtEtaPhiM(vlep.pt(), 0., vlep.phi(), 0.) # just the transverse component of the lepton
    vtransverse += vmet
    data["DER_mass_transverse_met_lep"] = vtransverse.m()
    
def update_mass_vis(data, vlep, vtau):
    vltau = vlep + vtau # lep + tau
    data["DER_mass_vis"] = vltau.m()

def update_pt_h(data, vlep, vmet, vtau):
    vltaumet = vlep + vtau + vmet # lep + tau + met
    data["DER_pt_h"] = vltaumet.pt()

def update_detla_R_tau_lep(data, vlep, vtau):
    data["DER_deltar_tau_lep"] = vtau.deltaR(vlep)

def update_pt_tot(data, vj1, vj2, vlep, vmet, vtau):
    vtot = vlep + vtau + vmet + vj1 + vj2
    data["DER_pt_tot"] = vtot.pt()

def update_sum_pt(data, vlep, vtau):
    data["DER_sum_pt"] = vlep.pt() + vtau.pt() + data["PRI_jet_all_pt"] # sum_pt is the scalar sum

def update_pt_ratio_lep_tau(data, vlep, vtau):
    data["DER_pt_ratio_lep_tau"] = vlep.pt()/vtau.pt()
    
def update_met_phi_centrality(data):
    data["DER_met_phi_centrality"] = METphi_centrality(data["PRI_lep_phi"], data["PRI_tau_phi"], data["PRI_met_phi"])

def update_all(data, vj1, vj2, vlep, vmet, vtau, missing_value):
    update_met(data, vmet)
    update_jet(data, vj1, vj2, missing_value)
    update_eta_centrality(data, missing_value)
    update_transverse_met_lep(data, vlep, vmet)
    update_mass_vis(data, vlep, vtau)
    update_pt_h(data, vlep, vmet, vtau)
    update_detla_R_tau_lep(data, vlep, vtau)
    update_pt_tot(data, vj1, vj2, vlep, vmet, vtau)
    update_sum_pt(data, vlep, vtau)
    update_pt_ratio_lep_tau(data, vlep, vtau)
    update_met_phi_centrality(data)


# ==================================================================================
# TES : Tau Energy Scale
# ==================================================================================
def tau_energy_scale(data, scale=1.0, missing_value=0.0):
    """
    Manipulate one primary input : the PRI_tau_pt and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_tau_pt <-- PRI_tau_pt * scale
        missing_value : (default=-999.0) the value used to code missing value. 
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
    # scale tau energy scale, arbitrary but reasonable value
    vtau_original = V4_tau(data) # tau 4-vector
    data["PRI_tau_pt"] *= scale 

    # first built 4-vectors
    vtau = V4_tau(data) # tau 4-vector
    vlep = V4_lep(data) # lepton 4-vector
    vmet = V4_met(data) # met 4-vector
    vj1 = V4_leading_jet(data) # first jet if it exists
    vj2 = V4_subleading_jet(data) # second jet if it exists

    # fix MET according to tau pt change
    vtau_original.scaleFixedM( scale - 1.0 )
    vmet -= vtau_original
    vmet.pz = 0.
    vmet.e = vmet.eWithM(0.)

    update_all(data, vj1, vj2, vlep, vmet, vtau, missing_value)


# ==================================================================================
# JES : Jet Energy Scale
# ==================================================================================
def jet_energy_scale(data, scale=1.0, missing_value=0.0):
    """
    Manipulate jet primaries input and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_jet_pt <-- PRI_jet_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.
    """
    vj1_original = V4_leading_jet(data) # first jet if it exists
    vj2_original = V4_subleading_jet(data) # second jet if it exists
    # scale jet energy, arbitrary but reasonable value
    data["PRI_jet_leading_pt"] *= scale
    data["PRI_jet_subleading_pt"] *= scale
    data["PRI_jet_all_pt"] *= scale

    # first built 4-vectors
    vtau = V4_tau(data) # tau 4-vector
    vlep = V4_lep(data) # lepton 4-vector
    vmet = V4_met(data) # met 4-vector
    vj1 = V4_leading_jet(data) # first jet if it exists
    vj2 = V4_subleading_jet(data) # second jet if it exists

    # fix MET according to jet pt change
    vj1_original.scaleFixedM( scale - 1.0 )
    vj2_original.scaleFixedM( scale - 1.0 )
    vmet -= vj1_original + vj2_original
    vmet.pz = 0.
    vmet.e = vmet.eWithM(0.)

    update_all(data, vj1, vj2, vlep, vmet, vtau, missing_value)


# ==================================================================================
# LES : Lep Energy Scale
# ==================================================================================
def lep_energy_scale(data, scale=1.0, missing_value=0.0):
    """
    Manipulate one primary input : the PRI_lep_pt and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_jet_pt <-- PRI_jet_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.
    """
    vlep_original = V4_lep(data) # lepton 4-vector
    # scale jet energy, arbitrary but reasonable value
    data["PRI_lep_pt"] *= scale 

    # first built 4-vectors
    vtau = V4_tau(data) # tau 4-vector
    vlep = V4_lep(data) # lepton 4-vector
    vmet = V4_met(data) # met 4-vector
    vj1 = V4_leading_jet(data) # first jet if it exists
    vj2 = V4_subleading_jet(data) # second jet if it exists

    # fix MET according to jet pt change
    vlep_original.scaleFixedM( scale - 1.0 )
    vmet -= vlep_original
    vmet.pz = 0.
    vmet.e = vmet.eWithM(0.)

    update_all(data, vj1, vj2, vlep, vmet, vtau, missing_value)

# ==================================================================================
# Soft term
# ==================================================================================
def soft_term(data, sigma_met=3.0, missing_value=0.0):
    """
    Manipulate MET primaries input and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        sigma_met : the mean energy (default = 3 GeV) of the missing v4.
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.
    """

    # Compute the missing v4 vector
    SIZE = data.shape[0]
    v4_soft_term = V4()
    v4_soft_term.px = np.random.normal(0, sigma_met, size=SIZE)
    v4_soft_term.py = np.random.normal(0, sigma_met, size=SIZE)
    v4_soft_term.pz = np.zeros(SIZE)
    v4_soft_term.e = v4_soft_term.eWithM(0.)

    # first built 4-vectors
    vtau = V4_tau(data) # tau 4-vector
    vlep = V4_lep(data) # lepton 4-vector
    vmet = V4_met(data) # met 4-vector
    vj1 = V4_leading_jet(data) # first jet if it exists
    vj2 = V4_subleading_jet(data) # second jet if it exists

    # fix MET according to soft term
    vmet = vmet + v4_soft_term

    update_all(data, vj1, vj2, vlep, vmet, vtau, missing_value)


def cut_tau_pt(data, cut_threshold=22.0):
    data_cut = data[data['PRI_tau_pt'] > cut_threshold]
    return data_cut
