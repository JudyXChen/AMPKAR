"""
    Model 5+CaMKK2 — MA nonessential with beta parameterization + calcium-CaMKK2 cascade.

    Extends MA_nonessential_phos_diffrax.py (Model 5) with a mechanistic calcium-
    CaMKK2 activation module based on Nate's dev_CaMKK branch:
        - Ca⁴ + CaM ⇌ CaCaM          (cooperative calcium-calmodulin binding)
        - CaCaM activates CaMKK       (Michaelis-Menten, Hill n=1)
        - CaMKK_act deactivates        (first-order)

    CaMKK (y[12]) is now the INACTIVE enzyme pool. The new CaMKK_act (y[38])
    is the active form that participates in AMPK phosphorylation.

    New states (4): Ca, CaM, CaCaM, CaMKK_act  →  35 + 4 = 39 total states
    New fixed params (5): kOnCaM, kOffCaM, kActCaMKK, KmCaM, kDeactCaMKK
"""
import jax.numpy as jnp
import equinox as eqx
import numpyro
import numpyro.distributions as dist

class MA_phos_CaMKK2(eqx.Module):
    """Right hand side of the AMPK MA nonessential phos (beta) regulation model
    with mechanistic calcium-CaMKK2 activation.

    Written in the format required by the diffrax package.
    """

    # fixed metabolism parameters
    kGly: float
    kHydro: float
    VforAK: float
    KeqAK: float
    kmm: float
    kmd: float
    kmt: float
    VmaxOxPhos: float
    Kadp: float
    n: float
    VforCK: float
    Kb: float
    Kia: float
    Kib: float
    Kiq: float
    Kp: float
    KeqCK: float
    TCr: float


    def __call__(self, t, y, args):
        """Right hand side of the AMPK MA nonessential phos (beta) + CaMKK2 model.

        Written in the format required by the diffrax package.
        """
        # unpack parameters (25 existing + 5 new calcium = 30 total)
        kOnAMP      = args[0]  # AMP binding
        kOffAMP     = args[1]
        kOnADP      = args[2]  # ADP binding
        kOffADP     = args[3]
        kOnATP      = args[4]  # ATP binding
        kOffATP     = args[5]
        kOnCaMKK    = args[6]  # CaMKK binding to AMPK substrates
        kOffCaMKK   = args[7]
        kPhosCaMKK  = args[8]  # CaMKK-mediated phosphorylation
        betaCaMKK   = args[9]  # > 1; AMP/ADP increase CaMKK phosphorylation rate
        kOnLKB1     = args[10] # LKB1 binding
        kOffLKB1    = args[11]
        betaLKB1    = args[12] # > 1; AMP/ADP increase LKB1 phosphorylation rate
        kPhosLKB1   = args[13] # phosphorylation
        kOnPP       = args[14] # Phosphatase AMPK binding
        kOffPP      = args[15]
        betaPP      = args[16] # < 1; AMP/ADP decrease PP dephosphorylation rate
        kDephosPP   = args[17] # dephosphorylation
        kOnAMPK     = args[18] # pAMPK binds AMPKAR
        kOffAMPK    = args[19]
        kPhosAMPK   = args[20] # pAMPK phosphorylates AMPKAR
        betaAMP     = args[21] # > 1 phos enhancement factor due to AMP allo act
        kOnPP1      = args[22] # Phosphatase AMPKAR binding
        kOffPP1     = args[23]
        kDephosPP1  = args[24]
        # Calcium-CaMKK2 cascade parameters (all fixed)
        kOnCaM      = args[25] # Ca⁴ + CaM forward binding rate
        kOffCaM     = args[26] # CaCaM dissociation rate
        kActCaMKK    = args[27] # CaMKK2 activation rate by CaCaM
        KmCaM       = args[28] # CaMKK activation half-max CaCaM
        kDeactCaMKK = args[29] # CaMKK deactivation rate

        # unpack states (39 total)
        AMP                 = y[0]
        ADP                 = y[1]
        ATP                 = y[2]
        PCr                 = y[3]
        AMPK                = y[4]
        pAMPK               = y[5]
        AMP_AMPK            = y[6]
        ADP_AMPK            = y[7]
        ATP_AMPK            = y[8]
        AMP_pAMPK           = y[9]
        ADP_pAMPK           = y[10]
        ATP_pAMPK           = y[11]
        CaMKK               = y[12]  # INACTIVE CaMKK pool
        CaMKK_AMPK          = y[13]  # complexes formed from CaMKK_act
        CaMKK_AMP_AMPK      = y[14]
        CaMKK_ADP_AMPK      = y[15]
        CaMKK_ATP_AMPK      = y[16]
        LKB1                = y[17]
        LKB1_AMPK           = y[18]
        LKB1_AMP_AMPK       = y[19]
        LKB1_ADP_AMPK       = y[20]
        LKB1_ATP_AMPK       = y[21]
        PP                  = y[22]
        PP_pAMPK            = y[23]
        PP_AMP_pAMPK        = y[24]
        PP_ADP_pAMPK        = y[25]
        PP_ATP_pAMPK        = y[26]
        AMPKAR              = y[27]
        pAMPKAR             = y[28]
        AMPKAR_pAMPK        = y[29]
        AMPKAR_AMP_pAMPK    = y[30]
        AMPKAR_ADP_pAMPK    = y[31]
        AMPKAR_ATP_pAMPK    = y[32]
        PP1                 = y[33]
        PP1_pAMPKAR         = y[34]
        # New calcium-CaMKK2 states
        Ca                  = y[35]
        CaM                 = y[36]
        CaCaM               = y[37]
        CaMKK_act           = y[38]  # ACTIVE CaMKK pool

        # =====================================================================
        # CALCIUM-CaMKK2 ACTIVATION CASCADE
        # =====================================================================
        # Reaction 1: 4Ca + CaM ⇌ Ca₄CaM (cooperative binding, exponent 4)
        JCa = kOnCaM*(Ca**4)*CaM - kOffCaM*CaCaM
        # Reaction 2: CaMKK activation (Michaelis-Menten on CaCaM, Hill n=1)
        JCaMKK_act = (kActCaMKK*CaCaM*CaMKK)/(KmCaM + CaCaM)
        # Reaction 3: CaMKK deactivation (first-order)
        JCaMKK_deact = kDeactCaMKK*CaMKK_act

        # =====================================================================
        # AMPK SIGNALING FLUXES (CaMKK_act replaces old CaMKK in binding)
        # =====================================================================
        J1 = kOnAMP*AMP**3*AMPK - kOffAMP*AMP_AMPK
        J2 = kOnADP*ADP*AMPK - kOffADP*ADP_AMPK
        J3 = kOnATP*ATP*AMPK - kOffATP*ATP_AMPK
        J4 = kOnAMP*AMP**3*pAMPK - kOffAMP*AMP_pAMPK
        J5 = kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK
        J6 = kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK
        # CaMKK_act binding and phosphorylation (CaMKK_act replaces old CaMKK)
        J7 = kOnCaMKK*CaMKK_act*AMPK - kOffCaMKK*CaMKK_AMPK
        J8 = kPhosCaMKK*CaMKK_AMPK
        J9 = kOnCaMKK*CaMKK_act*AMP_AMPK - kOffCaMKK*CaMKK_AMP_AMPK
        J10 = kOnAMP*AMP**3*CaMKK_AMPK - kOffAMP*CaMKK_AMP_AMPK
        J11 = betaCaMKK*kPhosCaMKK*CaMKK_AMP_AMPK                          # beta on catalysis
        J12 = kOnCaMKK*CaMKK_act*ADP_AMPK - kOffCaMKK*CaMKK_ADP_AMPK
        J13 = kOnADP*ADP*CaMKK_AMPK - kOffADP*CaMKK_ADP_AMPK
        J14 = betaCaMKK*kPhosCaMKK*CaMKK_ADP_AMPK                          # beta on catalysis
        J15 = kOnCaMKK*CaMKK_act*ATP_AMPK - kOffCaMKK*CaMKK_ATP_AMPK
        J16 = kOnATP*ATP*CaMKK_AMPK - kOffATP*CaMKK_ATP_AMPK
        J17 = kPhosCaMKK*CaMKK_ATP_AMPK
        # LKB1 binding and phosphorylation
        J18 = kOnLKB1*LKB1*AMPK - kOffLKB1*LKB1_AMPK
        J19 = kPhosLKB1*LKB1_AMPK
        J20 = kOnLKB1*LKB1*AMP_AMPK - kOffLKB1*LKB1_AMP_AMPK
        J21 = kOnAMP*AMP**3*LKB1_AMPK - kOffAMP*LKB1_AMP_AMPK
        J22 = betaLKB1*kPhosLKB1*LKB1_AMP_AMPK                             # beta on catalysis
        J23 = kOnLKB1*LKB1*ADP_AMPK - kOffLKB1*LKB1_ADP_AMPK
        J24 = kOnADP*ADP*LKB1_AMPK - kOffADP*LKB1_ADP_AMPK
        J25 = betaLKB1*kPhosLKB1*LKB1_ADP_AMPK                             # beta on catalysis
        # PP binding and dephosphorylation
        J26 = kOnPP*PP*pAMPK - kOffPP*PP_pAMPK
        J27 = kDephosPP*PP_pAMPK
        J28 = kOnPP*PP*AMP_pAMPK - kOffPP*PP_AMP_pAMPK
        J29 = kOnAMP*AMP**3*PP_pAMPK - kOffAMP*PP_AMP_pAMPK
        J30 = betaPP*kDephosPP*PP_AMP_pAMPK                                 # beta on catalysis
        J31 = kOnPP*PP*ADP_pAMPK - kOffPP*PP_ADP_pAMPK
        J32 = kOnADP*ADP*PP_pAMPK - kOffADP*PP_ADP_pAMPK
        J33 = betaPP*kDephosPP*PP_ADP_pAMPK                                 # beta on catalysis
        J34 = kOnPP*PP*ATP_pAMPK - kOffPP*PP_ATP_pAMPK
        J35 = kOnATP*ATP*PP_pAMPK - kOffATP*PP_ATP_pAMPK
        J36 = kDephosPP*PP_ATP_pAMPK
        # AMPKAR phosphorylation by pAMPK
        J37 = kOnAMPK*AMPKAR*pAMPK - kOffAMPK*AMPKAR_pAMPK
        J38 = kPhosAMPK*AMPKAR_pAMPK
        J39 = kOnAMPK*AMPKAR*AMP_pAMPK - kOffAMPK*AMPKAR_AMP_pAMPK
        J40 = kOnAMP*AMP**3*AMPKAR_pAMPK - kOffAMP*AMPKAR_AMP_pAMPK
        J41 = betaAMP*kPhosAMPK*AMPKAR_AMP_pAMPK
        J42 = kOnAMPK*AMPKAR*ADP_pAMPK - kOffAMPK*AMPKAR_ADP_pAMPK
        J43 = kOnADP*ADP*AMPKAR_pAMPK - kOffADP*AMPKAR_ADP_pAMPK
        J44 = kPhosAMPK*AMPKAR_ADP_pAMPK
        # PP1 dephosphorylation of pAMPKAR
        J45 = kOnPP1*pAMPKAR*PP1 - kOffPP1*PP1_pAMPKAR
        J46 = kDephosPP1*PP1_pAMPKAR
        # LKB1 phosphorylation of ATP-AMPK (no beta — ATP does not enhance catalysis)
        J47 = kOnLKB1*LKB1*ATP_AMPK - kOffLKB1*LKB1_ATP_AMPK
        J48 = kOnATP*ATP*LKB1_AMPK - kOffATP*LKB1_ATP_AMPK
        J49 = kPhosLKB1*LKB1_ATP_AMPK
        # AMPKAR phosphorylation by ATP-pAMPK (no beta — ATP does not enhance activity)
        J50 = kOnAMPK*AMPKAR*ATP_pAMPK - kOffAMPK*AMPKAR_ATP_pAMPK
        J51 = kOnATP*ATP*AMPKAR_pAMPK - kOffATP*AMPKAR_ATP_pAMPK
        J52 = kPhosAMPK*AMPKAR_ATP_pAMPK

        # =====================================================================
        # METABOLIC FLUXES
        # =====================================================================
        Jgly = self.kGly*ADP
        Jhydro = self.kHydro*ATP
        # Adenylate Kinase
        num_for = (self.VforAK*ATP*AMP)/(self.kmt*self.kmm)
        den_ak = (1 + (ATP/self.kmt) + (AMP/self.kmm) + ((ATP*AMP)/(self.kmt*self.kmm)) +
                    ((2*ADP)/self.kmd) + ((ADP**2)/(self.kmd**2)))
        VrevAK = (self.VforAK*(self.kmd**2))/(self.KeqAK*self.kmt*self.kmm)
        num_rev = (VrevAK*(ADP**2))/(self.kmd**2)
        JAK = (num_for - num_rev)/den_ak
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((ADP/self.Kadp)**self.n))/(1 + ((ADP/self.Kadp)**self.n))
        # Creatine kinase
        den_ck = 1 + (ADP/self.Kia) + (PCr/self.Kib) + (ATP/self.Kiq) + ((ADP*PCr)/(self.Kia*self.Kb)) + (((self.TCr - PCr)*ATP)/(self.Kiq*self.Kp))
        num_forCK = ((self.VforCK*ADP*PCr)/(self.Kia*self.Kb))
        VrevCK = (self.VforCK*self.Kiq*self.Kp)/(self.KeqCK*self.Kia*self.Kb)
        num_revCK = ((VrevCK*ATP*(self.TCr - PCr))/(self.Kiq*self.Kp))
        JCK = (num_revCK - num_forCK)/den_ck

        # =====================================================================
        # ODEs
        # =====================================================================
        # Nucleotides
        d_AMP = -3*J1-3*J4-3*J10-3*J21-3*J29-3*J40-JAK
        d_ADP = -J2-J5-J13-J24-J32-J43-Jgly+2*JAK+Jhydro-Joxphos+JCK
        d_ATP = -J3-J6-J16-J35-J48-J51+Jgly-JAK-Jhydro+Joxphos-JCK
        d_PCr = JCK
        # AMPK
        d_AMPK = -J1-J2-J3-J7-J18+J27
        d_pAMPK = -J4-J5-J6+J8+J19-J26-J37+J38
        d_AMP_AMPK = J1-J9-J20-J28+J30
        d_ADP_AMPK = J2-J12-J23+J33
        d_ATP_AMPK = J3-J15-J47+J36
        d_AMP_pAMPK = J4+J11+J22-J39+J41
        d_ADP_pAMPK = J5+J14+J25-J31-J42+J44
        d_ATP_pAMPK = J6+J17-J34+J49-J50+J52
        # CaMKK — now split into inactive and active pools
        d_CaMKK = -JCaMKK_act + JCaMKK_deact                               # inactive pool
        d_CaMKK_AMPK = J7-J8-J10-J13-J16
        d_CaMKK_AMP_AMPK = J9+J10-J11
        d_CaMKK_ADP_AMPK = J12+J13-J14
        d_CaMKK_ATP_AMPK = J15+J16-J17
        # LKB1
        d_LKB1 = -J18+J19-J20+J22-J23+J25-J47+J49
        d_LKB1_AMPK = J18-J19-J21-J24-J48
        d_LKB1_AMP_AMPK = J20+J21-J22
        d_LKB1_ADP_AMPK = J23+J24-J25
        d_LKB1_ATP_AMPK = J47+J48-J49
        # PP
        d_PP = -J26+J27-J28+J30-J31+J33-J34+J36
        d_PP_pAMPK = J26-J27-J29-J32-J35
        d_PP_AMP_pAMPK = J28+J29-J30
        d_PP_ADP_pAMPK = J31+J32-J33
        d_PP_ATP_pAMPK = J34+J35-J36
        # AMPKAR
        d_AMPKAR = -J37-J39-J42-J50+J46
        d_pAMPKAR = J38+J41+J44+J52-J45
        d_AMPKAR_pAMPK = J37-J38-J40-J43-J51
        d_AMPKAR_AMP_pAMPK = J39+J40-J41
        d_AMPKAR_ADP_pAMPK = J42+J43-J44
        d_AMPKAR_ATP_pAMPK = J50+J51-J52
        # PP1
        d_PP1 = -J45+J46
        d_PP1_pAMPKAR = J45-J46
        # Calcium-CaMKK2 cascade
        d_Ca = -4*JCa                                                        # 4:1 stoichiometry
        d_CaM = -JCa
        d_CaCaM = JCa
        d_CaMKK_act = JCaMKK_act - JCaMKK_deact - J7+J8-J9+J11-J12+J14-J15+J17  # active pool

        return [d_AMP, d_ADP, d_ATP, d_PCr,
                d_AMPK, d_pAMPK, d_AMP_AMPK, d_ADP_AMPK, d_ATP_AMPK,
                d_AMP_pAMPK, d_ADP_pAMPK, d_ATP_pAMPK,
                d_CaMKK, d_CaMKK_AMPK, d_CaMKK_AMP_AMPK, d_CaMKK_ADP_AMPK, d_CaMKK_ATP_AMPK,
                d_LKB1, d_LKB1_AMPK, d_LKB1_AMP_AMPK, d_LKB1_ADP_AMPK, d_LKB1_ATP_AMPK,
                d_PP, d_PP_pAMPK, d_PP_AMP_pAMPK, d_PP_ADP_pAMPK, d_PP_ATP_pAMPK,
                d_AMPKAR, d_pAMPKAR, d_AMPKAR_pAMPK,
                d_AMPKAR_AMP_pAMPK, d_AMPKAR_ADP_pAMPK, d_AMPKAR_ATP_pAMPK,
                d_PP1, d_PP1_pAMPKAR,
                d_Ca, d_CaM, d_CaCaM, d_CaMKK_act]


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly
