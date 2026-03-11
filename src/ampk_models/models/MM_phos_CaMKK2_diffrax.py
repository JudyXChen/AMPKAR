"""
    Model 6+CaMKK2 — MM nonessential with beta parameterization + calcium-CaMKK2 cascade.

    Extends MM_nonessential_phos_diffrax.py (Model 6) with a mechanistic calcium-
    CaMKK2 activation module:
        - Ca⁴ + CaM ⇌ CaCaM          (cooperative calcium-calmodulin binding)
        - CaCaM activates CaMKK       (Michaelis-Menten, Hill n=1)
        - CaMKK_act deactivates        (first-order)

    CaMKKtot is no longer a constant parameter. Instead, CaMKK (y[14], inactive)
    and CaMKK_act (y[15], active) are ODE states. CaMKK_act replaces CaMKKtot
    in all Michaelis-Menten rate expressions.

    New states (5): Ca, CaM, CaCaM, CaMKK, CaMKK_act  →  14 + 5 = 19 total states
    New fixed params (5): kOnCaM, kOffCaM, kActCaMKK, KmCaM, kDeactCaMKK
    Removed param (1): CaMKKtot
"""
import jax.numpy as jnp
import equinox as eqx
import numpyro
import numpyro.distributions as dist

class MM_phos_CaMKK2(eqx.Module):
    """Right hand side of the AMPK MM nonessential phos (beta) regulation model
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
        """Right hand side of the AMPK MM nonessential phos (beta) + CaMKK2 model.

        Written in the format required by the diffrax package.
        """

        # unpack parameters (23 existing + 5 new calcium = 28 total)
        kOnAMP      = args[0]  # AMP binding
        kOffAMP     = args[1]
        kOnADP      = args[2]  # ADP binding
        kOffADP     = args[3]
        kOnATP      = args[4]  # ATP binding
        kOffATP     = args[5]
        kPhosCaMKK  = args[6]  # CaMKK MM phosphorylation rate (kcat)
        KmCaMKK     = args[7]
        betaCaMKK   = args[8]  # > 1; AMP/ADP increase CaMKK phosphorylation rate
        kPhosLKB1   = args[9]  # LKB1
        KmLKB1      = args[10]
        betaLKB1    = args[11] # > 1; AMP/ADP increase LKB1 phosphorylation rate
        kDephosPP   = args[12] # AMPK Phosphatase
        KmPP        = args[13]
        betaPP      = args[14] # < 1; AMP/ADP decrease PP dephosphorylation rate
        kPhosAMPK   = args[15] # AMPK kinase
        KmAMPK      = args[16]
        betaAMP     = args[17] # > 1 phos enhancement factor due to AMP allo act
        kDephosPP1  = args[18] # pAMPKAR Phosphatase
        KmPP1       = args[19]
        # external enzyme concentrations (CaMKKtot removed — now dynamic)
        LKB1tot     = args[20]
        PPtot       = args[21]
        PP1tot      = args[22]
        # Calcium-CaMKK2 cascade parameters (all fixed)
        kOnCaM      = args[23] # Ca⁴ + CaM forward binding rate
        kOffCaM     = args[24] # CaCaM dissociation rate
        kActCaMKK   = args[25] # CaMKK2 activation rate by CaCaM
        KmCaM       = args[26] # CaMKK activation half-max CaCaM
        kDeactCaMKK = args[27] # CaMKK deactivation rate

        # unpack states (19 total)
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
        AMPKAR              = y[12]
        pAMPKAR             = y[13]
        # New calcium-CaMKK2 states
        CaMKK               = y[14]  # INACTIVE CaMKK pool
        CaMKK_act           = y[15]  # ACTIVE CaMKK pool
        Ca                  = y[16]
        CaM                 = y[17]
        CaCaM               = y[18]

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
        # AMPK SIGNALING FLUXES (CaMKK_act replaces CaMKKtot)
        # =====================================================================
        J1 = (kOnAMP*AMP**3*AMPK-kOffAMP*AMP_AMPK)
        J2 = (kOnADP*ADP*AMPK-kOffADP*ADP_AMPK)
        J3 = (kOnATP*ATP*AMPK-kOffATP*ATP_AMPK)
        J4 = (kOnAMP*AMP**3*pAMPK-kOffAMP*AMP_pAMPK)
        J5 = (kOnADP*ADP*pAMPK-kOffADP*ADP_pAMPK)
        J6 = (kOnATP*ATP*pAMPK-kOffATP*ATP_pAMPK)
        # CaMKK phosphorylation — CaMKK_act replaces CaMKKtot
        J7 = ((kPhosCaMKK*CaMKK_act*AMPK)/(KmCaMKK + AMPK))
        J8 = ((betaCaMKK*kPhosCaMKK*CaMKK_act*AMP_AMPK)/(KmCaMKK + AMP_AMPK))
        J9 = ((betaCaMKK*kPhosCaMKK*CaMKK_act*ADP_AMPK)/(KmCaMKK + ADP_AMPK))
        J10 = ((kPhosCaMKK*CaMKK_act*ATP_AMPK)/(KmCaMKK + ATP_AMPK))
        # LKB1 phosphorylation — beta on numerator
        J11 = ((kPhosLKB1*LKB1tot*AMPK)/(KmLKB1 + AMPK))
        J12 = ((betaLKB1*kPhosLKB1*LKB1tot*AMP_AMPK)/(KmLKB1 + AMP_AMPK))
        J13 = ((betaLKB1*kPhosLKB1*LKB1tot*ADP_AMPK)/(KmLKB1 + ADP_AMPK))
        # PP dephos — beta on numerator
        J14 = ((kDephosPP*PPtot*pAMPK)/(KmPP + pAMPK))
        J15 = ((betaPP*kDephosPP*PPtot*AMP_pAMPK)/(KmPP + AMP_pAMPK))
        J16 = ((betaPP*kDephosPP*PPtot*ADP_pAMPK)/(KmPP + ADP_pAMPK))
        J17 = ((kDephosPP*PPtot*ATP_pAMPK)/(KmPP + ATP_pAMPK))
        # AMPKAR phos
        J18 = (kPhosAMPK*pAMPK*AMPKAR)/(KmAMPK + AMPKAR)
        J19 = (betaAMP*kPhosAMPK*AMP_pAMPK*AMPKAR)/(KmAMPK + AMPKAR)
        J20 = (kPhosAMPK*ADP_pAMPK*AMPKAR)/(KmAMPK + AMPKAR)
        # LKB1 phosphorylation of ATP-AMPK (no beta — ATP does not enhance catalysis)
        J22 = (kPhosLKB1*LKB1tot*ATP_AMPK)/(KmLKB1 + ATP_AMPK)
        # AMPKAR phosphorylation by ATP-pAMPK (no beta — ATP does not enhance activity)
        J23 = (kPhosAMPK*ATP_pAMPK*AMPKAR)/(KmAMPK + AMPKAR)
        # PP1 dephos
        J21 = (kDephosPP1*PP1tot*pAMPKAR)/(KmPP1 + pAMPKAR)

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
        d_AMP = -JAK-3*J1-3*J4
        d_ADP = -Jgly+2*JAK+Jhydro-Joxphos + JCK-J2-J5
        d_ATP = Jgly-JAK-Jhydro+Joxphos-JCK-J3-J6
        d_PCr = JCK
        # AMPK
        d_AMPK = -J1-J2-J3-J7-J11+J14
        d_pAMPK = -J4-J5-J6+J7+J11-J14
        d_AMP_AMPK = J1-J8-J12+J15
        d_ADP_AMPK = J2-J9-J13+J16
        d_ATP_AMPK = J3-J10-J22+J17
        d_AMP_pAMPK = J4+J8+J12-J15
        d_ADP_pAMPK = J5+J9+J13-J16
        d_ATP_pAMPK = J6+J10+J22-J17
        # AMPKAR
        d_AMPKAR = -J18-J19-J20-J23+J21
        d_pAMPKAR = J18+J19+J20+J23-J21
        # Calcium-CaMKK2 cascade
        d_CaMKK = -JCaMKK_act + JCaMKK_deact                               # inactive pool
        d_CaMKK_act = JCaMKK_act - JCaMKK_deact                            # active pool (no binding fluxes in MM)
        d_Ca = -4*JCa                                                        # 4:1 stoichiometry
        d_CaM = -JCa
        d_CaCaM = JCa

        return [d_AMP, d_ADP, d_ATP, d_PCr, d_AMPK, d_pAMPK, d_AMP_AMPK,
                d_ADP_AMPK, d_ATP_AMPK, d_AMP_pAMPK, d_ADP_pAMPK, d_ATP_pAMPK,
                d_AMPKAR, d_pAMPKAR,
                d_CaMKK, d_CaMKK_act, d_Ca, d_CaM, d_CaCaM]


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly
