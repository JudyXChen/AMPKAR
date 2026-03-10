"""
    Model 6 — MM nonessential with beta parameterization on catalytic rates (Vmax/kcat).

    AMP/ADP modulate phosphorylation/dephosphorylation rates (betaLKB1 > 1,
    betaCaMKK > 1, betaPP < 1) instead of Michaelis constants (alphaLKB1,
    alphaPP in Model 4).

    Based on MM_nonessential_diffrax.py (Model 4) with beta modifications
    mirroring MA_nonessential_phos_diffrax.py (Model 5).
"""
import jax.numpy as jnp
import equinox as eqx
import numpyro
import numpyro.distributions as dist

class MM_nonessential_phos(eqx.Module):
    """Right hand side of the AMPK MM nonessential phos (beta) regulation model.

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
        """Right hand side of the AMPK MM nonessential phos (beta) regulation model.

        Written in the format required by the diffrax package.
        """

        # unpack parameters (24 total)
        kOnAMP      = args[0]  # AMP binding
        kOffAMP     = args[1]
        kOnADP      = args[2]  # ADP binding
        kOffADP     = args[3]
        kOnATP      = args[4]  # ATP binding
        kOffATP     = args[5]
        kPhosCaMKK  = args[6]  # CaMKK
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
        # external enzyme concentrations
        CaMKKtot    = args[20]
        LKB1tot     = args[21]
        PPtot       = args[22]
        PP1tot      = args[23]

        # unpack states
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

        # FLUXES
        J1 = (kOnAMP*AMP**3*AMPK-kOffAMP*AMP_AMPK)
        J2 = (kOnADP*ADP*AMPK-kOffADP*ADP_AMPK)
        J3 = (kOnATP*ATP*AMPK-kOffATP*ATP_AMPK)
        J4 = (kOnAMP*AMP**3*pAMPK-kOffAMP*AMP_pAMPK)
        J5 = (kOnADP*ADP*pAMPK-kOffADP*ADP_pAMPK)
        J6 = (kOnATP*ATP*pAMPK-kOffATP*ATP_pAMPK)
        # CaMKK phosphorylation — unbound: no beta; AMP/ADP-bound: beta on numerator
        J7 = ((kPhosCaMKK*CaMKKtot*AMPK)/(KmCaMKK + AMPK))
        J8 = ((betaCaMKK*kPhosCaMKK*CaMKKtot*AMP_AMPK)/(KmCaMKK + AMP_AMPK))
        J9 = ((betaCaMKK*kPhosCaMKK*CaMKKtot*ADP_AMPK)/(KmCaMKK + ADP_AMPK))
        J10 = ((kPhosCaMKK*CaMKKtot*ATP_AMPK)/(KmCaMKK + ATP_AMPK))
        # LKB1 phosphorylation — beta on numerator (replaces alpha on Km denominator)
        J11 = ((kPhosLKB1*LKB1tot*AMPK)/(KmLKB1 + AMPK))
        J12 = ((betaLKB1*kPhosLKB1*LKB1tot*AMP_AMPK)/(KmLKB1 + AMP_AMPK))
        J13 = ((betaLKB1*kPhosLKB1*LKB1tot*ADP_AMPK)/(KmLKB1 + ADP_AMPK))
        # PP dephos — beta on numerator (replaces alpha on Km denominator)
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

        # Metabolic fluxes
        # glycolysis
        Jgly = self.kGly*ADP #2*kGly*ADP*ADP
        # ATP hydrolysis
        Jhydro = self.kHydro*ATP
        # Adenylate Kinase
        # written as (VforAK*ATP)/(kmt*kmm) in cocci, but units dont make sense
        num_for = (self.VforAK*ATP*AMP)/(self.kmt*self.kmm)
        den_ak = (1 + (ATP/self.kmt) + (AMP/self.kmm) + ((ATP*AMP)/(self.kmt*self.kmm)) +
                    ((2*ADP)/self.kmd) + ((ADP**2)/(self.kmd**2)))
        VrevAK = (self.VforAK*(self.kmd**2))/(self.KeqAK*self.kmt*self.kmm)
        num_rev = (VrevAK*(ADP**2))/(self.kmd**2)
        JAK = (num_for - num_rev)/den_ak # ADP forming direction
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((ADP/self.Kadp)**self.n))/(1 + ((ADP/self.Kadp)**self.n))
        # Creatine kinase
        den_ck = 1 + (ADP/self.Kia) + (PCr/self.Kib) + (ATP/self.Kiq) + ((ADP*PCr)/(self.Kia*self.Kb)) + (((self.TCr - PCr)*ATP)/(self.Kiq*self.Kp))
        num_forCK = ((self.VforCK*ADP*PCr)/(self.Kia*self.Kb))
        VrevCK = (self.VforCK*self.Kiq*self.Kp)/(self.KeqCK*self.Kia*self.Kb)
        num_revCK = ((VrevCK*ATP*(self.TCr - PCr))/(self.Kiq*self.Kp))
        JCK = (num_revCK - num_forCK)/den_ck # Pi forming direction

        # now return the odes for each state variable
        d_AMP = -JAK-3*J1-3*J4
        d_ADP = -Jgly+2*JAK+Jhydro-Joxphos + JCK-J2-J5
        d_ATP = Jgly-JAK-Jhydro+Joxphos-JCK-J3-J6
        d_PCr = JCK
        d_AMPK = -J1-J2-J3-J7-J11+J14
        d_pAMPK = -J4-J5-J6+J7+J11-J14
        d_AMP_AMPK = J1-J8-J12+J15
        d_ADP_AMPK = J2-J9-J13+J16
        d_ATP_AMPK = J3-J10-J22+J17
        d_AMP_pAMPK = J4+J8+J12-J15
        d_ADP_pAMPK = J5+J9+J13-J16
        d_ATP_pAMPK = J6+J10+J22-J17
        d_AMPKAR = -J18-J19-J20-J23+J21
        d_pAMPKAR = J18+J19+J20+J23-J21

        return [d_AMP, d_ADP, d_ATP, d_PCr, d_AMPK, d_pAMPK, d_AMP_AMPK,
                d_ADP_AMPK, d_ATP_AMPK, d_AMP_pAMPK, d_ADP_pAMPK, d_ATP_pAMPK,
                d_AMPKAR, d_pAMPKAR]


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly
