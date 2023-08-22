TITLE RS membrane mechanism

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX RS

    : Regular ionic currents
    NONSPECIFIC_CURRENT iNa : Sodium current
    NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
    NONSPECIFIC_CURRENT iLeak : non-specific leakage current
    : NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
    NONSPECIFIC_CURRENT iNaKPump : Sodium-potassium pump current

    : Artificial drive current
    NONSPECIFIC_CURRENT idrive

    : Python-accessible parameters
    RANGE I, gamma, ibaseline 
    RANGE Tref, alphaT, tauT_abs, tauT_diss
    RANGE Q10_rates, Q10_gNa, Q10_gKd, Q10_gNaK
    RANGE gNaKPump_ref

    : Python-accessible internal variables
    RANGE gLeak, gNabar, gKdbar, gNaKPump
}

PARAMETER {
    : Regular RS model parameters 
    ENa = 50.0 (mV) : Sodium reversal potential
    EK = -90.0 (mV) : Potassium reversal potential
    ELeak = -70.3 (mV) : Leak reversal potential
    EPump = -220 (mV) : Sodium-potassium pump reversal potential
    VT = -56.2 (mV) : Spike threshold adjustment parameter
    gNabar_ref = 0.056 (S/cm2) : Maximal conductance of iNa at 36 deg. C
    gKdbar_ref = 0.006 (S/cm2) : Maximal conductance of iKd at 36 deg. C
    gLeak_ref = 2.05e-05 (S/cm2) : Leak conductance (at 36 deg. C)
    gNaKPump_ref = 3e-6 :45.6e-6 (S/cm2) : Sodium-potassium pump maximal conductance (at 36 deg. C)
    :gMbar = 7.5e-05 (S/cm2) : Maximal conductance of iM (at 36 deg. C)
    :TauMax = 608 (ms) : Max. adaptation decay of slow non-inactivating Potassium current (at 36 deg. C)

    : Thermal parameters
    Tref = 36  : reference temperature (in deg. C)
    alphaT = .02 : max temperature increase (in deg. C) per stimulus intensity unit
    tauT_abs = 100 : heat absorption time constant (ms)
    tauT_diss = 100  : heat dissipation time constant (ms)
    Q10_rates = 3  : Q10 coefficient for temperature dependence of gating transitions
    Q10_gNa = 1.40  : Q10 coefficient for temperature dependence of iNa maximal conductance
    Q10_gKd = 4.75  : Q10 coefficient for temperature dependence of iKd maximal conductance
    Q10_gNaK = 1.88  : Q10 coefficient for temperature dependence of iNaKPump maximal conductance 

    : Baseline and Stimulus drive parameters
    ibaseline = 0 (mA/cm2) : baseline current (e.g. thalamic drive)
    gamma = 1e-5 : depolarizing force (mA/cm2) per stimulus intensity unit 
    I = 0  : time-varying stimulus intensity (t.b.d.)
}

STATE {
    : Regular ion channel gating states
    m : iNa activation gate
    h : iNa inactivation gate
    n : iKd gate
    :p : iM gate

    : Additional states
    T : temperature (celsius)
}

ASSIGNED {
    : Membrane potential
    v  (nC/cm2)
    
    : Membrane ionic currents
    iNa (mA/cm2)
    iKd (mA/cm2)
    iLeak (mA/cm2)
    iNaKPump (mA/cm2)
    :iM (mA/cm2)

    : Artificial drive current
    idrive (mA/cm2)

    : Ion channel conductances
    gNabar (S/cm2)
    gKdbar (S/cm2)
    gLeak (S/cm2)
    gNaKPump (S/cm2)
}

: VOLTAGE-DEPENDENT RATE CONSTANT FUNCTIONS TO COMPUTE GATING TRANSITIONS

FUNCTION vtrap(x, y) {
    : Generic bilinear function to compute voltage-dependent rate constants
    vtrap = x / (exp(x / y) - 1)
}

FUNCTION alpham(v) {
    : Sodium m-gate activation rate constant
    alpham = 0.32 * vtrap(13 - (v - VT), 4)  : ms-1
}

FUNCTION betam(v) {
    : Sodium m-gate inactivation rate constant
    betam = 0.28 * vtrap((v - VT) - 40, 5) : ms-1
}

FUNCTION alphah(v) {
    : Sodium h-gate activation rate constant
    alphah = 0.128 * exp(-((v - VT) - 17) / 18) : ms-1
}

FUNCTION betah(v) {
    : Sodium h-gate inactivation rate constant
    betah = 4 / (1 + exp(-((v - VT) - 40) / 5)) :  ms-1
}

FUNCTION alphan(v) {
    : Potassium n-gate activation rate constant
    alphan = 0.032 * vtrap(15 - (v - VT), 5) :  ms-1
}

FUNCTION betan(v) {
    : Potassium n-gate inactivation rate constant
    betan = 0.5 * exp(-((v - VT) - 10) / 40) :  ms-1
}

COMMENT
FUNCTION pinf(v) {
    : Slow non-inactivating Potassium current steady-state activation probability
    pinf = 1.0 / (1 + exp(-(v + 35) / 10))  : (-)
}

FUNCTION taup(v) {
    : Slow non-inactivating Potassium current activation time constant
    taup = TauMax / (3.3 * exp((v + 35) / 20) + exp(-(v + 35) / 20))  : ms
}
ENDCOMMENT

: TEMPERATURE EVOLUTION FUNCTIONS

FUNCTION Tinf(I) {
    Tinf = alphaT * I + Tref
}

FUNCTION tauT(I) {
    if (I > 0) {
        tauT = tauT_abs
    }
    else {
        tauT = tauT_diss
    }
}

: TEMPERATURE-DEPENDENT FUNCTIONS

FUNCTION phi(T, Q) {
    : Temperature-dependent rate constant scaling factor
    phi = Q^((T - Tref) / 10)
} 

INITIAL {
    : Initial ion channel gating states
    m = alpham(v) / (alpham(v) + betam(v))
    h = alphah(v) / (alphah(v) + betah(v))
    n = alphan(v) / (alphan(v) + betan(v))
    : p = pinf(v)

    : Initial temperature
    T = Tinf(I)
}

BREAKPOINT {
    : States integration
    SOLVE states METHOD cnexp

    : Update conductances based on temperature
    gLeak = gLeak_ref
    gNabar = gNabar_ref * phi(T, Q10_gNa)
    gKdbar = gKdbar_ref * phi(T, Q10_gKd)
    gNaKPump = gNaKPump_ref * phi(T, Q10_gNaK)

    : Membrane currents computation
    idrive = -(gamma * I + ibaseline)  : drive current = stimulus-driven current + baseline current
    iNa = gNabar * m * m * m * h * (v - ENa)
    iKd = gKdbar * n * n * n * n * (v - EK)
    iLeak = gLeak * (v - ELeak)
    iNaKPump = gNaKPump * (v - EPump)
    : iM = gMbar * p * (v - EK)
}

DERIVATIVE states {
    : Ion channels gating states derivatives (voltage- and temperature-dependent)
    m' = (alpham(v) * (1 - m) - betam(v) * m) * phi(T, Q10_rates)
    h' = (alphah(v) * (1 - h) - betah(v) * h) * phi(T, Q10_rates)
    n' = (alphan(v) * (1 - n) - betan(v) * n) * phi(T, Q10_rates)
    : p' = (pinf(v) - p) / taup(v) * phi(T, Q10_rates)

    : Temperature derivative
    T' = (Tinf(I) - T) / tauT(I)
}
