TITLE RS membrane mechanism

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX RS

    : Regular ionic currents
    NONSPECIFIC_CURRENT iNa : Sodium current
    NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
    NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
    NONSPECIFIC_CURRENT iLeak : non-specific leakage current

    : Artificial stimulus drive current
    NONSPECIFIC_CURRENT idrive

    : Python-accessible parameters
    RANGE I, gamma, Q10, Tref, alphaT, tauT_abs, tauT_diss

    : Python-accessible internal variables
    RANGE gLeak
}

PARAMETER {
    : Regular RS model parameters 
    ENa = 50.0 (mV) : Sodium reversal potential
    EK = -90.0 (mV) : Potassium reversal potential
    ELeak = -70.3 (mV) : Leak reversal potential
    gNabar = 0.056 (S/cm2) : Maximal conductance of iNa
    gKdbar = 0.006 (S/cm2) : Maximal conductance of iKd
    gMbar = 0.:7.5e-05 (S/cm2) : Maximal conductance of iM
    gLeak_ref = 2.05e-05 (S/cm2) : Reference leak conductance
    VT = -56.2 (mV) : Spike threshold adjustment parameter
    TauMax = 608 (ms) : Max. adaptation decay of slow non-inactivating Potassium current

    : Additional parameters
    Tref = 37  : reference temperature (celsius)
    alphaT = .017 : max temperature increase (in celsius) per stimulus intensity unit
    tauT_abs = 100 : heat absorption time constant (ms)
    tauT_diss = 100  : heat dissipation time constant (ms)
    Q10 = 2  : Q10 coefficient for temperature dependence of leak conductance
    gamma = 1e-5 : depolarizing force (mA/cm2) per stimulus intensity unit 
    I = 0  : time-varying stimulus intensity (t.b.d.)
}

STATE {
    : Regular ion channel gating states
    m : iNa activation gate
    h : iNa inactivation gate
    n : iKd gate
    p : iM gate

    : Additional states
    T : temperature (celsius)
}

ASSIGNED {
    v  (nC/cm2)
    iNa (mA/cm2)
    iKd (mA/cm2)
    iM (mA/cm2)
    iLeak (mA/cm2)
    idrive (mA/cm2)
    gLeak (S/cm2)
}

: VOLTAGE-DEPENDENT RATE CONSTANT FUNCTIONS TO COMPUTE GATING TRANSITIONS

FUNCTION vtrap(x, y) {
    vtrap = x / (exp(x / y) - 1)
}

FUNCTION alpham(v) {
    alpham = 0.32 * vtrap(13 - (v - VT), 4)  : ms-1
}

FUNCTION betam(v) {
    betam = 0.28 * vtrap((v - VT) - 40, 5) : ms-1
}

FUNCTION alphah(v) {
    alphah = 0.128 * exp(-((v - VT) - 17) / 18) : ms-1
}

FUNCTION betah(v) {
    betah = 4 / (1 + exp(-((v - VT) - 40) / 5)) :  ms-1
}

FUNCTION alphan(v) {
    alphan = 0.032 * vtrap(15 - (v - VT), 5) :  ms-1
}

FUNCTION betan(v) {
    betan = 0.5 * exp(-((v - VT) - 10) / 40) :  ms-1
}

FUNCTION pinf(v) {
    pinf = 1.0 / (1 + exp(-(v + 35) / 10))  : (-)
}

FUNCTION taup(v) {
    taup = TauMax / (3.3 * exp((v + 35) / 20) + exp(-(v + 35) / 20))  : ms
}

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

INITIAL {
    : Initial ion channel gating states
    m = alpham(v) / (alpham(v) + betam(v))
    h = alphah(v) / (alphah(v) + betah(v))
    n = alphan(v) / (alphan(v) + betan(v))
    p = pinf(v)
    T = Tinf(I)
}

BREAKPOINT {
    : States integration
    SOLVE states METHOD cnexp

    : Update leakage conductance based on temperature
    gLeak = gLeak_ref * Q10^((T - Tref) / 10)

    : Membrane currents computation
    iNa = gNabar * m * m * m * h * (v - ENa)
    iKd = gKdbar * n * n * n * n * (v - EK)
    iM = gMbar * p * (v - EK)
    iLeak = gLeak * (v - ELeak)
    idrive = -gamma * I
}

DERIVATIVE states {
    : Ion channels gating states derivatives
    m' = alpham(v) * (1 - m) - betam(v) * m
    h' = alphah(v) * (1 - h) - betah(v) * h
    n' = alphan(v) * (1 - n) - betan(v) * n
    p' = (pinf(v) - p) / taup(v)

    : Additional state derivatives 
    T' = (Tinf(I) - T) / tauT(I)
}