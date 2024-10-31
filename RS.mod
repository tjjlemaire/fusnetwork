TITLE RS membrane mechanism

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    : Membrane mechanism name
    SUFFIX RS

    : Transmembrane currents
    NONSPECIFIC_CURRENT iNa : Sodium current
    NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
    NONSPECIFIC_CURRENT iLeak : non-specific leakage current
    NONSPECIFIC_CURRENT iM : slow non-inactivating Potassium current
    NONSPECIFIC_CURRENT iStim : Stimulus-driven depolarizing current
    NONSPECIFIC_CURRENT iKT : Thermally-driven Potassium current

    : Python-accessible parameters/variables
    RANGE Z : physical constants
    RANGE Pamp : stimulus parameters
    RANGE a, b :  stimulus-driven current parameters
    RANGE Tref, alphaT, tauT_abs, tauT_diss  : thermal model parameters
    RANGE Q10_rates, Q10_gNa, Q10_gKd, Q10_gNaK  : temperature dependence parameters
    RANGE gLeak, gNabar, gKdbar, gMbar : ion channel reference maximal conductance parameters
    RANGE gLeak_t, gNabar_t, gKdbar_t, gMbar_t : ion channel maximal conductance variables
    RANGE EKT, gKT, gKT_t : KT parameters and variables
}

PARAMETER {
    : Physical constants
    Z = 1.62  : acoustic impedance of brain tissue (MRayl = 1e6*kg/(m^2*s))

    : Spiking model parameters 
    ENa = 50.0 (mV) : Sodium reversal potential
    EK = -90.0 (mV) : Potassium reversal potential
    ELeak = -70.3 (mV) : Leak reversal potential
    VT = -56.2 (mV) : Spike threshold adjustment parameter
    gNabar = 0.056 (S/cm2) : Maximal conductance of iNa (at 36 deg. C)
    gKdbar = 0.006 (S/cm2) : Maximal conductance of iKd (at 36 deg. C)
    gLeak = 2.05e-05 (S/cm2) : Leak conductance (at 36 deg. C)
    gMbar = 7.5e-05 (S/cm2) : Maximal conductance of iM (at 36 deg. C)
    TauMax = 608 (ms) : Max. adaptation decay of slow non-inactivating Potassium current (at 36 deg. C)

    : Stimulus parameters
    Pamp = 0 : peak pressure amplitude (MPa)

    : Stimulus-driven current parameters
    a = 0 : multiplying factor for stimulus sensitivity on pressure (mA/cm2 * MPa)

    : Thermal model parameters
    Tref = 36  : reference temperature (in deg. C)
    alphaT = .017 : max temperature increase (in deg. C) per stimulus intensity unit
    tauT_abs = 100 : heat absorption time constant (ms)
    tauT_diss = 100  : heat dissipation time constant (ms)

    : Q10 coefficients
    Q10_rates = 1 : temperature dependence of gating transitions
    Q10_gNa = 1 : temperature dependence of iNa maximal conductance
    Q10_gKd = 1 : temperature dependence of iKd maximal conductance

    : Thermally-driven Potassium current parameters
    EKT = -93 (mV)  : KT current reversal potential
    gKT = 0 : rate of KT conductance linear increase with temperature, in S/(cm2 * deg. C)
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
    : Membrane potential
    v  (nC/cm2)
    
    : Membrane ionic currents
    iNa (mA/cm2)
    iKd (mA/cm2)
    iLeak (mA/cm2)
    iM (mA/cm2)
    iKT (mA/cm2)

    : Stimulus intensity (W/cm2)
    Isppa (W/cm2)

    : Stimulus-driven current
    iStim (mA/cm2)

    : Time-varying ion channel conductances
    gNabar_t (S/cm2)
    gKdbar_t (S/cm2)
    gMbar_t (S/cm2)
    gLeak_t (S/cm2)
    gKT_t (S/cm2)
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

FUNCTION pinf(v) {
    : Slow non-inactivating Potassium current steady-state activation probability
    pinf = 1.0 / (1 + exp(-(v + 35) / 10))  : (-)
}

FUNCTION taup(v) {
    : Slow non-inactivating Potassium current activation time constant
    taup = TauMax / (3.3 * exp((v + 35) / 20) + exp(-(v + 35) / 20))  : ms
}

: TEMPERATURE EVOLUTION FUNCTIONS

FUNCTION P2I(P) {  : convert acoustic pressure (in MPa) to acoustic intensity (in W/cm2)
    : [MPa]^2 / [MRayl] 
    := (1e6 * [kg/(m*s^2)])^2 / (1e6 * [kg/(m^2*s)])
    := 1e12 * kg^2/(m^2*s^4)] / (1e6 * [kg/(m^2*s)])
    := 1e6 [kg/s^3]
    := 1e6 [W/m2]
    := 1e2 [W/cm2]
    P2I = P*P / (2 * Z) * 1e2  : W/cm2
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
    p = pinf(v)

    : Initial temperature
    Isppa = P2I(Pamp)  : W/cm2
    T = Tinf(Isppa)  : deg. C
}

BREAKPOINT {
    : States integration
    SOLVE states METHOD cnexp

    : Update conductances based on temperature
    gLeak_t = gLeak
    gNabar_t = gNabar * phi(T, Q10_gNa)
    gKdbar_t = gKdbar * phi(T, Q10_gKd)
    gMbar_t = gMbar * phi(T, Q10_gKd)
    gKT_t = gKT * (T - Tref)

    : Stimulus-driven current computation
    iStim = - a * Pamp

    : Membrane currents computation
    iNa = gNabar_t * m * m * m * h * (v - ENa)
    iKd = gKdbar_t * n * n * n * n * (v - EK)
    iM = gMbar_t * p * (v - EK)
    iLeak = gLeak_t * (v - ELeak)
    iKT = gKT_t * (v - EKT)
}

DERIVATIVE states {
    : Ion channels gating states derivatives (voltage- and temperature-dependent)
    m' = (alpham(v) * (1 - m) - betam(v) * m) * phi(T, Q10_rates)
    h' = (alphah(v) * (1 - h) - betah(v) * h) * phi(T, Q10_rates)
    n' = (alphan(v) * (1 - n) - betan(v) * n) * phi(T, Q10_rates)
    p' = (pinf(v) - p) / taup(v) * phi(T, Q10_rates)

    : Temperature derivative
    Isppa = P2I(Pamp)  : W/cm2
    T' = (Tinf(Isppa) - T) / tauT(Isppa)  : deg. C/ms
}
