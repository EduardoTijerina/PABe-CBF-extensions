"""Parámetros globales del banco de pruebas PABe-CBF (simulación planar simplificada).

Todas las cifras físicas son valores de *simulación* elegidos para el banco de pruebas;
no son límites validados del Unitree G1 (ver Sec. 4.1 del manuscrito).
"""
import numpy as np

# --- Tiempo -------------------------------------------------------------------------
DT = 0.01                # periodo del lazo rápido CBF (100 Hz)
HMM_EVERY = 3            # el HMM corre cada 3 ciclos (~33 Hz): lazo lento desacoplado
T_MAX = 7.0              # duración máxima del episodio (s)
LATENCY_BUDGET_MS = 150.0

# --- Robot (brazo planar de 3 GDL, abstracción del WBC del G1) ----------------------
SHOULDER = np.array([0.0, -0.20])
LINKS = np.array([0.30, 0.28, 0.10])          # brazo, antebrazo, mano
U_MAX = 2.5                                    # rad/s por articulación
W_MAX = 0.03                                   # cota de perturbación articular (rad/s)
HOME_PALM = np.array([0.22, -0.38])

# --- Humano -------------------------------------------------------------------------
TORSO = np.array([1.05, 0.0])
H_SHOULDER = np.array([1.00, 0.20])
H_LINKS = (0.30, 0.26)
HAND_REST = np.array([0.92, 0.36])
MEET = np.array([0.56, 0.0])
V_MAX_H = 1.5            # cota de velocidad de la mano humana (m/s)
A_MAX_H = 1.5            # cota de aceleración usada para el conjunto alcanzable en oclusión (m/s^2)

# --- Percepción / cotas de error ----------------------------------------------------
EPS_D = 0.015            # |d - d_hat| <= EPS_D cuando hay visión
EPS_V = 0.10             # error de velocidad humana estimada (m/s)
EPS_F = 0.6              # |F - F_hat| <= EPS_F (N)
NOISE_POS = 0.006        # ruido uniforme por eje (cumple EPS_D)
NOISE_F = 0.5

# --- Barreras físicas ---------------------------------------------------------------
D_PHYS_LINK = 0.06       # separación mínima eslabones no autorizados <-> humano
D_PHYS_PALM_BODY = 0.05  # palma <-> muñeca/codo/torso humano
D_SEP_PALM = 0.07        # palma <-> mano humana fuera del modo C
R_C = 0.04               # distancia palma-mano en agarre
TOUCH = 0.005            # tolerancia de "toque"
K_C = 300.0              # rigidez de contacto (N/m)
F_PHYS = 20.0            # límite físico de fuerza (N)
F_HW = 18.0              # disparo de liberación por hardware
F_SOC0 = 12.0            # límite social nominal
F_PULL_EXPLICIT = 8.0    # tirón táctil explícito = señal de retirada
F_BREAK = 40.0           # deslizamiento del agarre
GRIP_DELAY = 0.08        # apertura de la mano robótica (s)
T_SHAKE = 1.2            # duración del apretón para cierre exitoso (s)
SAMPLE_MARGIN_D = 0.003  # margen de muestreo discreto (m)
SAMPLE_MARGIN_F = 0.5    # (N)
GAMMA_D = 6.0
GAMMA_F = 15.0

# --- Márgenes sociales theta = [theta_d, theta_F] ----------------------------------
THETA_MAX = np.array([0.10, 6.0])
THETA0 = np.array([0.0, 0.0])
LINK_THETA_GAIN = 0.5    # theta_{d,j} = c_j * theta_d para eslabones
K_R = np.array([0.08, 6.0])     # ganancia de p_W^+
K_E = np.array([0.05, 3.0])     # ganancia de E_exc (o H en B3/A2)
V_THETA_MAX = np.array([0.15, 15.0])
KAPPA = 4.0
RHO = np.array([20.0, 0.02])    # peso del término de v_theta en el QP
R_U = np.ones(3)

# --- Controlador nominal -------------------------------------------------------------
KP_NOM = 4.0
V_NOM_MAX = 0.6
K_NULL = 1.0

# --- Reglas de decisión ---------------------------------------------------------------
PERSIST_TICKS = 3
TAU_REL_PABE = 0.80
TAU_GATE_PABE = 0.50
C_PRUDENCE = 0.6
