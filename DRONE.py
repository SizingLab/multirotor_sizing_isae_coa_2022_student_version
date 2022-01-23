import openmdao.api as om
import fastoad.api as oad
import numpy as np


@oad.RegisterOpenMDAOSystem('drone')
class DRONE(om.Group):

	def setup(self):
		self.add_subsystem("SCENARIOS", SCENARIOS(), promotes=["*"])
		self.add_subsystem("PROPELLER", PROPELLER(), promotes=["*"])
		self.add_subsystem("MOTOR", MOTOR(), promotes=["*"])
		self.add_subsystem("BATTERY", BATTERY(), promotes=["*"])
		self.add_subsystem("ESC", ESC(), promotes=["*"])
		self.add_subsystem("FRAME", FRAME(), promotes=["*"])
		self.add_subsystem("OBJECTIVES", OBJECTIVES(), promotes=["*"])
		self.add_subsystem("CONSTRAINTS", CONSTRAINTS(), promotes=["*"])


class SCENARIOS(om.ExplicitComponent):
    def setup(self):
        self.add_input("optim:variable:k_os", val=np.nan)
        self.add_input("specifications:payload:mass:max", val=np.nan, units="kg")
        self.add_input("data:structure:arms:prop_per_arm", val=np.nan)
        self.add_input("data:structure:arms:number", val=np.nan)
        self.add_input(
            "specifications:acceleration:takeoff", val=np.nan, units="m/s**2"
        )
        self.add_output("data:system:MTOW:guess", units="kg")
        self.add_output("data:propeller:number")
        self.add_output("data:propeller:thrust:hover", units="N")
        self.add_output("data:propeller:thrust:takeoff", units="N")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        k_os = inputs["optim:variable:k_os"]
        M_pay = inputs["specifications:payload:mass:max"]
        N_pro_arm = inputs["data:structure:arms:prop_per_arm"]
        N_arm = inputs["data:structure:arms:number"]
        a_to = inputs["specifications:acceleration:takeoff"]

        # ---
        M_total = (
            k_os * M_pay
        )  # [kg] Estimation of the total mass (or equivalent weight of dynamic scenario)
        N_pro = N_pro_arm * N_arm  # Propellers number
        F_pro_hov = M_total * (9.81) / N_pro  # [N] Thrust per propeller for hover
        F_pro_to = (
            M_total * (9.81 + a_to) / N_pro
        )  # [N] Thrust per propeller for take-off

        outputs["data:system:MTOW:guess"] = M_total
        outputs["data:propeller:number"] = N_pro
        outputs["data:propeller:thrust:hover"] = F_pro_hov
        outputs["data:propeller:thrust:takeoff"] = F_pro_to

class PROPELLER(om.ExplicitComponent):
    def setup(self):
        self.add_input("optim:variable:beta_pro", val=np.nan)
        self.add_input("data:propeller:thrust:takeoff", val=np.nan, units="N")
        self.add_input("specifications:atmosphere:density", val=np.nan, units="kg/m**3")
        self.add_input("data:propeller:reference:ND:max", val=np.nan, units="Hz*m")
        self.add_input("optim:variable:k_ND", val=np.nan)
        self.add_input("data:propeller:reference:mass", val=np.nan, units="kg")
        self.add_input("data:propeller:reference:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:thrust:hover", val=np.nan, units="N")
        self.add_input("optim:variable:k_vb", val=np.nan)
        self.add_output("data:propeller:aerodynamics:CT")
        self.add_output("data:propeller:aerodynamics:CP")
        self.add_output("data:propeller:geometry:diameter", units="m")
        self.add_output("data:propeller:speed_Hz:takeoff", units="Hz")
        self.add_output("data:propeller:speed_rad_s:takeoff", units="rad/s")
        self.add_output("data:propeller:mass", units="kg")
        self.add_output("data:propeller:power:takeoff", units="W")
        self.add_output("data:propeller:torque:takeoff", units="N*m")
        self.add_output("data:propeller:speed_Hz:hover", units="Hz")
        self.add_output("data:propeller:speed_rad_s:hover", units="rad/s")
        self.add_output("data:propeller:power:hover", units="W")
        self.add_output("data:propeller:torque:hover", units="N*m")
        self.add_output("data:battery:voltage:guess", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        beta_pro = inputs["optim:variable:beta_pro"]
        F_pro_to = inputs["data:propeller:thrust:takeoff"]
        rho_air = inputs["specifications:atmosphere:density"]
        ND_max = inputs["data:propeller:reference:ND:max"]
        k_ND = inputs["optim:variable:k_ND"]
        M_pro_ref = inputs["data:propeller:reference:mass"]
        D_pro_ref = inputs["data:propeller:reference:diameter"]
        F_pro_hov = inputs["data:propeller:thrust:hover"]
        k_vb = inputs["optim:variable:k_vb"]

        # ---
        C_t = (
            4.27e-02 + 1.44e-01 * beta_pro
        )  # Thrust coef with T=C_T.rho.n^2.D^4 - 0.8 for de-rating of APC catalog
        C_p = -1.48e-03 + 9.72e-02 * beta_pro  # Power coef with P=C_p.rho.n^3.D^5
        # Propeller selection with take-off scenario
        D_pro = (
            F_pro_to / (C_t * rho_air * (ND_max / k_ND) ** 2.0)
        ) ** 0.5  # [m] Propeller diameter
        n_pro_to = ND_max / k_ND / D_pro  # [Hz] Propeller speed
        Omega_pro_to = n_pro_to * 2 * np.pi  # [rad/s] Propeller speed
        M_pro = M_pro_ref * (D_pro / D_pro_ref) ** 2.0  # [kg] Propeller mass
        P_pro_to = (
            C_p * rho_air * n_pro_to ** 3.0 * D_pro ** 5.0
        )  # [W] Power per propeller
        T_pro_to = P_pro_to / Omega_pro_to  # [N*m] Propeller torque
        # Propeller torque & speed for hover
        n_pro_hov = np.sqrt(
            F_pro_hov / (C_t * rho_air * D_pro ** 4.0)
        )  # [Hz] hover speed
        Omega_pro_hov = n_pro_hov * 2.0 * np.pi  # [rad/s] Propeller speed
        P_pro_hov = (
            C_p * rho_air * n_pro_hov ** 3.0 * D_pro ** 5.0
        )  # [W] Power per propeller
        T_pro_hov = P_pro_hov / Omega_pro_hov  # [N*m] Propeller torque
        U_bat_est = k_vb * 1.84 * P_pro_to ** (0.36)  # [V] battery voltage estimation

        outputs["data:propeller:aerodynamics:CT"] = C_t
        outputs["data:propeller:aerodynamics:CP"] = C_p
        outputs["data:propeller:geometry:diameter"] = D_pro
        outputs["data:propeller:speed_Hz:takeoff"] = n_pro_to
        outputs["data:propeller:speed_rad_s:takeoff"] = Omega_pro_to
        outputs["data:propeller:mass"] = M_pro
        outputs["data:propeller:power:takeoff"] = P_pro_to
        outputs["data:propeller:torque:takeoff"] = T_pro_to
        outputs["data:propeller:speed_Hz:hover"] = n_pro_hov
        outputs["data:propeller:speed_rad_s:hover"] = Omega_pro_hov
        outputs["data:propeller:power:hover"] = P_pro_hov
        outputs["data:propeller:torque:hover"] = T_pro_hov
        outputs["data:battery:voltage:guess"] = U_bat_est

class MOTOR(om.ExplicitComponent):
    def setup(self):
        self.add_input("optim:variable:k_mot", val=np.nan)
        self.add_input("data:propeller:torque:hover", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:mass", val=np.nan, units="kg")
        self.add_input("data:motor:reference:torque:nominal", val=np.nan, units="N*m")
        self.add_input("data:battery:voltage:guess", val=np.nan, units="V")
        self.add_input("optim:variable:k_speed_mot", val=np.nan)
        self.add_input("data:propeller:speed_rad_s:takeoff", val=np.nan, units="rad/s")
        self.add_input("data:motor:reference:resistance", val=np.nan, units="ohm")
        self.add_input(
            "data:motor:reference:torque:coefficient", val=np.nan, units="N*m/A"
        )
        self.add_input("data:motor:reference:torque:friction", val=np.nan, units="N*m")
        self.add_input("data:motor:reference:torque:max", val=np.nan, units="N*m")
        self.add_input("data:propeller:speed_rad_s:hover", val=np.nan, units="rad/s")
        self.add_input("data:propeller:torque:takeoff", val=np.nan, units="N*m")
        self.add_output("data:motor:torque:nominal", units="N*m")
        self.add_output("data:motor:mass", units="kg")
        self.add_output("data:motor:torque:coefficient", units="N*m/A")
        self.add_output("data:motor:resistance", units="ohm")
        self.add_output("data:motor:torque:friction", units="N*m")
        self.add_output("data:motor:torque:max")
        self.add_output("data:motor:current:hover", units="A")
        self.add_output("data:motor:voltage:hover", units="V")
        self.add_output("data:motor:power:hover", units="W")
        self.add_output("data:motor:current:takeoff", units="A")
        self.add_output("data:motor:voltage:takeoff", units="V")
        self.add_output("data:motor:power:takeoff", units="W")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        k_mot = inputs["optim:variable:k_mot"]
        T_pro_hov = inputs["data:propeller:torque:hover"]
        M_mot_ref = inputs["data:motor:reference:mass"]
        T_nom_mot_ref = inputs["data:motor:reference:torque:nominal"]
        U_bat_est = inputs["data:battery:voltage:guess"]
        k_speed_mot = inputs["optim:variable:k_speed_mot"]
        Omega_pro_to = inputs["data:propeller:speed_rad_s:takeoff"]
        R_mot_ref = inputs["data:motor:reference:resistance"]
        K_mot_ref = inputs["data:motor:reference:torque:coefficient"]
        T_mot_fr_ref = inputs["data:motor:reference:torque:friction"]
        T_max_mot_ref = inputs["data:motor:reference:torque:max"]
        Omega_pro_hov = inputs["data:propeller:speed_rad_s:hover"]
        T_pro_to = inputs["data:propeller:torque:takeoff"]

        # ---
        T_nom_mot = k_mot * T_pro_hov  # [N*m] Motor nominal torque per propeller
        M_mot = M_mot_ref * (T_nom_mot / T_nom_mot_ref) ** (
            3.0 / 3.5
        )  # [kg] Motor mass
        # Selection with take-off speed
        K_mot = U_bat_est / (
            k_speed_mot * Omega_pro_to
        )  # [N*m/A] or [V/(rad/s)] Kt motor
        R_mot = (
            R_mot_ref
            * (T_nom_mot / T_nom_mot_ref) ** (-5.0 / 3.5)
            * (K_mot / K_mot_ref) ** 2.0
        )  # [ohm] motor resistance
        T_mot_fr = T_mot_fr_ref * (T_nom_mot / T_nom_mot_ref) ** (
            3.0 / 3.5
        )  # [N*m] Friction torque
        T_max_mot = T_max_mot_ref * (T_nom_mot / T_nom_mot_ref)
        # Hover current and voltage
        I_mot_hov = (
            T_pro_hov + T_mot_fr
        ) / K_mot  # [A] Current of the motor per propeller
        U_mot_hov = (
            R_mot * I_mot_hov + Omega_pro_hov * K_mot
        )  # [V] Voltage of the motor per propeller
        P_el_mot_hov = U_mot_hov * I_mot_hov  # [W] Hover : electrical power
        # Takeoff current and voltage
        I_mot_to = (
            T_pro_to + T_mot_fr
        ) / K_mot  # [A] Current of the motor per propeller
        U_mot_to = (
            R_mot * I_mot_to + Omega_pro_to * K_mot
        )  # [V] Voltage of the motor per propeller
        P_el_mot_to = U_mot_to * I_mot_to  # [W] Takeoff : electrical power

        outputs["data:motor:torque:nominal"] = T_nom_mot
        outputs["data:motor:mass"] = M_mot
        outputs["data:motor:torque:coefficient"] = K_mot
        outputs["data:motor:resistance"] = R_mot
        outputs["data:motor:torque:friction"] = T_mot_fr
        outputs["data:motor:torque:max"] = T_max_mot
        outputs["data:motor:current:hover"] = I_mot_hov
        outputs["data:motor:voltage:hover"] = U_mot_hov
        outputs["data:motor:power:hover"] = P_el_mot_hov
        outputs["data:motor:current:takeoff"] = I_mot_to
        outputs["data:motor:voltage:takeoff"] = U_mot_to
        outputs["data:motor:power:takeoff"] = P_el_mot_to

class BATTERY(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:battery:voltage:guess", val=np.nan, units="V")
        self.add_input("optim:variable:k_mb", val=np.nan)
        self.add_input("specifications:payload:mass:max", val=np.nan, units="kg")
        self.add_input("data:battery:reference:energy", val=np.nan, units="J")
        self.add_input("data:battery:reference:mass", val=np.nan, units="kg")
        self.add_input("data:motor:power:hover", val=np.nan, units="W")
        self.add_input("data:propeller:number", val=np.nan)
        self.add_output("data:battery:cell:number:series")
        self.add_output("data:battery:voltage", units="V")
        self.add_output("data:battery:mass", units="kg")
        self.add_output("data:battery:energy", units="J")
        self.add_output("data:battery:capacity", units="A*s")
        self.add_output("data:battery:current", units="A")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        U_bat_est = inputs["data:battery:voltage:guess"]
        k_mb = inputs["optim:variable:k_mb"]
        M_pay = inputs["specifications:payload:mass:max"]
        E_bat_ref = inputs["data:battery:reference:energy"]
        M_bat_ref = inputs["data:battery:reference:mass"]
        P_el_mot_hov = inputs["data:motor:power:hover"]
        N_pro = inputs["data:propeller:number"]

        # ---
        # Battery selection & scaling laws sized from hover
        N_s_bat = U_bat_est / 3.7  # Cell number, round (up value)
        U_bat = 3.7 * N_s_bat  # [V] Battery voltage
        M_bat = k_mb * M_pay  # [kg] Battery mass
        # Hover --> autonomy
        E_bat = (
            E_bat_ref * M_bat / M_bat_ref * 0.8
        )  # [J] Energy  of the battery (.8 coefficient because 80% use only of the total capacity)
        C_bat = E_bat / U_bat  # [A*s] Capacity  of the battery
        I_bat = (P_el_mot_hov * N_pro) / 0.95 / U_bat  # [A] Current of the battery

        outputs["data:battery:cell:number:series"] = N_s_bat
        outputs["data:battery:voltage"] = U_bat
        outputs["data:battery:mass"] = M_bat
        outputs["data:battery:energy"] = E_bat
        outputs["data:battery:capacity"] = C_bat
        outputs["data:battery:current"] = I_bat

class ESC(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_input("data:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:ESC:reference:mass", val=np.nan, units="kg")
        self.add_input("data:ESC:reference:power", val=np.nan, units="W")
        self.add_output("data:ESC:power", units="W")
        self.add_output("data:ESC:mass", units="kg")
        self.add_output("data:ESC:voltage", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        P_el_mot_to = inputs["data:motor:power:takeoff"]
        U_bat = inputs["data:battery:voltage"]
        U_mot_to = inputs["data:motor:voltage:takeoff"]
        M_esc_ref = inputs["data:ESC:reference:mass"]
        P_esc_ref = inputs["data:ESC:reference:power"]

        P_esc = (
            P_el_mot_to * U_bat / U_mot_to
        )  # [W] power electronic power (corner power or apparent power)
        M_esc = M_esc_ref * (P_esc / P_esc_ref)  # [kg] Mass ESC
        V_esc = 1.84 * P_esc ** 0.36  # [V] ESC voltage

        outputs["data:ESC:power"] = P_esc
        outputs["data:ESC:mass"] = M_esc
        outputs["data:ESC:voltage"] = V_esc

class FRAME(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:structure:arms:number", val=np.nan)
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:thrust:takeoff", val=np.nan, units="N")
        self.add_input("data:structure:arms:prop_per_arm", val=np.nan)
        self.add_input(
            "data:structure:arms:material:stress:max", val=np.nan, units="N/m**2"
        )
        self.add_input("optim:variable:k_D", val=np.nan)
        self.add_input(
            "data:structure:arms:material:density", val=np.nan, units="kg/m**3"
        )
        self.add_output("data:structure:arms:angle", units="rad")
        self.add_output("data:structure:arms:length", units="m")
        self.add_output("data:structure:arms:diameter:outer", units="m")
        self.add_output("data:structure:arms:diameter:inner", units="m")
        self.add_output("data:structure:arms:mass", units="kg")
        self.add_output("data:structure:body:mass", units="kg")
        self.add_output("data:structure:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        N_arm = inputs["data:structure:arms:number"]
        D_pro = inputs["data:propeller:geometry:diameter"]
        F_pro_to = inputs["data:propeller:thrust:takeoff"]
        N_pro_arm = inputs["data:structure:arms:prop_per_arm"]
        sigma_max = inputs["data:structure:arms:material:stress:max"]
        k_D = inputs["optim:variable:k_D"]
        rho_s = inputs["data:structure:arms:material:density"]

        # ---
        alpha_sep = (
            2 * np.pi / N_arm
        )  # [rad] interior angle separation between propellers
        L_arm = D_pro / (2.0 * np.sin(alpha_sep / 2.0))  # [m] length of the arm
        # Tube diameter & thickness
        D_out_arm = (
            F_pro_to * N_pro_arm * L_arm * 32 / (np.pi * sigma_max * (1 - k_D ** 4))
        ) ** (
            1 / 3
        )  # [m] outer diameter of the arm (hollow cylinder)
        D_in_arm = k_D * D_out_arm  # [m] inner diameter of the arm (hollow cylinder)
        # Mass
        M_arms = (
            np.pi
            / 4
            * (D_out_arm ** 2 - (k_D * D_out_arm) ** 2)
            * L_arm
            * rho_s
            * N_arm
        )  # [kg] mass of the arms
        M_body = 1.5 * M_arms  # [kg] mass of the body (40% of total mass is the arms)
        M_frame = M_body + M_arms  # [kg] total mass of the frame

        outputs["data:structure:arms:angle"] = alpha_sep
        outputs["data:structure:arms:length"] = L_arm
        outputs["data:structure:arms:diameter:outer"] = D_out_arm
        outputs["data:structure:arms:diameter:inner"] = D_in_arm
        outputs["data:structure:arms:mass"] = M_arms
        outputs["data:structure:body:mass"] = M_body
        outputs["data:structure:mass"] = M_frame

class OBJECTIVES(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:battery:capacity", val=np.nan, units="A*s")
        self.add_input("data:battery:current", val=np.nan, units="A")
        self.add_input("data:ESC:mass", val=np.nan, units="kg")
        self.add_input("data:propeller:mass", val=np.nan, units="kg")
        self.add_input("data:motor:mass", val=np.nan, units="kg")
        self.add_input("data:propeller:number", val=np.nan)
        self.add_input("specifications:payload:mass:max", val=np.nan, units="kg")
        self.add_input("data:battery:mass", val=np.nan, units="kg")
        self.add_input("data:structure:mass", val=np.nan, units="kg")
        self.add_output("optim:objective:autonomy:hover", units="min")
        self.add_output("optim:objective:MTOW", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        C_bat = inputs["data:battery:capacity"]
        I_bat = inputs["data:battery:current"]
        M_esc = inputs["data:ESC:mass"]
        M_pro = inputs["data:propeller:mass"]
        M_mot = inputs["data:motor:mass"]
        N_pro = inputs["data:propeller:number"]
        M_pay = inputs["specifications:payload:mass:max"]
        M_bat = inputs["data:battery:mass"]
        M_frame = inputs["data:structure:mass"]

        # ---
        t_hov = C_bat / I_bat / 60.0  # [min] Hover time
        M_total_real = (
            (M_esc + M_pro + M_mot) * N_pro + M_pay + M_bat + M_frame
        )  # [kg] Total mass

        outputs["optim:objective:autonomy:hover"] = t_hov
        outputs["optim:objective:MTOW"] = M_total_real

class CONSTRAINTS(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:system:MTOW:guess", val=np.nan, units="kg")
        self.add_input("optim:objective:MTOW", val=np.nan, units="kg")
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_input("data:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:motor:torque:max", val=np.nan)
        self.add_input("data:propeller:torque:takeoff", val=np.nan, units="N*m")
        self.add_input("data:ESC:voltage", val=np.nan, units="V")
        self.add_input("optim:objective:autonomy:hover", val=np.nan, units="min")
        self.add_input("specifications:duration:hover", val=np.nan, units="min")
        self.add_input("specifications:MTOW", val=np.nan, units="kg")
        self.add_output("optim:constraint:c_1")
        self.add_output("optim:constraint:c_2")
        self.add_output("optim:constraint:c_3")
        self.add_output("optim:constraint:c_4")
        self.add_output("optim:constraint:c_5")
        self.add_output("optim:constraint:c_6")
        self.add_output("optim:constraint:c_7")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        M_total = inputs["data:system:MTOW:guess"]
        M_total_real = inputs["optim:objective:MTOW"]
        U_bat = inputs["data:battery:voltage"]
        U_mot_to = inputs["data:motor:voltage:takeoff"]
        T_max_mot = inputs["data:motor:torque:max"]
        T_pro_to = inputs["data:propeller:torque:takeoff"]
        V_esc = inputs["data:ESC:voltage"]
        t_hov = inputs["optim:objective:autonomy:hover"]
        t_hov_spec = inputs["specifications:duration:hover"]
        MTOW = inputs["specifications:MTOW"]

        cons_1 = M_total - M_total_real
        cons_2 = U_bat - U_mot_to
        cons_3 = T_max_mot - T_pro_to
        cons_4 = U_bat - V_esc
        cons_5 = V_esc - U_mot_to
        cons_6 = t_hov - t_hov_spec
        cons_7 = MTOW - M_total_real

        outputs["optim:constraint:c_1"] = cons_1
        outputs["optim:constraint:c_2"] = cons_2
        outputs["optim:constraint:c_3"] = cons_3
        outputs["optim:constraint:c_4"] = cons_4
        outputs["optim:constraint:c_5"] = cons_5
        outputs["optim:constraint:c_6"] = cons_6
        outputs["optim:constraint:c_7"] = cons_7

