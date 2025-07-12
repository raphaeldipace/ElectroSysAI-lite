import sys
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Configure OpenAI API
openai.api_key = 'YOUR OPENAI API KEY'

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ElectricalParams:
    phase_voltage: float = None
    line_voltage: float = None
    current: float = None
    line_current: float = None
    power_factor: float = None
    pf_type: str = 'lagging'
    phase_angle: float = 0.0
    selected_line: str = 'Vab'
    V_primary: float = None
    V_secondary: float = None
    I_primary: float = None
    length_km: float = None
    R_per_km: float = None
    X_per_km: float = None
    voltage: float = None

class ElectricalSystem(ABC):
    @abstractmethod
    def analyze(self):
        pass

class SinglePhaseSystem(ElectricalSystem):
    def __init__(self, p: ElectricalParams):
        self.p = p
    def analyze(self):
        P = self.p.phase_voltage * self.p.current * self.p.power_factor
        S = self.p.phase_voltage * self.p.current
        Q = np.sqrt(max(0, S**2 - P**2))
        return {'P (W)':P, 'Q (VAR)':Q, 'S (VA)':S}

class ThreePhaseSystem(ElectricalSystem):
    def __init__(self, p: ElectricalParams):
        self.p = p
        # store phase voltages with lowercase keys
        self.phases = {
            'vab': p.line_voltage * np.exp(1j * np.deg2rad(0 + p.phase_angle)),
            'vbc': p.line_voltage * np.exp(1j * np.deg2rad(-120 + p.phase_angle)),
            'vca': p.line_voltage * np.exp(1j * np.deg2rad(120 + p.phase_angle))
        }

    def analyze(self):
        # Select and validate phase
        key = self.p.selected_line.lower()
        if key not in self.phases:
            raise KeyError(
                f"Unknown phase '{self.p.selected_line}'. Choose among {list(self.phases.keys())}."
            )
        Vc = self.phases[key]
        Ic = self.p.line_current
        Z = Vc / Ic
        apparent = 3 * abs(Vc) * Ic
        P = apparent * self.p.power_factor
        Q = np.sqrt(max(0, apparent**2 - P**2))
        return {'Selected': self.p.selected_line, 'Z (Ω)': Z, 'P (W)': P, 'Q (VAR)': Q}

class TransformerSystem(ElectricalSystem):
    def __init__(self, p: ElectricalParams):
        self.p = p
    def analyze(self):
        turns_ratio = self.p.V_secondary / self.p.V_primary
        I_secondary = self.p.I_primary * (self.p.V_primary / self.p.V_secondary)
        return {'Turns_ratio':turns_ratio, 'I_secondary (A)':I_secondary}

class TransmissionLineSystem(ElectricalSystem):
    def __init__(self, p: ElectricalParams):
        self.p = p
    def analyze(self):
        R = self.p.R_per_km * self.p.length_km
        X = self.p.X_per_km * self.p.length_km
        Z = R + 1j*X
        I = self.p.voltage / Z
        return {'Z (Ω)':Z, 'I (A)':I}

# Plot routines

def plot_pq_s_curve(params: ElectricalParams, three_phase=False):
    sns.set(style="whitegrid")
    pf = np.linspace(0.2,1,50)
    V = params.line_voltage if three_phase else params.phase_voltage
    I = params.line_current if three_phase else params.current
    S = V*I*(np.sqrt(3) if three_phase else 1)
    P = S*pf
    Q = np.sqrt(np.maximum(0, S**2 - P**2))
    plt.figure(); plt.plot(pf,P,label='P'), plt.plot(pf,Q,label='Q'), plt.plot(pf,[S]*len(pf),label='S')
    plt.legend(); plt.title('P-Q-S vs PF'); plt.xlabel('PF'); plt.ylabel('Power'); plt.show()


def plot_3d_surface(V_range, I_range, pf=0.9):
    Vg, Ig = np.meshgrid(V_range, I_range)
    P = Vg*Ig*pf*np.sqrt(3)
    fig=plt.figure(); ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(Vg,Ig,P,cmap='plasma'); ax.set_xlabel('V'); ax.set_ylabel('I'); ax.set_zlabel('P'); plt.show()


def plot_phase_currents(phases, I_mag):
    sns.set(style='whitegrid')
    plt.figure()
    for name,Vc in phases.items():
        I = np.linspace(0,I_mag,50)
        P = np.real(Vc)*I
        plt.plot(I,P,label=name)
    plt.legend(); plt.title('Power vs Current per Phase'); plt.xlabel('I (A)'); plt.ylabel('Power (W)'); plt.show()

# AI assistant using v1.0 interface with error handling

def ai_suggest(params: dict):
    """Call AI for suggestions; failures are logged and skipped."""
    client = openai.OpenAI(api_key=openai.api_key)
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role':'system','content':'EE AI Assistant.'},
                {'role':'user','content':f'Analyze {params} and suggest enhancements in JSON.'}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.warning(f'AI suggestion skipped due to error: {e}')
        return None

# Interactive menu

if __name__=='__main__':
    print("Select system: 1) Single-phase 2) Three-phase 3) Transformer 4) Transmission line")
    choice=input('Option: ').strip()
    params=ElectricalParams()
    if choice=='1':
        params.phase_voltage=float(input('V (V): ')); params.current=float(input('I (A): ')); params.power_factor=float(input('PF: '))
        sys=SinglePhaseSystem(params)
        res=sys.analyze(); print(res); plot_pq_s_curve(params)
    elif choice=='2':
        params.line_voltage=float(input('Line V (V): ')); params.line_current=float(input('Line I (A): ')); params.power_factor=float(input('PF: '))
        params.pf_type = input('lagging or leading: ')
        params.selected_line = input('Select Vab, Vbc, Vca: ').strip().lower()
        params.phase_angle=float(input('Phase shift (deg): '))
        sys=ThreePhaseSystem(params)
        res=sys.analyze(); print(res)
        plot_pq_s_curve(params,three_phase=True); plot_phase_currents(sys.phases,params.line_current); plot_3d_surface(np.linspace(0,params.line_voltage,30),np.linspace(0,params.line_current,30),params.power_factor)
    elif choice=='3':
        params.V_primary=float(input('Vp (V): ')); params.V_secondary=float(input('Vs (V): ')); params.I_primary=float(input('Ip (A): '))
        sys=TransformerSystem(params); print(sys.analyze())
    elif choice=='4':
        params.length_km=float(input('Length (km): ')); params.R_per_km=float(input('R Ω/km: ')); params.X_per_km=float(input('X Ω/km: ')); params.voltage=float(input('V (V): '))
        sys=TransmissionLineSystem(params); print(sys.analyze())
    else:
        print('Invalid.')
    # AI suggestion for robustness
    suggestion = ai_suggest(params.__dict__)
    if suggestion:
        print('AI Suggestion:', suggestion)
    print('Done.')
