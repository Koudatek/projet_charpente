import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from charpente_entrais_brise import TrussAnalyzer as BaseTrussAnalyzer

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

class TrussAnalyzer(BaseTrussAnalyzer):
    def set_parameters(self, params):
        self.nom_projet = params["project_name"]
        self.nom_ingenieur = params["engineer_name"]
        self.geometry_type = params["geometry_type"]
        self.L = float(params["L"])
        self.m = int(params["m"])
        self.H = float(params["H"])
        self.alpha_deg = float(params["alpha_deg"])
        self.F = float(params["F"])

    def run_analysis(self, params):
        self.set_parameters(params)
        self.generate_structure(self.L, self.H, self.m)
        self.apply_loads(self.F)
        self.solve_system(self.L, self.F)
        self.calculate_internal_forces()
        # Générer et sauvegarder les figures
        plot_initial_path = os.path.join(STATIC_DIR, 'plot_initial.png')
        plot_deformed_path = os.path.join(STATIC_DIR, 'plot_deformed.png')
        self.plot_structure_initial().savefig(plot_initial_path)
        self.plot_deformed_structure().savefig(plot_deformed_path)
        plt.close('all')
        return {
            "nodes": self.nodes,
            "bars": self.bar_data,
            "displacements": self.displacements,
            "internal_forces": self.internal_forces,
            "plot_initial": "static/plot_initial.png",
            "plot_deformed": "static/plot_deformed.png"
        }

    def generate_pdf_report(self, filename=None):
        if filename is None:
            filename = os.path.join(STATIC_DIR, "report.pdf")
        super().generate_pdf_report(filename)
        return filename 