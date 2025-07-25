import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import datetime

class TrussAnalyzer:
    def __init__(self):
        self.geometry_type = "howe"  # défaut
        self.nom_projet = None
        self.nom_ingenieur = None
        self.nodes = []
        self.node_dict = {}
        self.bars = []
        self.bar_data = []
        self.forces = {}
        self.displacements = None
        self.reactions = None
        self.internal_forces = None
        self.strains = {}
        self.angular_variations = {}
        self.E = 210e9  # Module d'Young acier (Pa)
        self.A = 0.0001
        self.sections = {
            'entrait': 0.00015,       # 15 cm²
            'arbaletrier': 0.00012,   # 12 cm²
            'montant': 0.00008,       # 8 cm²
            'diagonale': 0.00006      # 6 cm²
        }

        self.K_global = None
        self.bar_stiffness_matrices = {}
        self.L = None
        self.H = None
        self.m = None
        self.F = None

    def get_input_parameters(self):
        """Demande à l'utilisateur les paramètres nécessaires selon la géométrie choisie"""
        print("🔧 Saisie des paramètres du projet")
        self.project_name = input("Donner le nom du projet : ")
        self.engineer_name = input("Donner le nom de l'ingénieur : ")

        while True:
            geo = input("Choisissez la géométrie ['howe_brise', 'pratt_brise', 'warren_brise_poincon'] : ").strip().lower()
            if geo in ['howe_brise', 'pratt_brise', 'warren_brise_poincon']:
                self.geometry_type = geo
                break
            else:
                print("❌ Géométrie invalide. Réessayez.")

        while True:
            try:
                self.L = float(input("Donner la demi-portée L (en m) : "))
                break
            except ValueError:
                print("❌ Valeur invalide pour L.")

        while True:
            try:
                self.m = int(input("Donner le nombre de divisions m (pair) : "))
                if self.m % 2 == 0:
                    break
                else:
                    print("❌ m doit être un nombre pair.")
            except ValueError:
                print("❌ Valeur invalide pour m.")

        while True:
            try:
                self.H = float(input("Donner la hauteur H du poinçon (en m) : "))
                break
            except ValueError:
                print("❌ Valeur invalide pour H.")

        while True:
            try:
                self.alpha_deg = float(input("Donner l'angle de brisure alpha ou beta (en degrés) : "))
                break
            except ValueError:
                print("❌ Valeur invalide pour l'angle.")

    def generate_structure_howe_brise(self, L, H, m, alpha_deg):
        import math
        alpha_rad = math.radians(alpha_deg)
        e = L / m
        self.nodes = []
        self.node_dict = {}
        coord_to_label = {}
        node_id = 1

        def add_node(x, y):
            nonlocal node_id
            key = (round(x, 6), round(y, 6))
            if key in coord_to_label:
                return coord_to_label[key]
            label = f"n{node_id}"
            self.nodes.append((label, x, y))
            self.node_dict[label] = (x, y)
            coord_to_label[key] = label
            node_id += 1
            return label

        base_left = []
        for i in range(m + 1):
            x = i * e
            y = math.tan(alpha_rad) * x
            base_left.append(add_node(x, y))

        x_p = L
        y_p = self.node_dict[base_left[-1]][1] + H
        poincon = add_node(x_p, y_p)

        top_left = []
        for i in range(1, m):
            x = i * e
            y_base = self.node_dict[base_left[i]][1]
            y = y_base + (H * (i / m))
            top_left.append(add_node(x, y))

        base_right = []
        for i in range(m + 1):
            x_left, y_left = self.node_dict[base_left[m - i]]
            x_sym = 2 * x_p - x_left
            base_right.insert(0, add_node(x_sym, y_left))

        top_right = []
        for i in range(m - 1):
            x_left, y_left = self.node_dict[top_left[m - 2 - i]]
            x_sym = 2 * x_p - x_left
            top_right.insert(0, add_node(x_sym, y_left))

        bars_set = set()

        def add_bar(a, b):
            bars_set.add(tuple(sorted((a, b))))

        for i in range(m):
            add_bar(base_left[i], base_left[i + 1])
            add_bar(base_right[i], base_right[i + 1])

        add_bar(base_left[0], top_left[0])
        for i in range(len(top_left) - 1):
            add_bar(top_left[i], top_left[i + 1])
        add_bar(top_left[-1], base_left[-1])
        add_bar(top_left[-1], poincon)
        add_bar(base_left[-1], poincon)

        add_bar(base_right[0], top_right[0])
        for i in range(len(top_right) - 1):
            add_bar(top_right[i], top_right[i + 1])
        add_bar(top_right[-1], base_right[-1])
        add_bar(top_right[-1], poincon)
        add_bar(base_right[-1], poincon)

        for i in range(1, m):
            add_bar(base_left[i], top_left[i - 1])
            add_bar(base_right[i], top_right[i - 1])

        for i in range(1, m):
            add_bar(top_left[i - 1], base_left[i + 1])
            add_bar(top_right[i - 1], base_right[i + 1])

        self.bars = sorted(list(bars_set), key=lambda x: (int(x[0][1:]), int(x[1][1:])))
        self.bar_data = [(f"B{idx+1}", a, b) for idx, (a, b) in enumerate(self.bars)]

    def generate_structure_pratt_brise(self, L, H, m, theta_deg):
        """
        Génère une structure en treillis de type Pratt avec entrait brisé.

        Paramètres:
        - L: Demi-portée (longueur de la moitié de la structure)
        - H: Hauteur du poinçon
        - m: Nombre d'entraxes sur la demi-portée
        - theta_deg: Angle de l'entrait brisé en degrés (0 pour entrait droit)
        """
        import math

        # Conversion de l'angle en radians
        theta_rad = math.radians(theta_deg)

        # Calcul de l'entraxe entre nœuds
        e = L / m

        # Initialisation des listes et dictionnaires
        self.nodes = []
        self.node_dict = {}
        node_id = 1

        # Base gauche (entrait brisé)
        base_left = []
        for i in range(m + 1):
            x = i * e
            y = math.tan(theta_rad) * x  # Entrait brisé avec angle theta
            label = f"n{node_id}"
            self.nodes.append((label, x, y))
            self.node_dict[label] = (x, y)
            base_left.append(label)
            node_id += 1

        # Poinçon (ajusté selon l'entrait brisé)
        x_p = L
        y_p = math.tan(theta_rad) * L + H  # Hauteur ajustée
        label_poincon = f"n{node_id}"
        self.nodes.append((label_poincon, x_p, y_p))
        self.node_dict[label_poincon] = (x_p, y_p)
        node_id += 1

        # Arbalétriers gauche (ajustés selon l'entrait brisé)
        upper_left = []
        for i in range(1, m):
            x = (m - i) * e
            y_base = math.tan(theta_rad) * x  # Hauteur de base à cette position
            y = y_base + (m - i) * H / m     # Hauteur totale
            label = f"n{node_id}"
            self.nodes.append((label, x, y))
            self.node_dict[label] = (x, y)
            upper_left.append(label)
            node_id += 1

        bars_left = []

        # Arbalétriers (membrures supérieures)
        bars_left.append((base_left[0], upper_left[-1]))
        for i in range(len(upper_left) - 1, 0, -1):
            bars_left.append((upper_left[i], upper_left[i - 1]))
        bars_left.append((upper_left[0], label_poincon))

        # Entrait brisé (membrure inférieure)
        for i in range(m):
            bars_left.append((base_left[i], base_left[i + 1]))

        # Montants (éléments verticaux)
        for i in range(1, m):
            bars_left.append((base_left[i], upper_left[-i]))

        # Diagonales GAUCHE : du coin bas-gauche vers le coin haut-droit de chaque panneau
        for i in range(m):
            if i < m - 1:
                # Panneaux normaux : base_left[i] vers upper_left[-(i+1)]
                bars_left.append((base_left[i], upper_left[-(i+1)]))
            else:
                # Dernier panneau (proche de l'axe de symétrie) : base_left[i] vers poinçon
                bars_left.append((base_left[i], label_poincon))

        # Création des nœuds de la partie droite par symétrie
        base_right = []
        symmetry_map = {}
        for i in range(m):
            x, y = self.node_dict[base_left[i]]
            x_sym = 2 * x_p - x  # Symétrie par rapport au poinçon
            label = f"n{node_id}"
            self.nodes.append((label, x_sym, y))
            self.node_dict[label] = (x_sym, y)
            base_right.append(label)
            symmetry_map[base_left[i]] = label
            node_id += 1
        # Le nœud central reste le même
        symmetry_map[base_left[-1]] = base_left[-1]

        upper_right = []
        for i, label in enumerate(upper_left):
            x, y = self.node_dict[label]
            x_sym = 2 * x_p - x  # Symétrie par rapport au poinçon
            label_new = f"n{node_id}"
            self.nodes.append((label_new, x_sym, y))
            self.node_dict[label_new] = (x_sym, y)
            upper_right.append(label_new)
            symmetry_map[label] = label_new
            node_id += 1

        # Création des barres de la partie droite par symétrie
        bars_right = []
        for a, b in bars_left:
            if a in symmetry_map and b in symmetry_map:
                bars_right.append((symmetry_map[a], symmetry_map[b]))
            elif a == label_poincon:
                bars_right.append((label_poincon, symmetry_map[b]))
            elif b == label_poincon:
                bars_right.append((symmetry_map[a], label_poincon))

        # Assemblage final des barres
        bars = bars_left + bars_right
        # Ajout de la barre du poinçon au centre
        bars.append((base_left[-1], label_poincon))

        # Stockage des données finales
        self.bars = bars
        self.bar_data = [(f"B{idx+1}", a, b) for idx, (a, b) in enumerate(bars)]

    def generate_structure_warren_brise_poincon(self, L, H, m, beta_deg):
        """Génère une structure de type Warren avec entrait brisé et poinçon."""
        from math import radians, tan, atan

        beta_rad = radians(beta_deg)
        theta_rad = atan(H / L)
        espacement = L / (m // 2)

        self.nodes = []
        self.node_dict = {}
        self.bars = []
        self.bar_data = []
        coord_to_label = {}
        node_id = 1

        def add_node(x, y):
            nonlocal node_id
            key = (round(x, 6), round(y, 6))
            if key in coord_to_label:
                return coord_to_label[key]
            label = f"n{node_id}"
            self.nodes.append((label, x, y))
            self.node_dict[label] = (x, y)
            coord_to_label[key] = label
            node_id += 1
            return label

        # === GÉNÉRATION DES NŒUDS DE LA MOITIÉ GAUCHE ===

        # Nœuds de la membrure inférieure (entraits brisés)
        base_left = []
        for i in range(m // 2 + 1):
            x = i * espacement
            y = (x / L) * tan(beta_rad) * L
            base_left.append(add_node(x, y))

        # Nœuds de la membrure supérieure (arbalétriers)
        top_left = []
        for i in range(1, m // 2 + 1):
            x = (i - 0.5) * espacement
            y = (x / L) * H
            top_left.append(add_node(x, y))

        # Nœud au faîtage
        label_faitage = add_node(L, H)

        # === GÉNÉRATION DES BARRES DE LA MOITIÉ GAUCHE ===

        bars_left = []

        # Entraits (membrure inférieure)
        for i in range(len(base_left) - 1):
            bars_left.append((base_left[i], base_left[i + 1]))

        # Arbalétriers (membrure supérieure)
        # Premier arbalétrier (du nœud base[0] au premier nœud supérieur)
        bars_left.append((base_left[0], top_left[0]))

        # Arbalétriers intermédiaires
        for i in range(len(top_left) - 1):
            bars_left.append((top_left[i], top_left[i + 1]))

        # Dernier arbalétrier (du dernier nœud supérieur vers le faîtage)
        bars_left.append((top_left[-1], label_faitage))

        # Diagonales (pattern Warren)
        for i in range(len(top_left)):
            # Diagonale descendante
            if i + 1 < len(base_left):
                bars_left.append((top_left[i], base_left[i + 1]))

            # Diagonale montante
            if i < len(top_left) - 1 and i + 1 < len(base_left):
                bars_left.append((base_left[i + 1], top_left[i + 1]))

        # === GÉNÉRATION DE LA STRUCTURE COMPLÈTE PAR SYMÉTRIE ===

        # Création de la carte de symétrie
        symmetry_map = {}
        for label, (x, y) in list(self.node_dict.items()):
            if abs(x - L) < 1e-6:  # Nœud au faîtage
                symmetry_map[label] = label
                continue
            x_sym = 2 * L - x
            label_sym = add_node(x_sym, y)
            symmetry_map[label] = label_sym

        # Génération des barres de droite par symétrie
        # IMPORTANT: Les nœuds avec x = L ne sont PAS concernés par la symétrie
        bars_right = []
        for a, b in bars_left:
            # Vérification si les nœuds touchent le faîtage (x = L)
            x_a, y_a = self.node_dict[a]
            x_b, y_b = self.node_dict[b]

            if abs(x_a - L) < 1e-6 and abs(x_b - L) < 1e-6:
                # Les deux nœuds sont au faîtage (x = L), cette barre ne sera PAS dupliquée
                # car elle concerne uniquement des nœuds non concernés par la symétrie
                continue
            elif abs(x_a - L) < 1e-6:
                # Le nœud a est au faîtage (x = L), on connecte a au symétrique de b
                # a reste le même car x = L n'est pas concerné par la symétrie
                b_sym = symmetry_map[b]
                bars_right.append((a, b_sym))
            elif abs(x_b - L) < 1e-6:
                # Le nœud b est au faîtage (x = L), on connecte le symétrique de a à b
                # b reste le même car x = L n'est pas concerné par la symétrie
                a_sym = symmetry_map[a]
                bars_right.append((a_sym, b))
            else:
                # Aucun nœud n'est au faîtage, symétrie complète normale
                a_sym = symmetry_map[a]
                b_sym = symmetry_map[b]
                bars_right.append((a_sym, b_sym))

        # === AJOUT DU POINÇON ===
        # Le poinçon relie les nœuds ayant x = L et n'est PAS concerné par la symétrie
        # car il est unique et situé au centre de la structure

        # Recherche des deux nœuds ayant pour x = L
        noeuds_poinceon = []
        for label, (x, y) in self.node_dict.items():
            if abs(x - L) < 1e-6:  # Comparaison avec tolérance numérique
                noeuds_poinceon.append((label, y))

        # Tri des nœuds par hauteur (y)
        noeuds_poinceon.sort(key=lambda item: item[1])

        # Création de la barre poinçon si on a exactement 2 nœuds
        # Cette barre n'est PAS dupliquée car elle concerne uniquement des nœuds x = L
        barre_poinceon = []
        if len(noeuds_poinceon) == 2:
            noeud_bas = noeuds_poinceon[0][0]  # Nœud le plus bas
            noeud_haut = noeuds_poinceon[1][0]  # Nœud le plus haut
            barre_poinceon.append((noeud_bas, noeud_haut))

        # === ASSEMBLAGE FINAL ===

        all_bars = bars_left + bars_right + barre_poinceon
        self.bars = all_bars
        self.bar_data = [(f"B{i+1}", a, b) for i, (a, b) in enumerate(all_bars)]

        return self.nodes, self.bars

    def generate_structure(self, L, H, m):
        if self.geometry_type == "howe_brise":
            self.generate_structure_howe_brise(L, H, m, self.alpha_deg)
        elif self.geometry_type == "pratt_brise":
            self.generate_structure_pratt_brise(L, H, m, self.alpha_deg)
        elif self.geometry_type == "warren_brise_poincon":
            self.generate_structure_warren_brise_poincon(L, H, m, self.alpha_deg)
        else:
            raise ValueError(f"❌ Géométrie inconnue : {self.geometry_type}.")

    def apply_loads(self, F):
        """Applique les charges selon le type de géométrie choisie"""

        self.forces = {label: [0.0, 0.0] for label, _, _ in self.nodes}

        if self.geometry_type == "warren_brise_poincon":
            eps = 1e-6
            m = self.m
            L = self.L
            e = L / m  # ✅ Espacement correct entre piquets (demi-portée)

            for label, x, y in self.nodes:
                # Appui gauche (x = 0, y = 0) ou appui droit (x = 2L, y = 0)
                if (abs(x) < eps and abs(y) < eps) or (abs(x - 2 * L) < eps and abs(y) < eps):
                    self.forces[label][1] = -0.5 * F

                # Poinçon (sommet central à x = L, y = H)
                elif abs(x - L) < eps and abs(y - self.H) < eps:
                    self.forces[label][1] = -0.5 * F

                # Charges sur les piquets impairs de la membrure supérieure
                else:
                    i_float = x / e
                    i_round = round(i_float)
                    if abs(i_float - i_round) < eps and i_round % 2 == 1 and y > 0:
                        self.forces[label][1] = -1.0 * F

        else:
            # Méthode classique : charges sur montants verticaux
            montant_nodes = set()

            for bar_id, n1, n2 in self.bar_data:
                x1, y1 = self.node_dict[n1]
                x2, y2 = self.node_dict[n2]

                if abs(x1 - x2) < 1e-6 and abs(y1 - y2) > 1e-3:
                    haut = n1 if y1 > y2 else n2
                    montant_nodes.add(haut)

            for node in montant_nodes:
                self.forces[node][1] = -F

            # Charges sur appuis
            for label, x, y in self.nodes:
                if abs(y) < 1e-6 and (abs(x) < 1e-6 or abs(x - 2 * self.L) < 1e-6):
                    self.forces[label][1] = -F / 2

    def get_bar_type(self, node1, node2):
        x1, y1 = self.node_dict[node1]
        x2, y2 = self.node_dict[node2]

        if abs(y1) < 1e-6 and abs(y2) < 1e-6:
            return 'entrait'
        elif y1 > 0 and y2 > 0:
            return 'arbaletrier'
        elif (y1 > 0 and abs(y2) < 1e-6) or (abs(y1) < 1e-6 and y2 > 0):
            # montant ou diagonale : distinction possible par angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            return 'montant' if np.isclose(angle, np.pi/2, atol=0.1) else 'diagonale'
        else:
            return 'diagonale'

    def calculate_bar_properties(self, node1, node2):
        """Calcul des propriétés d'une barre"""
        x1, y1 = self.node_dict[node1]
        x2, y2 = self.node_dict[node2]

        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        cos_theta = (x2 - x1) / L
        sin_theta = (y2 - y1) / L

        return L, cos_theta, sin_theta

    def local_stiffness_matrix(self, L, cos_theta, sin_theta, bar_type):
        """Matrice de rigidité locale d'une barre selon son type"""
        A = self.sections.get(bar_type, self.A)  # fallback si inconnu
        c, s = cos_theta, sin_theta
        k = self.E * A / L

        K_local = k * np.array([
            [c*c,   c*s,  -c*c,  -c*s],
            [c*s,   s*s,  -c*s,  -s*s],
            [-c*c, -c*s,   c*c,   c*s],
            [-c*s, -s*s,   c*s,   s*s]
        ])

        return K_local

    def assemble_global_matrix(self):
        """Assemblage de la matrice globale"""
        n_nodes = len(self.nodes)
        n_dof = 2 * n_nodes
        K_global = np.zeros((n_dof, n_dof))

        # Mapping des nœuds aux indices
        node_to_index = {label: i for i, (label, _, _) in enumerate(self.nodes)}

        # Stocker les matrices de rigidité de chaque barre
        self.bar_stiffness_matrices = {}

        for bar_id, node1, node2 in self.bar_data:
            L, cos_theta, sin_theta = self.calculate_bar_properties(node1, node2)
            bar_type = self.get_bar_type(node1, node2)
            K_local = self.local_stiffness_matrix(L, cos_theta, sin_theta, bar_type)

            # Stocker la matrice locale pour le rapport
            self.bar_stiffness_matrices[bar_id] = {
                'matrix': K_local,
                'length': L,
                'cos_theta': cos_theta,
                'sin_theta': sin_theta,
                'nodes': (node1, node2),
                'type': bar_type
            }

            # Indices globaux
            i1, i2 = node_to_index[node1], node_to_index[node2]
            dof_indices = [2*i1, 2*i1+1, 2*i2, 2*i2+1]

            # Assemblage
            for i, global_i in enumerate(dof_indices):
                for j, global_j in enumerate(dof_indices):
                    K_global[global_i, global_j] += K_local[i, j]

        self.K_global = K_global
        return K_global, node_to_index

    def apply_boundary_conditions(self, K_global, F_global, node_to_index, L):
        """Application des conditions aux limites"""
        # Appui simple en (0, 0) - blocage vertical
        # Appui double en (2L, 0) - blocage horizontal et vertical

        # Trouver les nœuds aux appuis
        left_support = None
        right_support = None

        for label, x, y in self.nodes:
            if abs(x - 0) < 1e-6 and abs(y - 0) < 1e-6:
                left_support = label
            elif abs(x - 2*L) < 1e-6 and abs(y - 0) < 1e-6:
                right_support = label

        # DOF à éliminer
        constrained_dofs = []
        if left_support:
            idx = node_to_index[left_support]
            constrained_dofs.append(2*idx + 1)  # Blocage Y

        if right_support:
            idx = node_to_index[right_support]
            constrained_dofs.extend([2*idx, 2*idx + 1])  # Blocage X et Y

        # Élimination des DOF contraints
        free_dofs = [i for i in range(len(F_global)) if i not in constrained_dofs]

        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = F_global[free_dofs]

        return K_reduced, F_reduced, free_dofs, constrained_dofs, left_support, right_support

    def solve_system(self, L, F):
        """Résolution du système"""
        # Assemblage
        K_global, node_to_index = self.assemble_global_matrix()

        # Vecteur des forces
        F_global = np.zeros(2 * len(self.nodes))
        for i, (label, _, _) in enumerate(self.nodes):
            F_global[2*i] = self.forces[label][0]      # Fx
            F_global[2*i + 1] = self.forces[label][1]  # Fy

        # Conditions aux limites
        K_reduced, F_reduced, free_dofs, constrained_dofs, left_support, right_support = \
            self.apply_boundary_conditions(K_global, F_global, node_to_index, L)

        # Résolution
        U_reduced = np.linalg.solve(K_reduced, F_reduced)
        rank = np.linalg.matrix_rank(K_reduced)
        if rank < K_reduced.shape[0]:
            raise ValueError(f"❌ La matrice K est singulière (rang {rank} < taille {K_reduced.shape[0]})")

        # Reconstruction du vecteur complet
        U_global = np.zeros(2 * len(self.nodes))
        U_global[free_dofs] = U_reduced

        # Stockage des déplacements
        self.displacements = {}
        for i, (label, _, _) in enumerate(self.nodes):
            self.displacements[label] = [U_global[2*i], U_global[2*i + 1]]

        # Calcul des réactions
        R_global = K_global @ U_global - F_global
        self.reactions = {}

        if left_support:
            idx = node_to_index[left_support]
            self.reactions[left_support] = [R_global[2*idx], R_global[2*idx + 1]]

        if right_support:
            idx = node_to_index[right_support]
            self.reactions[right_support] = [R_global[2*idx], R_global[2*idx + 1]]

    def calculate_internal_forces(self):
        """Calcul des efforts internes dans les barres"""
        self.internal_forces = {}
        self.strains = {}
        self.angular_variations = {}

        for bar_id, node1, node2 in self.bar_data:
            L, cos_theta, sin_theta = self.calculate_bar_properties(node1, node2)

            # Déplacements des nœuds
            u1, v1 = self.displacements[node1]
            u2, v2 = self.displacements[node2]

            # Déformation unitaire (strain)
            delta_u = u2 - u1
            delta_v = v2 - v1
            axial_strain = (cos_theta * delta_u + sin_theta * delta_v) / L

            # Effort normal
            axial_force = self.E * self.A * axial_strain

            # Variation angulaire (rotation relative)
            transverse_disp = (-sin_theta * delta_u + cos_theta * delta_v)
            angular_variation = transverse_disp / L  # radians

            self.internal_forces[bar_id] = axial_force
            self.strains[bar_id] = axial_strain
            self.angular_variations[bar_id] = angular_variation

    def draw_supports(self, ax, L):
        """Dessiner les appuis"""
        for label, x, y in self.nodes:
            if abs(y) < 1e-6:  # Nœuds sur la base
                if abs(x) < 1e-6:  # Appui simple gauche
                    triangle = Polygon([(x-0.3, y-0.3), (x+0.3, y-0.3), (x, y)],
                                     facecolor='green', edgecolor='black')
                    ax.add_patch(triangle)
                elif abs(x - 2*L) < 1e-6:  # Appui double droit
                    triangle = Polygon([(x-0.3, y-0.3), (x+0.3, y-0.3), (x, y)],
                                     facecolor='orange', edgecolor='black')
                    ax.add_patch(triangle)
                    # Carré pour appui double
                    square = Rectangle((x-0.2, y-0.5), 0.4, 0.2,
                                     facecolor='orange', edgecolor='black')
                    ax.add_patch(square)

    def plot_structure_initial(self):
        """Tracé de la structure initiale avec charges dans des boîtes jaunes et zone graphique élargie"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Tracer les barres
        for bar_id, node1, node2 in self.bar_data:
            x1, y1 = self.node_dict[node1]
            x2, y2 = self.node_dict[node2]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

            # Numérotation des barres au centre (sans boîte)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, bar_id, fontsize=9, color='red',
                    ha='center', va='center')

        # Tracer les nœuds
        for label, x, y in self.nodes:
            ax.plot(x, y, 'ko', markersize=6)
            ax.text(x + 0.1, y + 0.1, label, fontsize=10, fontweight='bold', color='black')

            # Affichage des charges dans une boîte jaune
            fy = self.forces.get(label, [0, 0])[1]
            if abs(fy) > 1e-3:
                ax.text(x, y + 0.4, f'{-fy:.1f} N',
                        fontsize=9, color='black',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                        ha='center')

        # Tracer les appuis
        self.draw_supports(ax, self.L)

        # Ajuster les limites pour laisser de la place aux boîtes de charge
        x_vals = [x for _, x, _ in self.nodes]
        y_vals = [y for _, _, y in self.nodes]
        ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
        ax.set_ylim(min(y_vals) - 1, max(y_vals) + 2)

        # Mise en forme
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title('Charpente triangulée - Nœuds et Barres numérotées', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_deformed_structure(self, scale_factor=1000):
        """Tracé de la structure déformée"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Structure initiale (en gris)
        for bar_id, node1, node2 in self.bar_data:
            x1, y1 = self.node_dict[node1]
            x2, y2 = self.node_dict[node2]
            ax.plot([x1, x2], [y1, y2], 'lightgray', linewidth=2, alpha=0.5, label='Initial' if bar_id == 'B1' else "")

        # Structure déformée
        deformed_nodes = {}
        for label, x, y in self.nodes:
            u, v = self.displacements[label]
            deformed_nodes[label] = (x + scale_factor * u, y + scale_factor * v)
            ax.plot(x + scale_factor * u, y + scale_factor * v, 'ro', markersize=6)

        for bar_id, node1, node2 in self.bar_data:
            x1, y1 = deformed_nodes[node1]
            x2, y2 = deformed_nodes[node2]
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Déformée' if bar_id == 'B1' else "")

        # Appuis sur position initiale
        self.draw_supports(ax, self.L)

        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title(f'Structure Déformée (Amplification ×{scale_factor})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        return fig

    def plot_stress_colored(self):
        """Tracé coloré par sollicitation"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Nœuds
        for label, x, y in self.nodes:
            ax.plot(x, y, 'ko', markersize=8)

        # Barres colorées
        max_force = max(abs(f) for f in self.internal_forces.values()) if self.internal_forces else 1

        for bar_id, node1, node2 in self.bar_data:
            x1, y1 = self.node_dict[node1]
            x2, y2 = self.node_dict[node2]

            force = self.internal_forces[bar_id]
            color = 'red' if force > 0 else 'blue'
            linewidth = 2 + 6 * abs(force) / max_force

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)

        # Appuis
        self.draw_supports(ax, self.L)

        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title('Sollicitations (Rouge: Traction, Bleu: Compression)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Légende
        ax.text(0.02, 0.98, 'Épaisseur proportionnelle à l\'effort\nRouge: Traction | Bleu: Compression',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        return fig

    def plot_bar_diagrams(self):
        """Tracé des diagrammes pour chaque barre"""
        # Organiser les barres par numéro
        bar_numbers = [int(bar_id[1:]) for bar_id, _, _ in self.bar_data]
        sorted_indices = np.argsort(bar_numbers)

        bar_ids_sorted = [self.bar_data[i][0] for i in sorted_indices]
        forces_sorted = [self.internal_forces[self.bar_data[i][0]] for i in sorted_indices]
        strains_sorted = [self.strains[self.bar_data[i][0]] for i in sorted_indices]
        angular_sorted = [self.angular_variations[self.bar_data[i][0]] for i in sorted_indices]

        # Créer trois figures séparées
        figs = []

        # Figure 1: Déformations unitaires
        fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        x_pos = np.arange(len(bar_ids_sorted))
        colors1 = ['red' if s > 0 else 'blue' for s in strains_sorted]
        bars1 = ax1.bar(x_pos, [s * 1e6 for s in strains_sorted], color=colors1, alpha=0.7, edgecolor='black')

        ax1.set_title('Déformations Unitaires par Barre (μm/m)', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Barres', fontsize=12)
        ax1.set_ylabel('Déformation (μm/m)', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bar_ids_sorted, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=1)

        for i, (bar, strain) in enumerate(zip(bars1, strains_sorted)):
            height = strain * 1e6
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, fontweight='bold')

        ax1.text(0.02, 0.98, 'Rouge: Traction | Bleu: Compression',
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        figs.append(fig1)

        # Figure 2: Efforts normaux
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
        colors2 = ['red' if f > 0 else 'blue' for f in forces_sorted]
        bars2 = ax2.bar(x_pos, [f/1000 for f in forces_sorted], color=colors2, alpha=0.7, edgecolor='black')

        ax2.set_title('Efforts Normaux par Barre (kN)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Barres', fontsize=12)
        ax2.set_ylabel('Effort Normal (kN)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bar_ids_sorted, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=1)

        for i, (bar, force) in enumerate(zip(bars2, forces_sorted)):
            height = force / 1000
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, fontweight='bold')

        ax2.text(0.02, 0.98, 'Rouge: Traction | Bleu: Compression',
                transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        figs.append(fig2)

        # Figure 3: Variations angulaires
        fig3, ax3 = plt.subplots(1, 1, figsize=(14, 8))
        colors3 = ['orange' if abs(a) > 1e-6 else 'gray' for a in angular_sorted]
        bars3 = ax3.bar(x_pos, [a * 1000 for a in angular_sorted], color=colors3, alpha=0.7, edgecolor='black')

        ax3.set_title('Variations Angulaires par Barre (mrad)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Barres', fontsize=12)
        ax3.set_ylabel('Variation Angulaire (mrad)', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bar_ids_sorted, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linewidth=1)

        for i, (bar, angular) in enumerate(zip(bars3, angular_sorted)):
            height = angular * 1000
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, fontweight='bold')

        ax3.text(0.02, 0.98, 'Orange: Rotation significative | Gris: Rotation négligeable',
                transform=ax3.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

        plt.tight_layout()
        figs.append(fig3)

        return figs

    def generate_pdf_report(self, filename="rapport_charpente.pdf"):
        """Génération du rapport PDF complet"""
        with PdfPages(filename) as pdf:
            # Page de titre
            fig_title = plt.figure(figsize=(8.5, 11))
            fig_title.text(0.5, 0.8, 'RAPPORT D\'ANALYSE', ha='center', fontsize=24, fontweight='bold')
            fig_title.text(0.5, 0.75, 'CHARPENTE TRIANGULÉE', ha='center', fontsize=20, fontweight='bold')
            fig_title.text(0.5, 0.7, f'Date: {datetime.datetime.now().strftime("%d/%m/%Y")}', ha='center', fontsize=12)
            fig_title.text(0.5, 0.66, f'Projet : {self.nom_projet}', ha='center', fontsize=12)
            fig_title.text(0.5, 0.63, f'Ingénieur : {self.nom_ingenieur}', ha='center', fontsize=12)
            fig_title.text(0.5, 0.60, f'Type de structure : {self.geometry_type.capitalize()}', ha='center', fontsize=12)

            # Données de modélisation
            fig_title.text(0.1, 0.56, 'DONNÉES DE MODÉLISATION:', fontsize=14, fontweight='bold')
            fig_title.text(0.1, 0.52, f'• Demi-portée L = {self.L} m', fontsize=12)
           # fig_title.text(0.1, 0.49, f'• Angle d\'inclinaison θ = {self.theta_deg}°', fontsize=12)
            fig_title.text(0.1, 0.46, f'• Hauteur calculée H = {self.H:.3f} m', fontsize=12)

            fig_title.text(0.1, 0.43, f'• Nombre d\'entraxes m = {self.m}', fontsize=12)
            fig_title.text(0.1, 0.40, f'• Charge nodale F = {self.F} N', fontsize=12)

            # Propriétés des matériaux
            fig_title.text(0.1, 0.37, 'PROPRIÉTÉS DES MATÉRIAUX:', fontsize=14, fontweight='bold')
            fig_title.text(0.1, 0.34, f'• Module d\'Young E = {self.E/1e9:.0f} GPa', fontsize=12)
            fig_title.text(0.1, 0.34, f'• Module d\'Young E = {self.E/1e9:.0f} GPa', fontsize=12)

            # Nouvelle ligne : section + R admissible
            epsilon_max = 0.001
            y_pos = 0.31
            for bar_type, A in self.sections.items():
                R = self.E * A * epsilon_max  # en N
                A_cm2 = A * 1e4
                R_kN = R / 1000
                fig_title.text(0.1, y_pos, f'• {bar_type.capitalize()} : A = {A_cm2:.1f} cm²   |   R ≤ {R_kN:.1f} kN', fontsize=12)
                y_pos -= 0.03

            # Statistiques de la structure
            fig_title.text(0.1, 0.18, 'STATISTIQUES DE LA STRUCTURE:', fontsize=14, fontweight='bold')
            fig_title.text(0.1, 0.14, f'• Nombre de nœuds: {len(self.nodes)}', fontsize=12)
            fig_title.text(0.1, 0.11, f'• Nombre de barres: {len(self.bar_data)}', fontsize=12)
            fig_title.text(0.1, 0.08, f'• Portée totale: {2*self.L} m', fontsize=12)

            plt.axis('off')
            pdf.savefig(fig_title, bbox_inches='tight')
            plt.close(fig_title)

            # Structure initiale
            fig_struct = self.plot_structure_initial()
            pdf.savefig(fig_struct, bbox_inches='tight')
            plt.close(fig_struct)

            # Structure déformée
            fig_deformed = self.plot_deformed_structure()
            pdf.savefig(fig_deformed, bbox_inches='tight')
            plt.close(fig_deformed)

            # Structure colorée par sollicitation
            fig_stress = self.plot_stress_colored()
            pdf.savefig(fig_stress, bbox_inches='tight')
            plt.close(fig_stress)

            # Diagrammes des barres
            bar_figs = self.plot_bar_diagrams()
            for fig in bar_figs:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # Page des nœuds
            self._add_nodes_page(pdf)

            # Page des barres
            self._add_bars_page(pdf)

            # Pages des matrices de rigidité
            self._add_stiffness_matrices_pages(pdf)

            # Page des résultats - déplacements
            self._add_displacements_page(pdf)

            # Page des résultats - sollicitations
            self._add_forces_page(pdf)

            # Page des verification contrainte
            self._add_resistance_check_page(pdf)

            # Page des réactions
            self._add_reactions_page(pdf)

    def _add_nodes_page(self, pdf):
        """Ajouter les pages des nœuds avec pagination"""
        nodes_per_page = 25
        node_index = 0
        total_nodes = len(self.nodes)

        while node_index < total_nodes:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, 'LISTE DES NŒUDS', ha='center', fontsize=16, fontweight='bold')

            y_pos = 0.9
            fig.text(0.1, y_pos, 'ID', fontsize=12, fontweight='bold')
            fig.text(0.3, y_pos, 'X (m)', fontsize=12, fontweight='bold')
            fig.text(0.5, y_pos, 'Y (m)', fontsize=12, fontweight='bold')
            fig.text(0.7, y_pos, 'Description', fontsize=12, fontweight='bold')
            y_pos -= 0.05

            for _ in range(nodes_per_page):
                if node_index >= total_nodes:
                    break

                label, x, y = self.nodes[node_index]

                # Déterminer la description du nœud
                description = ""
                if abs(y) < 1e-6:  # Nœud sur la base
                    if abs(x) < 1e-6:
                        description = "Appui simple (gauche)"
                    elif abs(x - 2*self.L) < 1e-6:
                        description = "Appui double (droit)"
                    else:
                        description = "Nœud de base"
                elif abs(x - self.L) < 1e-6 and abs(y - self.H) < 1e-6:
                    description = "Poinçon (sommet)"
                elif y > 0:
                    description = "Nœud supérieur"
                else:
                    description = "Nœud intermédiaire"

                fig.text(0.1, y_pos, label, fontsize=10)
                fig.text(0.3, y_pos, f'{x:.2f}', fontsize=10)
                fig.text(0.5, y_pos, f'{y:.2f}', fontsize=10)
                fig.text(0.7, y_pos, description, fontsize=10)

                y_pos -= 0.03
                node_index += 1

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _add_bars_page(self, pdf):
        """Ajouter la page des barres avec pagination"""
        bars_per_page = 25
        total_bars = len(self.bar_data)
        bar_index = 0

        while bar_index < total_bars:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, 'LISTE DES BARRES', ha='center', fontsize=16, fontweight='bold')

            y_pos = 0.9
            fig.text(0.05, y_pos, 'ID', fontsize=12, fontweight='bold')
            fig.text(0.15, y_pos, 'Nœud 1', fontsize=12, fontweight='bold')
            fig.text(0.25, y_pos, 'Nœud 2', fontsize=12, fontweight='bold')
            fig.text(0.35, y_pos, 'Longueur (m)', fontsize=12, fontweight='bold')
            fig.text(0.55, y_pos, 'Angle (°)', fontsize=12, fontweight='bold')
            fig.text(0.7, y_pos, 'Type', fontsize=12, fontweight='bold')
            y_pos -= 0.05

            for _ in range(bars_per_page):
                if bar_index >= total_bars:
                    break

                bar_id, node1, node2 = self.bar_data[bar_index]
                L, cos_theta, sin_theta = self.calculate_bar_properties(node1, node2)
                angle = np.degrees(np.arctan2(sin_theta, cos_theta))

                # Type
                bar_type = self.get_bar_type(node1, node2).capitalize()

                fig.text(0.05, y_pos, bar_id, fontsize=10)
                fig.text(0.15, y_pos, node1, fontsize=10)
                fig.text(0.25, y_pos, node2, fontsize=10)
                fig.text(0.35, y_pos, f'{L:.3f}', fontsize=10)
                fig.text(0.55, y_pos, f'{angle:.1f}', fontsize=10)
                fig.text(0.7, y_pos, bar_type, fontsize=10)

                y_pos -= 0.03
                bar_index += 1

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


    def _add_stiffness_matrices_pages(self, pdf):
        """Ajouter les pages des matrices de rigidité"""
        # Matrices locales (quelques exemples)
        bars_to_show = self.bar_data # Montrer toutes les barres

        for i, (bar_id, node1, node2) in enumerate(bars_to_show):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, f'MATRICE DE RIGIDITÉ LOCALE - {bar_id}',
                    ha='center', fontsize=14, fontweight='bold')

            bar_info = self.bar_stiffness_matrices[bar_id]
            K_local = bar_info['matrix']

            # Informations de la barre
            fig.text(0.1, 0.85, f'Nœuds: {node1} → {node2}', fontsize=12)
            fig.text(0.1, 0.82, f'Longueur: {bar_info["length"]:.3f} m', fontsize=12)
            fig.text(0.1, 0.79, f'cos θ: {bar_info["cos_theta"]:.3f}', fontsize=12)
            fig.text(0.1, 0.76, f'sin θ: {bar_info["sin_theta"]:.3f}', fontsize=12)

            # Matrice sous forme de tableau
            y_start = 0.65
            fig.text(0.1, y_start, 'Matrice de rigidité locale [K] (N/m):', fontsize=12, fontweight='bold')

            # En-têtes
            headers = ['u1', 'v1', 'u2', 'v2']
            x_positions = [0.3, 0.4, 0.5, 0.6]

            fig.text(0.2, y_start - 0.05, '', fontsize=10)
            for j, header in enumerate(headers):
                fig.text(x_positions[j], y_start - 0.05, header, fontsize=10, fontweight='bold', ha='center')

            # Valeurs de la matrice
            row_labels = ['u1', 'v1', 'u2', 'v2']
            for i in range(4):
                fig.text(0.2, y_start - 0.08 - i*0.03, row_labels[i], fontsize=10, fontweight='bold')
                for j in range(4):
                    value = K_local[i, j]
                    fig.text(x_positions[j], y_start - 0.08 - i*0.03, f'{value:.0f}',
                            fontsize=9, ha='center')

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Matrice globale
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, 'MATRICE DE RIGIDITÉ GLOBALE', ha='center', fontsize=16, fontweight='bold')

        # Créer une représentation visuelle de la matrice
        ax = fig.add_subplot(111)
        im = ax.imshow(self.K_global, cmap='RdBu', aspect='auto')
        ax.set_title('Matrice K globale (visualisation)', fontsize=12)
        ax.set_xlabel('DOF j')
        ax.set_ylabel('DOF i')
        plt.colorbar(im, ax=ax, shrink=0.8)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_displacements_page(self, pdf):
        """Ajouter les pages des déplacements avec pagination"""
        nodes_per_page = 25
        node_index = 0
        total_nodes = len(self.nodes)
        max_displacement = 0

        while node_index < total_nodes:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, 'DÉPLACEMENTS NODAUX', ha='center', fontsize=16, fontweight='bold')

            y_pos = 0.9
            fig.text(0.1, y_pos, 'Nœud', fontsize=12, fontweight='bold')
            fig.text(0.3, y_pos, 'Ux (mm)', fontsize=12, fontweight='bold')
            fig.text(0.5, y_pos, 'Uy (mm)', fontsize=12, fontweight='bold')
            fig.text(0.7, y_pos, '|U| (mm)', fontsize=12, fontweight='bold')
            y_pos -= 0.05

            for _ in range(nodes_per_page):
                if node_index >= total_nodes:
                    break

                label, x, y = self.nodes[node_index]
                ux, uy = self.displacements[label]
                magnitude = np.sqrt(ux**2 + uy**2)
                max_displacement = max(max_displacement, magnitude)

                fig.text(0.1, y_pos, label, fontsize=10)
                fig.text(0.3, y_pos, f'{ux*1000:.6f}', fontsize=10)
                fig.text(0.5, y_pos, f'{uy*1000:.6f}', fontsize=10)
                fig.text(0.7, y_pos, f'{magnitude*1000:.6f}', fontsize=10)

                y_pos -= 0.03
                node_index += 1

            # Dernière page : affichage déplacement max
            if node_index >= total_nodes:
                y_pos -= 0.05
                fig.text(0.1, y_pos, f'Déplacement maximal: {max_displacement*1000:.6f} mm',
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _add_forces_page(self, pdf):
        """Ajouter les pages des sollicitations"""
        bar_index = 0
        bars_par_page = 25  # Nombre approximatif de barres par page
        total_bars = len(self.bar_data)

        while bar_index < total_bars:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, 'EFFORTS INTERNES DANS LES BARRES', ha='center', fontsize=16, fontweight='bold')

            y_pos = 0.9
            fig.text(0.05, y_pos, 'Barre', fontsize=12, fontweight='bold')
            fig.text(0.2, y_pos, 'Effort N (kN)', fontsize=12, fontweight='bold')
            fig.text(0.4, y_pos, 'Type', fontsize=12, fontweight='bold')
            fig.text(0.55, y_pos, 'Déformation (μm/m)', fontsize=12, fontweight='bold')
            fig.text(0.8, y_pos, 'Rotation (mrad)', fontsize=12, fontweight='bold')

            y_pos -= 0.05

            for i in range(bars_par_page):
                if bar_index >= total_bars:
                    break

                bar_id, node1, node2 = self.bar_data[bar_index]
                force = self.internal_forces[bar_id]
                strain = self.strains[bar_id]
                angular = self.angular_variations[bar_id]

                force_type = "Traction" if force > 0 else "Compression"
                color = 'red' if force > 0 else 'blue'

                fig.text(0.05, y_pos, bar_id, fontsize=10)
                fig.text(0.2, y_pos, f'{force/1000:.6f}', fontsize=10)
                fig.text(0.4, y_pos, force_type, fontsize=10, color=color)
                fig.text(0.55, y_pos, f'{strain*1e6:.3f}', fontsize=10)
                fig.text(0.8, y_pos, f'{angular*1000:.2f}', fontsize=10)

                y_pos -= 0.03
                bar_index += 1

            # Ajout effort max sur dernière page
            if bar_index >= total_bars:
                max_force = max(abs(f) for f in self.internal_forces.values())
                y_pos -= 0.05
                fig.text(0.05, y_pos, f'Effort maximal: {max_force/1000:.6f} kN',
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _add_resistance_check_page(self, pdf):
        """Ajouter des pages de vérification de la tenue à la résistance"""
        bars_per_page = 25
        epsilon_max = 0.001

        # Préparer les données avec taux de travail
        bar_stats = []
        for bar_id, node1, node2 in self.bar_data:
            force = abs(self.internal_forces[bar_id])
            bar_type = self.get_bar_type(node1, node2)
            A = self.sections.get(bar_type, self.A)
            R = self.E * A * epsilon_max
            taux = force / R
            bar_stats.append({
                'id': bar_id,
                'type': bar_type,
                'force': force,
                'R': R,
                'taux': taux
            })

        # Trier par taux décroissant
        bar_stats.sort(key=lambda x: x['taux'], reverse=True)

        # Générer les pages
        page_index = 0
        while page_index < len(bar_stats):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, 'VÉRIFICATION DE LA TENUE À LA RÉSISTANCE ÉLASTIQUE',
                     ha='center', fontsize=16, fontweight='bold')

            y_pos = 0.9
            fig.text(0.05, y_pos, 'Barre', fontsize=12, fontweight='bold')
            fig.text(0.15, y_pos, 'Type', fontsize=12, fontweight='bold')
            fig.text(0.3, y_pos, 'Effort N (kN)', fontsize=12, fontweight='bold')
            fig.text(0.45, y_pos, 'R admissible (kN)', fontsize=12, fontweight='bold')
            fig.text(0.65, y_pos, 'Taux', fontsize=12, fontweight='bold')
            fig.text(0.75, y_pos, 'État', fontsize=12, fontweight='bold')
            y_pos -= 0.05

            for i in range(bars_per_page):
                if page_index >= len(bar_stats):
                    break

                bar = bar_stats[page_index]
                bar_id = bar['id']
                bar_type = bar['type']
                force = bar['force']
                R = bar['R']
                taux = bar['taux']

                # Déterminer état
                if taux < 0.9:
                    etat = "OK"
                    color = 'green'
                elif taux <= 1.0:
                    etat = "Avertissement"
                    color = 'orange'
                else:
                    etat = "Changer"
                    color = 'red'

                fig.text(0.05, y_pos, bar_id, fontsize=10)
                fig.text(0.15, y_pos, bar_type.capitalize(), fontsize=10)
                fig.text(0.3, y_pos, f'{force/1000:.4f}', fontsize=10)
                fig.text(0.45, y_pos, f'{R/1000:.4f}', fontsize=10)
                fig.text(0.65, y_pos, f'{taux:.4f}', fontsize=10)
                fig.text(0.75, y_pos, etat, fontsize=10, color=color,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

                y_pos -= 0.03
                page_index += 1

            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _add_reactions_page(self, pdf):
        """Ajouter la page des réactions"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'RÉACTIONS AUX APPUIS', ha='center', fontsize=16, fontweight='bold')

        y_pos = 0.8
        fig.text(0.1, y_pos, 'Appui', fontsize=12, fontweight='bold')
        fig.text(0.3, y_pos, 'Rx (kN)', fontsize=12, fontweight='bold')
        fig.text(0.5, y_pos, 'Ry (kN)', fontsize=12, fontweight='bold')
        fig.text(0.7, y_pos, 'Type d\'appui', fontsize=12, fontweight='bold')

        y_pos -= 0.05
        total_rx = 0
        total_ry = 0

        for node, reaction in self.reactions.items():
            rx, ry = reaction
            total_rx += rx
            total_ry += ry

            # Déterminer le type d'appui
            x, y = self.node_dict[node]
            if abs(x) < 1e-6:
                appui_type = "Simple (Ry bloqué)"
            else:
                appui_type = "Double (Rx, Ry bloqués)"

            fig.text(0.1, y_pos, node, fontsize=10)
            fig.text(0.3, y_pos, f'{rx/1000:.6f}', fontsize=10)
            fig.text(0.5, y_pos, f'{ry/1000:.6f}', fontsize=10)
            fig.text(0.7, y_pos, appui_type, fontsize=10)
            y_pos -= 0.05

        # Vérification d'équilibre
        y_pos -= 0.05
        fig.text(0.1, y_pos, 'VÉRIFICATION D\'ÉQUILIBRE:', fontsize=12, fontweight='bold')
        y_pos -= 0.03
        fig.text(0.1, y_pos, f'ΣRx = {total_rx/1000:.3f} kN', fontsize=11)
        y_pos -= 0.03
        fig.text(0.1, y_pos, f'ΣRy = {total_ry/1000:.3f} kN', fontsize=11)

        # Charges appliquées totales
        total_applied = sum(self.forces[label][1] for label, _, _ in self.nodes)
        y_pos -= 0.03
        fig.text(0.1, y_pos, f'Charges totales appliquées = {total_applied/1000:.3f} kN', fontsize=11)

        # Vérification
        equilibrium_check = abs(total_ry - abs(total_applied)) < 1e-3
        y_pos -= 0.05
        fig.text(0.1, y_pos, f'Équilibre vertical: {"✓ OK" if equilibrium_check else "✗ ERREUR"}',
                fontsize=7, fontweight='bold',
                color='green' if equilibrium_check else 'red',
                bbox=dict(boxstyle="round,pad=0.3",
                         facecolor='lightgreen' if equilibrium_check else 'lightcoral'))

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def print_results(self):
        """Affichage des résultats dans la console"""
        print("\n" + "="*60)
        print("RÉSULTATS DE L'ANALYSE")
        print("="*60)

        # Déplacements
        print("\nDÉPLACEMENTS NODAUX:")
        print("-" * 50)
        print(f"{'Nœud':<8} {'Ux (mm)':<12} {'Uy (mm)':<12} {'|U| (mm)':<12}")
        print("-" * 50)

        max_displacement = 0
        for label, x, y in self.nodes:
            ux, uy = self.displacements[label]
            magnitude = np.sqrt(ux**2 + uy**2)
            max_displacement = max(max_displacement, magnitude)
            print(f"{label:<8} {ux*1000:>8.3f}    {uy*1000:>8.3f}    {magnitude*1000:>8.3f}")

        print(f"\nDéplacement maximal: {max_displacement*1000:.3f} mm")

        # Efforts internes
        print("\nEFFORTS INTERNES:")
        print("-" * 70)
        print(f"{'Barre':<8} {'Effort (kN)':<12} {'Type':<12} {'Déf. (μm/m)':<15} {'Rot. (mrad)':<12}")
        print("-" * 70)

        max_force = 0
        for bar_id, node1, node2 in self.bar_data:
            force = self.internal_forces[bar_id]
            strain = self.strains[bar_id]
            angular = self.angular_variations[bar_id]
            max_force = max(max_force, abs(force))

            force_type = "Traction" if force > 0 else "Compression"
            print(f"{bar_id:<8} {force/1000:>8.6f}    {force_type:<12} {strain*1e6:>10.3f}     {angular*1000:>8.2f}")

        print(f"\nEffort maximal: {max_force/1000:.2f} kN")

        # Réactions
        print("\nRÉACTIONS AUX APPUIS:")
        print("-" * 40)
        print(f"{'Appui':<8} {'Rx (kN)':<12} {'Ry (kN)':<12}")
        print("-" * 40)

        total_rx = total_ry = 0
        for node, reaction in self.reactions.items():
            rx, ry = reaction
            total_rx += rx
            total_ry += ry
            print(f"{node:<8} {rx/1000:>8.4f}    {ry/1000:>8.4f}")

        print(f"\nΣRx = {total_rx/1000:.3f} kN")
        print(f"ΣRy = {total_ry/1000:.3f} kN")

    def run_analysis(self):
        """Exécution complète de l'analyse"""
        print("  ANALYSEUR DE CHARPENTE TRIANGULÉE")
        print("="*50)

        # Saisie des données
        self.get_input_parameters()  # On utilise directement les attributs de l'objet

        # Saisie de la charge nodale F
        while True:
            try:
                self.F = float(input("Donner la charge nodale F (en N) : "))
                break
            except ValueError:
                print("❌ Valeur invalide pour F.")

        # Génération de la structure
        print("\n Génération de la structure...")
        self.generate_structure(self.L, self.H, self.m)

        # Application des charges
        print("  Application des charges...")
        self.apply_loads(self.F)
        print("Forces verticales appliquées :")
        for label, f in self.forces.items():
            if abs(f[1]) > 1e-5:
                print(f"{label}: Fy = {f[1]:.2f} N")

        # Résolution
        print(" Résolution du système...")
        self.solve_system(self.L, self.F)

        # Calcul des efforts internes
        print(" Calcul des efforts internes...")
        self.calculate_internal_forces()

        # Affichage des résultats
        self.print_results()

        # Génération des graphiques
        print("\n Génération des graphiques...")

        print("Structure initiale...")
        fig1 = self.plot_structure_initial()
        plt.show()

        print("Structure déformée...")
        fig2 = self.plot_deformed_structure()
        plt.show()

        print("Sollicitations...")
        fig3 = self.plot_stress_colored()
        plt.show()

        print("Diagrammes des barres...")
        bar_figs = self.plot_bar_diagrams()
        for i, fig in enumerate(bar_figs):
            titles = ["Déformations unitaires", "Efforts normaux", "Variations angulaires"]
            print(f"{titles[i]}...")
            plt.show()

        # Génération du rapport PDF
        print("\n Génération du rapport PDF...")
        self.generate_pdf_report()
        print(f" Rapport généré: rapport_charpente.pdf")

        print("\n Analyse terminée avec succès!")

# Utilisation
if __name__ == "__main__":
    analyzer = TrussAnalyzer()
    analyzer.run_analysis()