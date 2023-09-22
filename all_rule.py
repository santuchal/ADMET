from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem import rdqueries


class RULE_CHECK:
    def __init__(self, _mol):
        if isinstance(_mol, str):
            self.mol = Chem.MolFromSmiles(_mol)
        else:
            self.mol = _mol

    def druglikeness_egan(self, verbose=False):
        violations = []

        tpsa = self._tpsa()
        if tpsa > 131.6:
            violations.append(f"PSA {tpsa}")

        logp = self._logp()
        if logp > 5.88:
            violations.append(f"logP {logp}")

        if verbose:
            return violations

        if len(violations) < 1 :
            return "Pass"
        else:
            return "Failed"

        return len(violations) < 1

    def druglikeness_ghose(self, verbose=False):
        violations = []

        logp = self._logp()
        if logp > 5.6 or logp < -0.4:
            violations.append(f"LOGP {logp}")

        molecular_weight = self._molecular_weight()
        if molecular_weight < 160 or molecular_weight > 480:
            violations.append(f"Molecular Mass {molecular_weight}")

        molar_refractivity = self._molar_refractivity()
        if molar_refractivity < 40 or molar_refractivity > 130:
            violations.append(f"MR {molar_refractivity}")

        n_atoms = self._n_atoms()
        if n_atoms < 20 or n_atoms > 70:
            violations.append(f"N Atoms {n_atoms}")

        if verbose:
            return violations

        if len(violations) < 1 :
            return "Pass"
        else:
            return "Failed"

        return len(violations) < 1

    def druglikeness_ghose_pref(self, verbose=False):
        violations = []

        logp = self._logp()
        if logp > 4.1 or logp < 1.3:
            violations.append(f"LOGP {logp}")

        molecular_weight = self._molecular_weight()
        if molecular_weight < 230 or molecular_weight > 390:
            violations.append(f"Molecular Mass {molecular_weight}")

        molar_refractivity = self._molar_refractivity()
        if molar_refractivity < 70 or molar_refractivity > 110:
            violations.append(f"MR {molar_refractivity}")

        n_atoms = self._n_atoms()
        if n_atoms < 30 or n_atoms > 55:
            violations.append(f"N Atoms {n_atoms}")

        if verbose:
            return violations

        if len(violations) < 1 :
            return "Pass"
        else:
            return "Failed"

        return len(violations) < 1

    def druglikeness_lipinski(self, verbose=False):
        violations = []

        h_bond_donors = self._h_bond_donors()
        if h_bond_donors > 5:
            violations.append(f"H Bond Donors {h_bond_donors}>5")

        h_bond_acceptors = self._h_bond_acceptors()
        if h_bond_acceptors > 10:
            violations.append(f"H Bond Acceptors {h_bond_acceptors}>10")

        molecular_weight = self._molecular_weight()
        if molecular_weight > 500:
            violations.append(f"Molecular Weight {molecular_weight}>500")

        logp = self._logp()
        if logp > 5:
            violations.append(f"LOGP {logp}>5")

        if verbose:
            if not violations:
                return "No violations found"
            return violations

        if len(violations) < 1 :
            return "Pass"
        else:
            return "Failed"

        return len(violations) < 1

    def druglikeness_muegge(self, verbose=False):
        violations = []

        molecular_weight = self._molecular_weight()
        if molecular_weight > 600 or molecular_weight < 200:
            violations.append(f"MW {molecular_weight}")

        logp = self._logp()
        if logp > 5 or logp < -2:
            violations.append(f"LOGP {logp}")

        tpsa = self._tpsa()
        if tpsa > 150:
            violations.append(f"TPSA {tpsa}")

        n_rings = self._n_rings()
        if n_rings > 7:
            violations.append(f"N Rings {n_rings}")

        n_carbon = self._n_carbons()
        if n_carbon < 5:
            violations.append(f"N Carbon {n_carbon}")

        n_heteroatoms = self._n_heteroatoms()
        if n_heteroatoms < 2:
            violations.append(f"N Heteroatoms {n_heteroatoms}")

        n_rot_bonds = self._n_rot_bonds()
        if n_rot_bonds > 15:
            violations.append(f"N Rot Bonds {n_rot_bonds}")

        h_bond_acc = self._h_bond_acceptors()
        if h_bond_acc > 10:
            violations.append(f"H Bond Acc {h_bond_acc}")

        h_bond_don = self._h_bond_donors()
        if h_bond_don > 5:
            violations.append(f"H Bond Don {h_bond_don}")

        if verbose:
            return violations

        if len(violations) < 1 :
            return "Pass"
        else:
            return "Failed"

        return len(violations) < 1

    def druglikeness_veber(self, verbose=False):
        violations = []

        tpsa = self._tpsa()
        if tpsa > 140:
            violations.append(f"TPSA {tpsa}")

        n_rot_bonds = self._n_rot_bonds()
        if n_rot_bonds > 10:
            violations.append(f"N Rotatable Bonds {n_rot_bonds}")

        if verbose:
            return violations

        if len(violations) < 1 :
            return "Pass"
        else:
            return "Failed"
        return len(violations) < 1


    def brenk(self):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        catalog = FilterCatalog.FilterCatalog(params)
        return catalog.HasMatch(self.mol)

    def pains(self):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        return catalog.HasMatch(self.mol)

    def _h_bond_donors(self) -> int:
        """Number of Hydrogen Bond Donors"""
        return Chem.Lipinski.NumHDonors(self.mol)

    def _h_bond_acceptors(self) -> int:
        """Number of Hydrogen Bond Acceptors"""
        return Chem.Lipinski.NumHAcceptors(self.mol)

    def _molar_refractivity(self):
        """Wildman-Crippen Molar Refractivity"""
        return Chem.Crippen.MolMR(self.mol)

    def _molecular_weight(self):
        """Molecular weight"""
        return Descriptors.ExactMolWt(self.mol)

    def _n_atoms(self):
        """Number of atoms"""
        return self.mol.GetNumAtoms()

    def _n_carbons(self):
        """Number of carbon atoms"""
        carbon = Chem.rdqueries.AtomNumEqualsQueryAtom(6)
        return len(self.mol.GetAtomsMatchingQuery(carbon))

    def _n_heteroatoms(self):
        """Number of heteroatoms"""
        return Descriptors.rdMolDescriptors.CalcNumHeteroatoms(self.mol)

    def _n_rings(self):
        """Number of rings"""
        return Descriptors.rdMolDescriptors.CalcNumRings(self.mol)

    def _n_rot_bonds(self):
        """Number of rotatable bonds"""
        return Chem.Lipinski.NumRotatableBonds(self.mol)

    def _logp(self):
        """Log of partition coefficient"""
        return Descriptors.MolLogP(self.mol)

    def _tpsa(self):
        """Topological polar surface area"""
        return Descriptors.TPSA(self.mol)