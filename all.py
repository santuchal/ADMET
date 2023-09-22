#!/usr/bin/env python
# coding: utf-8
import re
import os
import sys
import pandas as pd
from tqdm import tqdm
import joblib
import streamlit as st
from all_rule import RULE_CHECK 

from rdkit.Chem import MolFromSmiles, MolToSmiles, AllChem, MACCSkeys, RDKFingerprint, Lipinski
from rdkit.Chem.Descriptors import CalcMolDescriptors


class Molecule:
	def __init__(self, smiles: str):

		if not smiles :
			print("Empty smiles are given")
			sys.exit()
		self.smiles = smiles
		self.mol = MolFromSmiles(smiles)

	def descriptor_generator(self):
		return CalcMolDescriptors(self.mol)

	def fingerprint_generator(self, fp_type: str, n_bits: int=2048):
		fingerprint_type = {
		'Morgan': AllChem.GetMorganFingerprintAsBitVect,
		'RDKit': RDKFingerprint,
		'Atom': AllChem.GetHashedAtomPairFingerprintAsBitVect,
		'MACCS': MACCSkeys.GenMACCSKeys,
		'Topological': AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect,
		'ECFP2': AllChem.GetMorganFingerprintAsBitVect,
		'ECFP4': AllChem.GetMorganFingerprintAsBitVect
		}

		if fp_type not in fingerprint_type:
			print("Unknown Fingerprint")
			sys.exit()
		if n_bits < 1:
			print("Number of bits can't be less than 1")
			sys.exit()
		fingerprint_function = fingerprint_type[fp_type]
		if fp_type == 'Morgan':
			fingerprint= fingerprint_function(self.mol, 2, nBits=n_bits)
		elif fp_type == 'ECFP2':
			fingerprint= fingerprint_function(self.mol, 1, nBits=n_bits)
		elif fp_type == 'MACCS':
			fingerprint = fingerprint_function(self.mol)
		else:
			fingerprint = fingerprint_function(self.mol, nBits=n_bits)

		fingerprint_list = fingerprint.ToList()
		fingerprint_dict = {f'{i}': int(bit) for i, bit in enumerate(fingerprint_list)}

		return fingerprint_dict


def convert_to_molecule(smiles: str):
    try:
        return Molecule(smiles)
    except ValueError:
        return None

def convert_to_check_rule(smiles: str):
    try:
        return RULE_CHECK(smiles)
    except ValueError:
        return None	

SMI = st.text_input('Input SMILE', 'O=Cc1ccc(Cl)cc1')
st.write('The input SMILE is', str(SMI))
mol = convert_to_molecule(SMI)
mol2 = convert_to_check_rule(SMI)


descriptors = mol.descriptor_generator()
descriptors_1 = pd.DataFrame([list(descriptors.values())], columns= list(descriptors.keys()))

MACCS_fingerprint = mol.fingerprint_generator('MACCS', 2048)
ECFP4_fingerprint = mol.fingerprint_generator('Morgan', 1024)
ECFP2_fingerprint = mol.fingerprint_generator('ECFP2', 1024)
Morgan_fingerprints_dataframe = mol.fingerprint_generator("Morgan", 2048)

MACCS_fingerprint_1 = pd.DataFrame([list(MACCS_fingerprint.values())], columns= list(MACCS_fingerprint.keys()))
ECFP4_fingerprint_1 = pd.DataFrame([list(ECFP4_fingerprint.values())], columns= list(ECFP4_fingerprint.keys()))
ECFP2_fingerprint_1 = pd.DataFrame([list(ECFP2_fingerprint.values())], columns= list(ECFP2_fingerprint.keys()))
Morgan_fingerprints_dataframe = pd.DataFrame([list(Morgan_fingerprints_dataframe.values())], columns= list(Morgan_fingerprints_dataframe.keys()))

descriptors_without_IPC = descriptors_1.drop(['Ipc'], axis=1)
desc_with_fp = pd.merge(descriptors_1,Morgan_fingerprints_dataframe, how='left', left_index=True, right_index=True)
descriptors_without_IPC2 = descriptors_without_IPC
descriptors_without_IPC2 = descriptors_without_IPC2.reset_index()
descriptors_without_IPC2['index'] = descriptors_without_IPC2.pop('index')



st.markdown(''':rainbow[PHYSICOCHEMICAL]''')
# st.write("PHYSICOCHEMICAL")
# st.write("HeavyAtomCount : ", mol.lipinski())
# st.write("HeavyAtomCount: ", round(descriptors_1['HeavyAtomCount'][0],4))

list_descriptors = ['MolWt','MolLogP','MolMR','qed','HeavyAtomMolWt','ExactMolWt','NumValenceElectrons','NumRadicalElectrons','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','MinAbsPartialCharge','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','TPSA','FractionCSP3','HeavyAtomCount','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','RingCount']
for desc in list_descriptors:
	st.write(desc,':', round(descriptors_1[desc][0],4))

st.markdown(''':rainbow[Different Rule Check]''')
# st.write("Rule Check:")
st.write("Lipinski Rule Violation Check: ", mol2.druglikeness_lipinski())
st.write("Conditions for failure : h_bond_donors > 5, h_bond_acceptors > 10, molecular_weight > 500 and logp > 5 ")
st.write("Egan Rule Violation Check: ", mol2.druglikeness_egan())
st.write("Conditions for failure : tpsa > 131.6 and logp > 5.88 ")
st.write("Ghose Rule Violation Check: ", mol2.druglikeness_ghose())
st.write("Conditions for failure : logp > 5.6 or logp < -0.4 ,molar_refractivity < 40 or molar_refractivity > 130, molecular_weight < 160 or molecular_weight > 480 and n_atoms < 20 or n_atoms > 70")
st.write("Ghose Preferred Rule Violation Check: ", mol2.druglikeness_ghose_pref())
st.write("Conditions for failure : logp > 4.1 or logp < 1.3, molecular_weight < 230 or molecular_weight > 390, molar_refractivity < 70 or molar_refractivity > 110 and n_atoms < 30 or n_atoms > 55 ")
st.write("Muegge Rule Violation Check: ", mol2.druglikeness_muegge())
st.write("Conditions for failure : molecular_weight > 600 or molecular_weight < 200, logp > 5 or logp < -2, tpsa > 150, n_rings > 7, n_carbon < 5, n_heteroatoms < 2, n_rot_bonds > 15, h_bond_acc > 10 and h_bond_don > 5 ")
st.write("Veber Rule Violation Check: ", mol2.druglikeness_veber())
st.write("Conditions for failure : tpsa > 140 and n_rot_bonds > 10")
st.write(f"PAINS filter: {mol2.pains()}")
st.write(f"Brenk filter: {mol2.brenk()}")

# Absorption
print("Absorption loading .....")
# absorption_caco2_rf = joblib.load('model_saved/absorption_caco2_rf_model.joblib')
# absorption_HIA_rf = joblib.load('model_saved/absorption_HIA_rf_model.joblib')
# absorption_Bioavailability_Ma_MACCS_rf = joblib.load('model_saved/absorption_Bioavailability_Ma_MACCS_rf_model.joblib')
# Lipophilicity_AstraZeneca_MACCS_rf = joblib.load('model_saved/Lipophilicity_AstraZeneca_MACCS_rf_model.joblib')
# Pgp_Broccatelli_inhibitor_ECFP4 = joblib.load('model_saved/Pgp_Broccatelli_inhibitor_ECFP4_fingerprint.joblib')
st.markdown(''':rainbow[ABSORPTION]''')
# st.write("ABSORPTION")
st.write("Absorption_Caco2 : ", "Removed due to Github LFS policy")#round(absorption_caco2_rf.predict(descriptors_1)[0],4))
st.write("Absorption_HIA : ", "Removed due to Github LFS policy")# round(absorption_HIA_rf.predict(MACCS_fingerprint_1)[0],4))
st.write("Absorption_Bioavailability : ", "Removed due to Github LFS policy")# round(absorption_Bioavailability_Ma_MACCS_rf.predict(MACCS_fingerprint_1)[0],4))
st.write("Absorption_Lipophilicity : ",  "Removed due to Github LFS policy")#round(Lipophilicity_AstraZeneca_MACCS_rf.predict(MACCS_fingerprint_1)[0],4))
st.write("Absorption_PgpInhibotor : ", "Removed due to Github LFS policy")# round(Pgp_Broccatelli_inhibitor_ECFP4.predict(ECFP4_fingerprint_1)[0],4))

# Distribution
print("Distribution loading .....")
# VDss_Lombardo_rf = joblib.load('model_saved/VDss_Lombardo_rf_model.joblib')
# PPBR_AZ_rf = joblib.load('model_saved/PPBR_AZ_rf_model.joblib')
# BBB_Martins_ECFP2_fingerprint = joblib.load('model_saved/BBB_Martins_ECFP2_fingerprint.joblib')
st.markdown(''':rainbow[DISTRIBUTION]''')
# st.write("DISTRIBUTION")
st.write("VDss : ", "Removed due to Github LFS policy")# round(VDss_Lombardo_rf.predict(descriptors_without_IPC)[0],4))
st.write("PPBR : ", "Removed due to Github LFS policy")# round(PPBR_AZ_rf.predict(descriptors_1)[0],4))
st.write("BBB_Martins : ", "Removed due to Github LFS policy")# round(BBB_Martins_ECFP2_fingerprint.predict(ECFP2_fingerprint_1)[0],4))

# Metabolism
print("Metabolism loading .....")
# CYP1A2_random_forest = joblib.load('model_saved/CYP1A2_random_forest_model.joblib')
# CYP2C9_random_forest = joblib.load('model_saved/CYP2C9_random_forest_model.joblib')
# CYP2C19_random_forest = joblib.load('model_saved/CYP2C19_random_forest_model.joblib')
# CYP2D6_random_forest = joblib.load('model_saved/CYP2D6_random_forest_model.joblib')
# CYP3A4_random_forest = joblib.load('model_saved/CYP3A4_random_forest_model.joblib')
st.markdown(''':rainbow[METABOLISM]''')
# st.write("METABOLISM")
st.write("CYP1A2 : ", "Removed due to Github LFS policy")# round(CYP1A2_random_forest.predict(desc_with_fp)[0],4))
st.write("CYP2C9 : ", "Removed due to Github LFS policy")# round(CYP2C9_random_forest.predict(desc_with_fp)[0],4))
st.write("CYP2C19 : ", "Removed due to Github LFS policy")# round(CYP2C19_random_forest.predict(desc_with_fp)[0],4))
st.write("CYP2D6 : ", "Removed due to Github LFS policy")# round(CYP2D6_random_forest.predict(desc_with_fp)[0],4))
st.write("CYP3A4 : ", "Removed due to Github LFS policy")# round(CYP3A4_random_forest.predict(desc_with_fp)[0],4))

# Excretion
print("Excretion loading .....")
# half_life = joblib.load('model_saved/excretion_Half_Life_Obach_descriptor_RF_model.joblib')
# clearance = joblib.load('model_saved/Clearance_Hepatocyte_AZ_rf_model.joblib')
st.markdown(''':rainbow[EXCREATION]''')
# st.write("EXCREATION")
st.write("T 1/2 half life", "Removed due to Github LFS policy")#round(half_life.predict(descriptors_without_IPC)[0],4))
st.write("Clearance ", "Removed due to Github LFS policy")# round(clearance.predict(descriptors_1)[0],4))

# Toxicity
print("Toxicity loading .....")
# AMES_MACCS = joblib.load('model_saved/AMES_MACCS_rf_model.joblib')
# Carcinogens_Lagunin = joblib.load('model_saved/Carcinogens_Lagunin_MACCS_rf_model.joblib')
# ClinTox_MACCS = joblib.load('model_saved/ClinTox_MACCS_rf_model.joblib')
# DILI_MACCS = joblib.load('model_saved/DILI_MACCS_rf_model.joblib')
# hERG_tox = joblib.load('model_saved/hERG_tox_rf_model.joblib')
# LD50 = joblib.load('model_saved/LD50_Zhu_tox_rf_model.joblib')
# Skin_Reaction = joblib.load('model_saved/Skin_Reaction_MACCS_rf_model.joblib')

## Tox 21
print("Tox 21 loading ...")
# result = []
# tox_21_list = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
# for i, cx in enumerate(tox_21_list):
# 	result.append(joblib.load('model_saved/Tox21/'+cx+'_rf_model_descriptor.joblib'))

st.markdown(''':rainbow[TOXICITY]''')
# st.write("TOXICITY")
st.write("AMES : ", "Removed due to Github LFS policy")# round(AMES_MACCS.predict(MACCS_fingerprint_1)[0],4))
st.write("Carcinogens : ", "Removed due to Github LFS policy")# round(Carcinogens_Lagunin.predict(MACCS_fingerprint_1)[0],4))
st.write("ClinTox : ", "Removed due to Github LFS policy")# round(ClinTox_MACCS.predict(MACCS_fingerprint_1)[0],4))
st.write("DILI : ", "Removed due to Github LFS policy")# round(DILI_MACCS.predict(MACCS_fingerprint_1)[0],4))
st.write("hERG : ", "Removed due to Github LFS policy")# round(hERG_tox.predict(descriptors_1)[0],4))
st.write("LD50 : ", "Removed due to Github LFS policy")# round(LD50.predict(descriptors_1)[0]),4)
st.write("Skin_Reaction : ", "Removed due to Github LFS policy")# round(Skin_Reaction.predict(MACCS_fingerprint_1)[0],4))
st.write("\n\nTox 21")
# descriptors_without_IPC = descriptors_without_IPC.reset_index()
# descriptors_without_IPC['index'] = descriptors_without_IPC.pop('index')
tox_21_list = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
for i, tox_21 in enumerate(tox_21_list):
	st.write("Tox 21 with value is :", "Removed due to Github LFS policy")#round(result[i].predict(descriptors_without_IPC2)[0],4))


