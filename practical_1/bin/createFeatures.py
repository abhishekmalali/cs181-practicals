def elem_counts(df):
    df['len_smiles'] = df['smiles'].str.len()
    elements = ['nH', 'n', 'c', 'c1', 'Si', 'SiH2',\
            '=', '-', 'CC', 'ncc', 'C1', 'C', 'H',\
            'cc', 'ccc', 'cccc', 'cc1','(C1)', '(c1)',\
            '(o1)', '(s1)', 'nc', 'c12', 'c2', 'c1cc',\
            '(cc1)', 'c2C', 'cc3', 'oc', 'ncc', 'C1=C',\
            'C=c', 'C=C', 'ccn', 'c3', '[se]', '=CCC=',\
            'c21', 'c1c', 'cn', 'c4c', 'c3c', 'coc',\
            'ccccc', '[SiH2]C', 'cc4', 'sc', 'cccnc',\
            'cnc', 'scc', 'c1s', 'cc4', 'sc2', '2c2',\
            'c5', 'c6','c2c', '[nH]c', 'cnc4', 'C1=C',\
            'Cc', 'nsnc', 'sccc', 'cocc', '(o2)', '(cn1)']
    for elem in elements:
        col_name = 'count_' + elem
        df[col_name] = df['smiles'].str.count(elem)
    return df

#For creating RDKIT features
from rdkit.Chem import Descriptors
def generate_rdk_features(rdk_str):
    return_dict ={}
    rdk = Chem.MolFromSmiles(str(rdk_str))
    return_dict['TPSA'] = Descriptors.TPSA(rdk)
    return_dict['MolLogP'] = Descriptors.MolLogP(rdk)
    return_dict['RingCount'] = Descriptors.RingCount(rdk)
    return_dict['NumHAcceptors'] = Descriptors.NumHAcceptors(rdk)
    return_dict['NumHDonors'] = Descriptors.NumHDonors(rdk)
    return_dict['NumHeteroAtoms'] = Descriptors.NumHeteroatoms(rdk)
    return_dict['NumValenceElectrons'] = Descriptors.NumValenceElectrons(rdk)
    return_dict['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(rdk)
    return_dict['NOCount'] = Descriptors.NOCount(rdk)
    return_dict['NumSaturatedRings'] = Descriptors.NumSaturatedRings(rdk)
    return return_dict


