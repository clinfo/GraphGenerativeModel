from lib.chem_filter import Filter
from rdkit import Chem

class RingSizeFilter(Filter):
    def check(mol):
        Chem.SanitizeMol(mol)
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        return max_ring_size <= 6

#        min_ring_size = min((len(r) for r in ri.AtomRings()), default=100)
#        return max_ring_size <= 6 and min_ring_size >= 3
