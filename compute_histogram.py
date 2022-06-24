from argparse import ArgumentParser
from rdkit import Chem
from rdkit.Chem import AllChem, PyMol
import os
import numpy as np
from lib.calculators import CalculatorFactory


parser = ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("--output", type=str, default="hist/")
parser.add_argument("--target", type=str, default="all")
args = parser.parse_args()
smi_list=[]
for line in open(args.input):
  if len(line.strip())>0:
    smi_list.append(line)
np.random.shuffle(smi_list)

options=CalculatorFactory.get_options()
all_data={key:[] for key in options.keys()}
for smi in smi_list[:1000]:
    mol=Chem.MolFromSmiles(smi)
    if args.target=="all":
        for key, calc in options.items():
            if calc.enable:
                s=calc.calculate(mol)
                all_data[key].append(s)
    else:
        key=args.target
        calc=options[key]
        if calc.enable:
            s=calc.calculate(mol)
            all_data[key].append(s)

basepath="hist/"
for k,v in all_data.items():
    if len(v)>0:
        freq,bins = np.histogram(v)
        #bin_p = np.percentile(v, q=range(0,101,5))
        #bin_p=np.unique(bin_p)
        filename=basepath+"hist_"+k+".tsv"
        print("[SAVE]",filename)
        with open(filename,"w") as fp:
            for i,f in enumerate(freq):
                if i==0:
                    fp.write("\t"+str(bins[i+1])+"\t"+str(f))
                elif i==len(freq)-1:
                    fp.write(str(bins[i])+"\t"+"\t"+str(f))
                else:
                    fp.write(str(bins[i])+"\t"+str(bins[i+1])+"\t"+str(f))
                fp.write("\n")


