from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
import kgcn
import kgcn.data_util
import kgcn.core
import kgcn.layers
from kgcn.data_util import dense_to_sparse
from kgcn.default_model import DefaultModel
from kgcn.gcn import get_default_config
from kgcn.feed_index import construct_feed
import importlib
from lib.calculators import AbstractCalculator

from calc.kgcn_preprocess import create_adjancy_matrix, create_feature_matrix, create_multi_adjancy_matrix

def make_obj(target_list):
    atom_num_limit = 50
    multi = True
    mol_list = []
    adj_list = []
    feature_list = []
    mol_name_list = []
    for i, mol in enumerate(target_list):
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
        if mol.GetNumAtoms() >= atom_num_limit:
            continue
        mol_list.append(mol)
        mol_name_list.append(f"index_{str(i)}")
        if multi:
            adj = create_multi_adjancy_matrix(mol)
            adjs = [dense_to_sparse(a) for a in adj]
            adj_list.append(adjs)
        else:
            adj = create_adjancy_matrix(mol)
            adj_list.append([(dense_to_sparse(adj))])
        feature = create_feature_matrix(mol, atom_num_limit)
        feature = np.array(feature)
        feature_list.append(feature)
    # This dictionary is used as an input of kGCN
    obj = {
        "feature": np.asarray(feature_list),
        "adj": np.asarray(adj_list),
        "max_node_num": atom_num_limit,
        "mol_info": {"obj_list": mol_list, "name_list": mol_name_list},
    }
    return obj


def build_dataset(obj):
    config = {
        "with_feature": True,
        "with_node_embedding": False,
        "normalize_adj_flag": False,
        "split_adj_flag": False,
        "shuffle_data": True,
    }
    all_data, info = kgcn.data_util.build_data(config, obj, prohibit_shuffle=True, verbose=False)
    graph_index_list = []
    for i in range(all_data.num):
        graph_index_list.append([i, i])
    info.graph_index_list = graph_index_list
    return all_data, info


def get_default_config():
    config = {
        "learning_rate": 0.01,
        "batch_size": 1,
        "param": None,
        "retrain": None,
        "save_model_path": "model",
        "epoch": 100,
        "profile": None,
        "patience":100,
        "dropout_rate": 0.,
        "save_interval": 2,
        "embedding_dim": 4,
        "task":"generate"
    }
    return config

class Calculator(AbstractCalculator):
    def __init__(self, config):
        self.sess=tf.Session()
        self.model=None
        self.model_mod_name=None
        self.model_path=None
        if "kgcn_model_py" in config:
            self.model_mod_name=config["kgcn_model_py"]
        if "kgcn_model" in config:
            self.model_path=config["kgcn_model"]

    def model_build(self, info):
        config=get_default_config()
        model = kgcn.core.CoreModel(self.sess, config, info, construct_feed_callback=construct_feed)
        # loading model.py (model_vae.py)
        model.build(importlib.import_module(self.model_mod_name),is_train=False)
        # loading the model parameters
        saver = tf.train.Saver()
        load_model=self.model_path
        saver.restore(self.sess,load_model)
        return model

    def calculate(self, mol):
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        obj = make_obj([mol])
        data,info=build_dataset(obj)
        # model definition
        if self.model is None:
            self.model=self.model_build(info)
        cost,acc,pred_data = self.model.pred_and_eval(data)
        return cost


