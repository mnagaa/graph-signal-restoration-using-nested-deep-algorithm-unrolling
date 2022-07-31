#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import pdb
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from .common import Eval
from .dataset import set_dataset
from .models import set_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Test:
    def __init__(self, params, save_dir, device, model, test_loader):
        self.params = params
        self.params = params
        self.save_dir = save_dir
        self.device = device
        self.model = model
        self.test_loader = test_loader

        # criterion
        self.eval = Eval()

        model_path = os.path.join(self.save_dir, 'weight_best.pth')
        model_load = torch.load(model_path)  # load trained model parameter
        self.model.load_state_dict(model_load)  # load parameter to model

        self.test_info = {
            'step': []
        }
        print("# [Log]: Successfully prepared train. \n")

    def exec_test(self, x, g):
        pred_x = self.model(g)
        _, rmse_origin = self.eval.root_mean_squared_error(x, g)
        _, rmse_result = self.eval.root_mean_squared_error(x, pred_x)
        return rmse_origin, rmse_result, pred_x

    def main_test(self):
        self.test_time = time.time()

        self.model.eval()  # only evaluate
        for idx, (x, g) in enumerate(self.test_loader):
            rmse_origin, rmse_result, pred_x = self.exec_test(x, g)
            test_result = {
                'idx': idx,
                'rmse_origin': rmse_origin,
                'rmse_result': rmse_result,
            }
            self.test_info['step'].append(test_result)
            print(f"# [Log]: {test_result}")

    def run(self):
        self.model.eval()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        self.main_test()

        df_test = pd.DataFrame(self.test_info['step'])
        fname = os.path.join(self.save_dir, 'test_result.csv')
        df_test.to_csv(fname)
        print(df_test)
        print(f"# [Log]: save to ... {fname}")

class TestInt:
    def __init__(self, params, save_dir, device, model, test_loader):
        self.params = params
        print(params)
        self.learning_type = params['learning_type']
        self.save_dir = save_dir
        self.device = device
        self.model = model
        self.test_loader = test_loader

        # criterion
        self.eval = Eval()

        model_path = os.path.join(self.save_dir, 'weight_best.pth')
        model_load = torch.load(model_path)  # load trained model parameter
        self.model.load_state_dict(model_load)  # load parameter to model

        self.test_info = {
            'step': []
        }
        print("# [Log]: Successfully prepared train. \n")

    def exec_test(self, x, g, H):
        pred_x = self.model(g, H)
        _, rmse_origin = self.eval.root_mean_squared_error(x, g)
        _, rmse_result = self.eval.root_mean_squared_error(x, pred_x)
        return rmse_origin, rmse_result, pred_x

    def main_test(self):
        self.test_time = time.time()

        learning_type = self.learning_type
        self.model.eval()  # only evaluate
        for idx, (x, g, H) in enumerate(self.test_loader):
            rmse_origin, rmse_result, pred_x = self.exec_test(x, g, H)
            test_result = {
                'learning_type': learning_type,
                'idx': idx,
                'rmse_origin': rmse_origin,
                'rmse_result': rmse_result,
            }
            self.test_info['step'].append(test_result)
            print(f"# [Log]: {test_result}")

            #
            if idx < 3:
                # visual
                save_name = f"{self.save_dir}/id_{idx}_origin"
                community_visual(x.squeeze().detach().numpy(), save_name)

                save_name = f"{self.save_dir}/id_{idx}_degrade"
                community_visual(g.squeeze().detach().numpy(), save_name)

                save_name = f"{self.save_dir}/id_{idx}_reconst"
                community_visual(pred_x.squeeze().detach().numpy(), save_name)
                # import pdb; pdb.set_trace()
                pass

    def run(self):
        self.model.eval()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        self.main_test()

        df_test = pd.DataFrame(self.test_info['step'])
        fname = os.path.join(self.save_dir, 'test_result.csv')
        df_test.to_csv(fname)
        print(f"# [Log]: save to ... {fname}")
        # import pdb; pdb.set_trace()
        print(df_test)

        params = {
            'learning_type': self.learning_type,
            'save_dir': self.save_dir,
            'rmse_origin': df_test.mean()['rmse_origin'],
            'rmse_result': df_test.mean()['rmse_result'],
        }

        import json
        fname = f'{self.save_dir}/test.json'
        with open(fname, mode='w') as f:
            json.dump(params, f, indent=4)
        print(f"# [Log]: save json to ... {fname}")





def community_visual(signal, save_name, show=False, pdf=True):
    ''' Visualize signals on community graphA

    Parameters
    ----------
    signal :
    '''
    from pygsp import graphs
    # plt.rcParams['image.cmap'] = 'gnuplot'
    # plt.rcParams['image.cmap'] = 'viridis'
    plt.rcParams['image.cmap'] = 'jet'
    G = graphs.StochasticBlockModel(
            N=250, k=3,
            p=[0.1, 0.2, 0.15], q =0.004,  # 下げるとcommunity 間のスパース
            seed=42
        )
    G.set_coordinates(seed=42)
    G.plot_signal(
        signal, limits=[0, 6], # sigが1-5の値をとるので
        vertex_size=120, plot_name='')
    plt.axis('off')
    print(f"# [VISUAL] => {save_name}")
    if pdf == True:
        plt.savefig(f"{save_name}.pdf", bbox_inches='tight', pad_inches=0.1)
    else:
        plt.savefig(f'{save_name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    if show:
        pass
    else:
        plt.clf()
        plt.close()



class TestPointCloud:
    def __init__(self, params, save_dir):
        self.params = params
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = save_dir
        data_path = os.path.join(params['mode'], config[params['mode']][params['dataset']][params['tag']]['path'])

        self.testset = set_dataset_point(
            partition='test',
            data_path=data_path,
            npoints=params['npoints'],
            person=params['person'],
            suffix=params['suffix']
        )
        self.test_loader = DataLoader(
                self.testset, batch_size=self.params['batch_size'],
                shuffle=False, drop_last=False, num_workers=1)

        fname = os.path.join(self.save_dir, 'run_test.txt')
        fname = os.path.join(self.save_dir, 'run_summary.txt')

        self.model = set_model(params=params, dataset=self.testset, device=self.device)  # define model
        self.model = self.model.float()

        model_path = os.path.join(self.save_dir, 'weight_best.pth')
        model_load = torch.load(model_path)  # load trained model parameter
        self.model.load_state_dict(model_load)  # load parameter to model
        self.loss = Eval(setting='RMSE')
        logger.info("### Successfully prepared test. \n")

    def exec_test(self, x, g, idx):
        list_G = self.testset.getitem2(idx)
        pred_x = self.model(g, list_G)
        _, rmse_origin = self.loss.eval_loss(x, g)
        _, rmse_result = self.loss.eval_loss(x, pred_x)
        return rmse_origin, rmse_result, pred_x

    def main_test(self):
        self.test_time = time.time()
        epoch_test_origin = []
        epoch_test_recons = []
        self.save_pred = {}
        self.model.eval()  # only evaluate
        for i, (x, g, idx) in enumerate(self.test_loader):
            rmse_origin, rmse_result, pred_x = self.exec_test(x, g, idx)
            epoch_test_origin.append(rmse_origin)
            epoch_test_recons.append(rmse_result)
            self.save_pred[i] = pred_x.squeeze().detach().numpy()
            logger.info(f"[Test Epoch: {i}] RMSE origin: {rmse_origin:.5f}, reconst: {rmse_result:.5f}")

        mean_test_rmse_origin = np.array(epoch_test_origin).mean()
        mean_test_rmse_recons = np.array(epoch_test_recons).mean()

        pred_np = np.stack(list(self.save_pred.values())).squeeze()
        sname = os.path.join(self.save_dir, 'test_result.npy')
        logger.info(pred_np.shape)
        np.save(sname, pred_np)

        logger.info(f"[Test  time]: {time.time() - self.test_time:.5f} [sec], {mean_test_rmse_origin:.5f} -> {mean_test_rmse_recons:.5f}")  # write checekpoint
        logger.info(f"{self.params}")
        logger.info(f"  Origin: {mean_test_rmse_origin:.5f}, Reconst: {mean_test_rmse_recons:.5f}")
        self.io_model.cprint(f"(exp_name), {self.params['expname']}, (Origin), {mean_test_rmse_origin:.5f}, (Reconst), {mean_test_rmse_recons:.5f},")

    def run(self):
        self.model.eval()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        self.main_test()


class TestGraphCorrupt:
    def __init__(self, params, save_dir):
        self.params = params
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = save_dir
        data_path = os.path.join(params['mode'], config_graph_corrupt['sensor'][params['signal_type']][params['tag']]['path'])

        self.testset = set_dataset_graph_corrupt(
            partition='test',
            data_path=data_path,
            suffix=params['suffix']
        )
        self.test_loader = DataLoader(
                self.testset, batch_size=self.params['batch_size'],
                shuffle=False, drop_last=False, num_workers=1)

        fname = os.path.join(self.save_dir, 'run_test.txt')
        fname = os.path.join(self.save_dir, 'run_summary.txt')

        self.model = set_model(params=params, dataset=self.testset, device=self.device)  # define model
        self.model = self.model.float()

        model_path = os.path.join(self.save_dir, 'weight_best.pth')
        model_load = torch.load(model_path)  # load trained model parameter
        self.model.load_state_dict(model_load)  # load parameter to model
        self.loss = Eval(setting='RMSE')
        logger.info("### Successfully prepared test. \n")

    def exec_test(self, x, g, idx):
        list_G = self.testset.getitem2(idx)
        pred_x = self.model(g, list_G)
        _, rmse_origin = self.loss.eval_loss(x, g)
        _, rmse_result = self.loss.eval_loss(x, pred_x)
        return rmse_origin, rmse_result, pred_x

    def main_test(self):
        self.test_time = time.time()
        epoch_test_origin = []
        epoch_test_recons = []
        self.save_pred = {}
        self.model.eval()  # only evaluate
        for i, (x, g, idx) in enumerate(self.test_loader):
            rmse_origin, rmse_result, pred_x = self.exec_test(x, g, idx)
            epoch_test_origin.append(rmse_origin)
            epoch_test_recons.append(rmse_result)
            self.save_pred[i] = pred_x.squeeze().detach().numpy()
            logger.info(f"[Test Epoch: {i}] RMSE origin: {rmse_origin:.5f}, reconst: {rmse_result:.5f}")

        mean_test_rmse_origin = np.array(epoch_test_origin).mean()
        mean_test_rmse_recons = np.array(epoch_test_recons).mean()

        pred_np = np.stack(list(self.save_pred.values())).squeeze()
        sname = os.path.join(self.save_dir, 'test_result.npy')
        logger.info(pred_np.shape)
        np.save(sname, pred_np)

        logger.info(f"[Test  time]: {time.time() - self.test_time:.5f} [sec], {mean_test_rmse_origin:.5f} -> {mean_test_rmse_recons:.5f}")  # write checekpoint
        logger.info(f"{self.params}")
        logger.info(f"  Origin: {mean_test_rmse_origin:.5f}, Reconst: {mean_test_rmse_recons:.5f}")
        self.io_model.cprint(f"(exp_name), {self.params['expname']}, (Origin), {mean_test_rmse_origin:.5f}, (Reconst), {mean_test_rmse_recons:.5f},")

    def run(self):
        self.model.eval()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        self.main_test()
