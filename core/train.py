import os
import time
import typing as t
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from core.common import Eval
from utils.logger import logger


class TrainingTools:
    def __init__(
        self,
        trainset,
        validset,
        model,
        optimizer_name: t.Literal['SGD', 'Adam'],
        batch_size: int = 1,
        _device: t.Literal['cpu', 'cuda'] = 'cpu',
        param_clamp: bool = True,
    ):
        self.trainset = trainset
        self.validset = validset
        self.model = model.float()
        self.batch_size = batch_size
        self.param_clamp = param_clamp
        self.device = torch.device(_device)
        self.optimizer_name = optimizer_name
        self.optimizer = self.set_optimizer(self.model)
        self.scheduler = self.set_scheduler(self.optimizer, name='StepLR')
        self.eval = Eval()

    def set_optimizer(self, model):
        if self.optimizer_name == 'SGD':  # use SGD
            return optim.SGD(model.parameters(), lr=0.9)
        elif self.optimizer_name == 'Adam':  # use Adam
            return optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

    def set_scheduler(self, optimizer, name: t.Literal['StepLR', 'ExLR', 'CALR']):
        if name == 'StepLR':
            return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)
        elif name == 'ExLR':
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif name == 'CALR':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=0.1)

    def clamp_parameter(self):
        """
        # if the updated parameter becomes negative value,
        # the parameters are replaced by the small positive value.
        """
        if self.param_clamp:
            state_dict = self.model.state_dict()
            for k in state_dict.keys():
                state_dict[k] = torch.clamp(state_dict[k], min=1e-8)
            self.model.load_state_dict(state_dict)


class Train:
    def __init__(self, tools: TrainingTools, save_dir: str):
        self.tools = tools
        self.train_loader = DataLoader(
            tools.trainset, batch_size=tools.batch_size,
            shuffle=True, drop_last=False, num_workers=1)
        self.valid_loader = DataLoader(
            tools.validset, batch_size=tools.batch_size,
            shuffle=False, drop_last=False, num_workers=1)
        self.train_info = defaultdict(list)
        self.step = 0
        self.mean_rmse_origin_valid = None
        self.save_dir = save_dir
        logger.success("# [Log]: Successfully prepared train.")

    def exec_forward_train(self, x, g):
        self.tools.optimizer.zero_grad()
        pred_x = self.tools.model(g)
        _, rmse_origin = self.tools.eval.root_mean_squared_error(x, g)
        mse_result, rmse_result = self.tools.eval.root_mean_squared_error(x, pred_x)
        mse_result.backward()
        return rmse_origin, rmse_result

    def main_train(self, epoch: int, is_valid_each: bool=False):
        self.train_time = time.time()
        self.tools.model.train()  # start training
        _train_result = defaultdict(list)
        for _, (x, g) in enumerate(self.train_loader):
            # passing through model
            rmse_origin, rmse_result = self.exec_forward_train(x, g)
            _train_result['rmse_origin'].append(rmse_origin)
            _train_result['rmse_pred'].append(rmse_result)
            self.step += 1
            if is_valid_each:
                self.main_valid(num=self.step, interval='step')
            epoch_info = {'step': self.step, 'rmse_origin': rmse_origin, 'rmse_result': rmse_result}
            self.train_info['step'].append(epoch_info)
            self.tools.optimizer.step()
            self.tools.clamp_parameter()
            if self.step % 20 == 0: # 20 step ごとに表示する
                logger.info(f"step.{self.step}, RMSE origin: {rmse_origin:.5f}, reconst: {rmse_result:.5f}")

        self.tools.scheduler.step()  # update schduler
        train_info_epoch = {
            'epoch': epoch,
            'mean_rmse_origin': np.mean(_train_result['rmse_origin']),
            'mean_rmse_recons': np.mean(_train_result['rmse_pred'])}
        self.train_info['epoch'].append(train_info_epoch)
        logger.info(f"Epoch: {epoch}, Train time: {time.time() - self.train_time:.5f} [sec], {train_info_epoch}")

    def main_valid(self, num: int, interval: t.Literal['epoch', 'step']='step'):
        self.valid_time = time.time()
        _valid_result = defaultdict(list)
        self.tools.model.eval()  # only evaluate
        for i, (x, g) in enumerate(self.valid_loader):
            pred_x = self.tools.model(g)
            mse_pred, rmse_pred = self.tools.eval.root_mean_squared_error(x, pred_x)
            _valid_result['rmse_pred'].append(rmse_pred)
            if self.mean_rmse_origin_valid is None:
                # compute original mean rmse (compute only once)
                _, rmse_origin = self.tools.eval.root_mean_squared_error(x, g)
                _valid_result['rmse_origin'].append(rmse_origin)

        if self.mean_rmse_origin_valid is None:
            self.mean_rmse_origin_valid = np.mean(_valid_result['rmse_origin'])
        valid_info = {
            interval: num,
            'mean_rmse_origin': self.mean_rmse_origin_valid,
            'mean_rmse_pred': np.mean(_valid_result['rmse_pred'])}
        self.train_info[f'valid_{interval}'].append(valid_info)
        logger.info((
            f"Valid: {interval}={num}, "
            f"RMSE origin: {valid_info['mean_rmse_origin']:.5f}, "
            f"reconst: {valid_info['mean_rmse_pred']:.5f}"
        ))

        if interval == 'epoch':
            save_name = f'{self.save_dir}/weight_last.pth'
            torch.save(self.tools.model.state_dict(), save_name)
            if valid_info['mean_rmse_pred'] <= self.best_valid_loss:
                self.best_valid_loss = valid_info['mean_rmse_pred']
                save_name = f'{self.save_dir}/weight_best.pth'
                torch.save(self.tools.model.state_dict(), save_name)
                logger.info(f'Best valid RMSE: {self.best_valid_loss}, updated')

    # def main_test(self):
    #     self.valid_time = time.time()
    #     epoch_test_origin = []
    #     epoch_test_recons = []
    #     self.model.eval()  # only evaluate
    #     for i, (x, g) in enumerate(self.test_loader):
    #         rmse_origin, rmse = self.exec_forward(x, g, is_train=False)
    #         epoch_test_origin.append(rmse_origin)
    #         epoch_test_recons.append(rmse)

    #     test_info = {
    #         'rmse_origin': np.array(epoch_test_origin).mean(),
    #         'rmse_result': np.array(epoch_test_recons).mean()
    #     }
    #     self.train_info['test'] = [test_info]

    def run(self, epochs: int):
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        self.best_valid_loss = 1e+8
        for epoch in range(epochs):
            logger.info(f"[{epoch+1}/{epochs}]: processing train")
            self.main_train(epoch, is_valid_each=False)
            logger.info(f"[{epoch+1}/{epochs}]: processing valid")
            self.main_valid(num=epoch, interval='epoch')
        # self.main_test()
        self.save_log()

    def run_calc_valid_rmse_each_sample(self, epochs: int):
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        self.best_valid_loss = 1e+8
        for epoch in range(epochs):
            logger.info(f"[{epoch+1}/{epochs}]: processing train")
            self.main_train(epoch, is_valid_each=True)
            logger.info(f"[{epoch+1}/{epochs}]: processing valid")
            self.main_valid(num=epoch, interval='epoch')

        # self.main_test()
        self.save_log()

    def save_log(self):
        def save_df(fname: str, df: pd.DataFrame):
            df.to_csv(fname)
            logger.info(f"# [Log]: save to ... {fname}")

        save_df(fname=f"{self.save_dir}/loss_step.csv",
                df=pd.DataFrame(self.train_info['step']))
        save_df(fname=f"{self.save_dir}/loss_train.csv",
                df=pd.DataFrame(self.train_info['epoch']))
        save_df(fname=f"{self.save_dir}/valid_epoch.csv",
                df=pd.DataFrame(self.train_info['valid_epoch']))
        save_df(fname=f"{self.save_dir}/loss_test.csv",
                df=pd.DataFrame(self.train_info['test']))
        try:
            save_df(fname=f"{self.save_dir}/valid_step.csv",
                    df=pd.DataFrame(self.train_info['valid_step']))
        except:
            ...


class TrainInt(Train):
    '''
    info:
        interpolation
    '''
    def exec_forward(self, x, g, H, is_train=True):
        '''
        x ... original signal
        g ... degrade signal
        pred_x ... reconstract signal
        '''
        if is_train:
            self.opt.zero_grad()
        pred_x = self.model(g, H)
        _, rmse_origin = self.eval.root_mean_squared_error(x, g)
        mse_result, rmse_result = self.eval.root_mean_squared_error(x, pred_x)
        if is_train:
            mse_result.backward()
        return rmse_origin, rmse_result

    def main_train(self, epoch):
        self.train_time = time.time()
        self.model.train()  # start training
        epoch_train_origin = []
        epoch_train_recons = []
        for i, (x, g, H) in enumerate(self.train_loader):
            # passing through model
            rmse_origin, rmse_result = self.exec_forward(x, g, H, is_train=True)
            epoch_info = {'step': self.step, 'rmse_origin': rmse_origin, 'rmse_result': rmse_result}
            self.train_info['step'].append(epoch_info)
            self.opt.step()
            if self.params.clamp:
                self.clamp_parameter()
            if self.step % 20 == 0: # 20 step ごとに表示する
                str_info = f"# [Log]: (step.{self.step}), RMSE origin: {rmse_origin:.5f}, reconst: {rmse_result:.5f}"
                print(str_info)
                if False:  # check gradient
                    print(self.model.print_grad())

        self.scheduler.step()  # update schduler
        train_info_epoch = {
            'epoch': epoch,
            'mean_rmse_origin': np.array(epoch_train_origin).mean(),
            'mean_rmse_recons': np.array(epoch_train_recons).mean(),
        }
        print(f"# [Log]: Epoch:{epoch}, Train time: {time.time() - self.train_time:.5f} [sec], {train_info_epoch}")
        self.train_info['epoch'].append(train_info_epoch)

    def main_valid(self, epoch):
        self.valid_time = time.time()

        epoch_valid_origin = []
        epoch_valid_recons = []
        self.model.eval()  # only evaluate
        for i, (x, g, H) in enumerate(self.valid_loader):
            rmse_origin, rmse = self.exec_forward(x, g, H)
            epoch_valid_origin.append(rmse_origin)
            epoch_valid_recons.append(rmse)

            print(f"# [Log]: Valid Epoch: ({epoch}-{i}), RMSE origin: {rmse_origin:.5f}, reconst: {rmse:.5f}")

        valid_info = {
            'epoch': epoch,
            'rmse_origin': np.array(epoch_valid_origin).mean(),
            'rmse_result': np.array(epoch_valid_recons).mean()
        }
        self.train_info['valid'].append(valid_info)
        print(f"# [Log]: Epoch:{epoch}, Train time: {time.time() - self.train_time:.5f} [sec], {valid_info}")
        save_name = os.path.join(self.save_dir, 'weight_last.pth')
        torch.save(self.model.state_dict(), save_name)

        mean_valid_rmse_recons = valid_info['rmse_result']
        if mean_valid_rmse_recons <= self.best_valid_loss:
            self.best_valid_loss = mean_valid_rmse_recons
            save_name = os.path.join(self.save_dir, f'weight_best.pth')

            torch.save(self.model.state_dict(), save_name)
            print(f'# [Log]: Best valid RMSE: {self.best_valid_loss}, updated')


class TrainGraphCorrupt(Train):
    def __init__(self, params, save_dir):
        self.params = params
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = save_dir

        data_path = os.path.join(
            params['mode'], config_graph_corrupt['sensor'][params['signal_type']][params['tag']]['path'])

        self.trainset = set_dataset_graph_corrupt(
            partition='train',
            data_path=data_path,
            suffix=params['suffix'],
        )
        self.validset = set_dataset_graph_corrupt(
            partition='valid',
            data_path=data_path,
            suffix=params['suffix'],
        )
        self.train_loader = DataLoader(
            self.trainset, batch_size=self.params['batch_size'],
            shuffle=True, drop_last=False, num_workers=1)
        self.valid_loader = DataLoader(
            self.validset, batch_size=self.params['batch_size'],
            shuffle=False, drop_last=False, num_workers=1)

        fname = os.path.join(self.save_dir, 'run_train.txt')

        self.model = set_model(
            params=params,
            dataset=self.trainset,
            device=self.device)  # define model
        self.model = self.model.float()
        self.opt = set_optimizer(
            model=self.model,
            name='Adam',
            params=params)       # define optimizer
        self.scheduler = set_scheduler(
            self.opt, name='StepLR')  # define scheduler
        self.loss = Eval(setting='RMSE')

        self.train_rmse = {}
        self.valid_rmse = {}
        self.step_rmse_dict = {}
        self.save_parameter = {}
        print("### Successfully prepared train. \n")

    def clamp_parameter(self):
        # if the updated parameter becomes negative value,
        # the parameters are replaced by the small positive value.
        state_dict = self.model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = torch.clamp(state_dict[k], min=1e-8)
        self.model.load_state_dict(state_dict)

    def exec_train(self, x, g, idx):
        self.opt.zero_grad()
        list_G = self.trainset.getitem2(idx)
        pred_x = self.model(g, list_G)
        _, rmse_origin = self.loss(x, g)
        mse_result, rmse_result = self.loss(x, pred_x)
        mse_result.backward()
        if self.params['clamp']:
            # If self.params['clamp'] is True, trained_parameter is clipped within the set value.
            # The default of self.params['clamp'] is False.
            self.clamp_parameter()
        return rmse_origin, rmse_result

    def exec_valid(self, x, g, idx):
        list_G = self.trainset.getitem2(idx)
        pred_x = self.model(g, list_G)
        _, rmse_origin = self.loss(x.detach(), g.detach())
        _, rmse_result = self.loss(x.detach(), pred_x.detach())
        return rmse_origin, rmse_result

    def main_train(self, epoch):
        print(f"\n*** Processing {epoch} epoch.")
        print(f"[Train, Epoch]:{epoch}")
        self.train_time = time.time()
        self.model.train()  # start training
        epoch_train_origin = []
        epoch_train_recons = []
        for i, (x, g, idx) in enumerate(self.train_loader):
            rmse_origin, rmse_result = self.exec_train(x, g, idx)
            epoch_train_origin.append(rmse_origin)
            epoch_train_recons.append(rmse_result)
            self.step += 1
            self.step_rmse_dict[self.step] = rmse_result
            grad_norm_sum = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1)
            self.opt.step()
            if self.step % 10 == 0:
                print(
                    f"[Train Epoch: {epoch}-{self.step}] RMSE origin: {rmse_origin:.5f}, reconst: {rmse_result:.5f}")
                sdf = pd.DataFrame(self.model.state_dict()).T
                fname = os.path.join(
                    self.save_dir, f'learned_parameter_{self.step}.csv')
                sdf.to_csv(fname)

        self.scheduler.step()  # update schduler
        print(f"*** Update.schduler: ({self.step}) step \n")
        mean_train_rmse_origin = np.array(epoch_train_origin).mean()
        mean_train_rmse_recons = np.array(epoch_train_recons).mean()
        self.train_rmse[epoch] = mean_train_rmse_recons
        print(
            f"[Train time (Epoch:{epoch})]: {time.time() - self.train_time:.5f} [sec], {mean_train_rmse_origin:.5f} -> {mean_train_rmse_recons:.5f}")

    def main_valid(self, epoch):
        print(f"[Valid, Epoch]:{epoch}")
        self.valid_time = time.time()
        epoch_valid_origin = []
        epoch_valid_recons = []
        self.model.eval()  # only evaluate
        with torch.no_grad():
            for i, (x, g, idx) in enumerate(self.valid_loader):
                rmse_origin, rmse_result = self.exec_valid(x, g, idx)
                epoch_valid_origin.append(rmse_origin)
                epoch_valid_recons.append(rmse_result)
                print(
                    f"[Valid Epoch: {epoch}-{i}] RMSE origin: {rmse_origin:.5f}, reconst: {rmse_result:.5f}")
        # import pdb; pdb.set_trace()
        mean_valid_rmse_origin = np.array(epoch_valid_origin).mean()
        mean_valid_rmse_recons = np.array(epoch_valid_recons).mean()
        self.valid_rmse[epoch] = mean_valid_rmse_recons
        # write checekpoint
        print(
            f"[Valid  time (Epoch:{epoch})]: {time.time() - self.valid_time:.5f} [sec], {mean_valid_rmse_origin:.5f} -> {mean_valid_rmse_recons:.5f}")

        save_name = os.path.join(self.save_dir, 'weight_last.pth')
        torch.save(self.model.state_dict(), save_name)
        if mean_valid_rmse_recons <= self.best_valid_loss:
            self.best_valid_loss = mean_valid_rmse_recons
            save_name = os.path.join(self.save_dir, f'weight_best.pth')
            torch.save(self.model.state_dict(), save_name)
            print(f'Best valid RMSE: {self.best_valid_loss}')

    def run(self):
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        print("save initial parameters")
        save_name = os.path.join(self.save_dir, f'weight_step{self.step}.pth')
        torch.save(self.model.state_dict(), save_name)

        self.best_valid_loss = 1e+8
        for epoch in range(self.params['epochs']):
            self.main_train(epoch)
            self.main_valid(epoch)

        df_step = pd.DataFrame({'step': list(self.step_rmse_dict.values())})
        fname = os.path.join(self.save_dir, 'loss_step.csv')
        df_step.to_csv(fname)

        df_loss = pd.DataFrame({
            'train_loss': list(self.train_rmse.values()),
            'valid_loss': list(self.valid_rmse.values())
        })
        fname = os.path.join(self.save_dir, 'loss.csv')
        df_loss.to_csv(fname)
