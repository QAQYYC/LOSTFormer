from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for loader, data in zip(vali_loader, vali_data):
                if self.args.adj_graph:
                    self.model.adj_mx=data.adj_mx
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    if not isinstance(outputs,dict): outputs = outputs
                    # outputs['loss'] = [
                    #     _loss.detach().cpu()
                    #     for _loss in outputs['loss']
                    # ]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs
                    true = batch_y

                    loss = criterion(pred, true)
                    loss = loss.detach().cpu()

                    total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # ========== 新增：计算并输出模型参数量 ==========
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        params_info = f"Total parameters: {total_params}\nTrainable parameters: {trainable_params}\n"
        print(params_info)
        # 保存参数量信息到文件
        with open(os.path.join(path, "model_params.txt"), 'w') as f:
            f.write(params_info)
        # ===========================================

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for loader, data in zip(train_loader,train_data):
                if self.args.adj_graph:
                    self.model.adj_mx=data.adj_mx
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                        torch.cuda.empty_cache()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading models')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model parameters after loading - Total: {total_params}, Trainable: {trainable_params}")

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        total_time = 0
        test_times = []
        warmup_steps = 10

        self.model.eval()
        with torch.no_grad():
            for data, loader in zip(test_data, test_loader):
                if self.args.adj_graph:
                    self.model.adj_mx=data.adj_mx
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    torch.cuda.synchronize()
                    start_time = time.time()

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time

                    if i >= warmup_steps:
                        test_times.append(elapsed)
                        total_time += elapsed
                    # =====================================

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    if data.scale and self.args.inverse:
                        shape = batch_y.shape
                        outputs = data.inverse_transform(outputs.reshape((shape[0] * shape[1], ) + shape[2:])).reshape(shape)
                        batch_y = data.inverse_transform(batch_y.reshape((shape[0] * shape[1], ) + shape[2:])).reshape(shape)

                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        if data.scale and self.args.inverse:
                            shape = input.shape
                            input = data.inverse_transform(input.reshape((shape[0] * shape[1], ) + shape[2:])).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if test_times:
            avg_time = total_time / len(test_times)
            avg_samples_per_sec = self.args.batch_size / avg_time
            avg_time_per_pred = avg_time / self.args.pred_len

            print(f"\n[推理速度测试]")
            print(f"测试批次数量: {len(test_times)} (已跳过前{warmup_steps}个批次)")
            print(f"平均推理时间: {avg_time:.6f} 秒/批次")
            print(f"平均吞吐量: {avg_samples_per_sec:.2f} 样本/秒")
            print(f"平均预测步长时间: {avg_time_per_pred:.6f} 秒/时间步")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        preds = torch.from_numpy(preds)
        trues = torch.from_numpy(trues)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mse, mae = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        if test_times:
            f.write(f"\nInference speed: {avg_time:.6f} sec/batch, {avg_samples_per_sec:.2f} samples/sec")
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
