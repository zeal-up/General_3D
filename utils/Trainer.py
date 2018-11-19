import torch
import os
import traceback
import numpy as np

class Trainer_cls(object):
    '''
    一个通用的用来训练分类网络的类
    lr_scheduler : will update per epoch
    '''
    def __init__(self, model, loss_function, optimizer,\
                train_loader, device=torch.device('cpu'), 
                checkpoint_name=None, viz=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_function = loss_function
        self.viz = viz

        self.best_train_acc = 0
        self.best_train_loss = 1e5
        self.best_val_acc = 0
        self.best_val_loss = 1e5


    def train(self, nepochs, test_loader=None, loader_fn=None, lr_scheduler=None, \
                scheduler_metric='best_train_loss', bn_scheduler=None, saved_path='checkepoints/'):
        '''
        loader_fn : 可以用来指定如何使用train_loader 和 test_loader, 默认loader[0]=data, loader[1]=target
                batch_data = next(iter(train_loader))
                lambda batch_data : batch_data[0], batch_data[1]
        '''

        if test_loader is not None:
            self.log_interval = int(len(self.train_loader)/100) 
            # evaluation 100 times per epoch
        else:
            self.log_interval = 10
            # log per 10 batch without test
        try:
            os.makedirs(saved_path)
        except OSError:
            pass

        lr = 0
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        try:
            for epoch in range(nepochs):
                for batch, batch_data in enumerate(self.train_loader):
                    self.model.train()
                    if loader_fn is not None:
                        data, target = loader_fn(batch_data)
                    else:
                        data, target = batch_data[0], batch_data[1]
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    # print(data.size(), target.size(), output.size())
                    loss = self.loss_function(output, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if batch % self.log_interval == 0:
                        train_acc.update(self._acc(output.data, target), output.size()[0])
                        train_loss.update(loss.item(), n=target.size()[0])
                        print('Epoch {}: {}/{}  |train_loss:{:.4f}  |train_acc:{:.4F}%'.\
                            format(epoch, batch, len(self.train_loader), loss.item(), train_acc.val))
                        x_axis = round(epoch+batch/len(self.train_loader), 2)
                        self.viz.append_loss(loss.item(), x_axis, win_name='loss_win', id='train_loss')
                        self.viz.append_acc(train_acc.val, x_axis, win_name='acc_win', id='train_acc')

                if test_loader is None: # only saved when test is None
                    if train_loss.avg < self.best_train_loss:
                        self._checkpoint(epoch, train_loss.avg, train_acc.avg, name=os.path.join(saved_path, 'best_model.pth'))
                        self.best_train_loss = train_loss.avg
                else: # valuation per epoch
                    val_acc, val_loss = self.evaluation(test_loader, loader_fn=loader_fn)
                    x_axis = epoch+1
                    # print(epoch, batch, len(train_loader), loss.item, train_acc, val_loss, val_acc)
                    print('Epoch {} finished |train_loss:{:.4f}  |train_acc:{:.2f}%  |val_loss:{:.4f} |val_acc:{:.2f}'.\
                        format(epoch, train_loss.avg, train_acc.avg, val_loss, val_acc))
                    self.viz.append_loss(val_loss, x_axis, win_name='loss_win', id='val_loss')
                    self.viz.append_acc(val_acc, x_axis, win_name='acc_win', id='val_acc')
                    
                    if val_acc > self.best_val_acc :
                        self._checkpoint(epoch, val_loss, val_acc, name=os.path.join(saved_path, 'best_model.pth'))
                        self.best_val_loss = val_loss
                        self.best_val_acc = val_acc
                train_acc.reset()
                train_loss.reset()

                if lr_scheduler is not None:
                    if scheduler_metric is None:
                        lr_scheduler.step()
                    else:
                        assert scheduler_metric in ['best_train_loss', 'best_train_acc', 'best_val_loss', 'best_val_acc'],\
                            'illegal schedular metric' 
                        lr_scheduler.step(getattr(self, scheduler_metric))
                    if lr != self.optimizer.param_groups[0]['lr']:
                        lr = self.optimizer.param_groups[0]['lr']
                        self.viz.append_text('\n lr change to {} in epoch {}'.format(lr, epoch))
                if bn_scheduler is not None:
                    bn_scheduler.step()

                if epoch % 20 == 1:
                    torch.save(self.model, os.path.join(saved_path, 'checkpoint_'+str(epoch)+'.pth'))
        except KeyboardInterrupt:
            print('End train early in epoch:', epoch)
            self.viz.append_text('End train early in epoch: ' + str(epoch), win_name='error')
            torch.save(self.model.state_dict(), os.path.join(saved_path, 'end_early_epoch_'+str(epoch)+'.pth'))
        except Exception as E :
            print('Something error: \n', E)
            self.viz.append_text('Something error: ' + str(E) + '\n', win_name='error')
            traceback.print_exc()
        else:
            print('Train finished, best_train_loss:{:.4f} |best_train_acc:{:.2f}  |best_val_loss:{:.4f}   |best_val_acc:{:.2f}'.format(\
                self.best_train_loss, self.best_train_acc, self.best_val_loss, self.best_val_acc))
            self.viz.append_text(\
                'Train finished, best_train_loss:{:.4f} |best_train_acc:{:.2f}  |best_val_loss:{:.4f}   |best_val_acc:{:.2f}'.format(\
                self.best_train_loss, self.best_train_acc, self.best_val_loss, self.best_val_acc),
                win_name='save_information')

    def evaluation(self, test_loader, loader_fn=None):
        val_acc = AverageMeter()
        val_loss = AverageMeter()
        with torch.no_grad():
            for batch_data in test_loader:
                self.model.eval()
                if loader_fn is not None:
                    data, target = loader_fn(batch_data)
                else:
                    data, target = batch_data[0], batch_data[1]
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss.update(self.loss_function(output, target).item(), n=output.size()[0])
                val_acc.update(self._acc(output.data, target), n=output.size()[0])
        return val_acc.avg, val_loss.avg

    def _acc(self, output, target, k=1):
        '''
        output: torch.tensor 
                N x C
        target: torch.tensor
                N
        return : accuracy%
        '''
        with torch.no_grad():
            pred = torch.topk(output, k, dim=1)[1]
            # print(pred.size(), output.size(), output.size()[0])
            correct = pred.eq(target.view(output.size()[0], 1)).sum().item()
            
            return correct*100.0/output.size()[0]

    def _checkpoint(self, epoch, loss, acc, name='./checkpoints/best_model.pth',):
        torch.save(self.model.state_dict(), name)
        self.viz.append_text('save model to ' + name + \
                            '\n Epoch %d, loss:%.4f, acc:%.2f \n' %(epoch, loss, acc), win_name='save_information')




class Trainer_seg(object):
    '''
    一个通用的用来训练segamentation网络的类
    lr_scheduler : will update per epoch
    '''
    def __init__(self, model, loss_function, optimizer,\
                train_loader, device=torch.device('cpu'), 
                viz=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_function = loss_function
        self.viz = viz

        self.best_train_loss = 1e5
        self.best_val_mIoUs = 0


    def train(self, nepochs, test_loader=None, loader_fn=None, lr_scheduler=None, \
                scheduler_metric='best_train_loss', bn_scheduler=None, saved_path='checkepoints/'):
        '''
        loader_fn : 可以用来指定如何使用train_loader 和 test_loader, 默认loader[0]=data, loader[1]=target
                batch_data = next(iter(train_loader))
                lambda batch_data : batch_data[0], batch_data[1]
        '''

        if test_loader is not None:
            self.log_interval = int(len(self.train_loader)/100) 
            # log 100 times per epoch
        else:
            self.log_interval = 10
            # log per 10 batch without test
        try:
            os.makedirs(saved_path)
        except OSError:
            pass

        lr = 0
        train_loss = AverageMeter()
        try:
            for epoch in range(nepochs):
                for batch, batch_data in enumerate(self.train_loader):
                    self.model.train()
                    if loader_fn is not None:
                        data, target = loader_fn(batch_data)
                    else:
                        data, target = batch_data[0], batch_data[1]
                    if isinstance(data, dict):
                        for key in data.keys():
                            data[key] = data[key].to(self.device)
                        target = target.to(self.device)
                    else:
                        data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.loss_function(output, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss.update(loss.item(), n=target.size()[0])
                    if batch % self.log_interval == 0:  
                        print('Epoch {}: {}/{}  |train_loss:{:.4f}'.\
                            format(epoch, batch, len(self.train_loader), loss.item()))
                        x_axis = round(epoch+batch/len(self.train_loader), 2)
                        self.viz.append_loss(loss.item(), x_axis, win_name='loss_win', id='train_loss')

                if test_loader is None: # only saved when test is None
                    if train_loss.avg < self.best_train_loss:
                        self._checkpoint(epoch, train_loss.avg, 0, name=os.path.join(saved_path, 'best_model.pth'))
                        self.best_train_loss = train_loss.avg
                else: # valuation per epoch
                    mIoUs, mpIoUs, oAcc, cAcc, cat_IoUs  = self.evaluation(test_loader, loader_fn=loader_fn)
                    x_axis = epoch+1
                    print('Epoch {} finished |train_loss:{:.4f}  |  mIoUs:{:.2f}'.\
                        format(epoch, train_loss.avg, mIoUs))
                    self.viz.append_acc(mIoUs, x_axis, win_name='IoUs_win', id='mIoUs')
                    self.viz.append_acc(mpIoUs, x_axis, win_name='IoUs_win', id='mpIoUs')
                    self.viz.append_acc(oAcc, x_axis, win_name='acc_win', id='oAcc')
                    self.viz.append_acc(cAcc, x_axis, win_name='acc_win', id='cAcc')
                    self.viz.append_text(cat_IoUs, win_name='IoUs for each category')
                    
                    if mIoUs > self.best_val_mIoUs :
                        self._checkpoint(epoch, 0, mIoUs, name=os.path.join(saved_path, 'best_model.pth'))
                        self.best_val_mIoUs = mIoUs
                train_loss.reset()

                if lr_scheduler is not None:
                    if scheduler_metric is None:
                        lr_scheduler.step()
                    else:
                        assert scheduler_metric in ['best_train_loss', 'best_val_mIoUs'],\
                            'illegal schedular metric' 
                        lr_scheduler.step(getattr(self, scheduler_metric))
                    if lr != self.optimizer.param_groups[0]['lr']:
                        lr = self.optimizer.param_groups[0]['lr']
                        self.viz.append_text('\n lr change to {} in epoch {}'.format(lr, epoch))
                if bn_scheduler is not None:
                    bn_scheduler.step()

                if epoch % 20 == 1:
                    torch.save(self.model, os.path.join(saved_path, 'checkpoint_'+str(epoch)+'.pth'))

        except KeyboardInterrupt:
            print('End train early in epoch:', epoch)
            self.viz.append_text('End train early in epoch: ' + str(epoch), win_name='error')
            torch.save(self.model.state_dict(), os.path.join(saved_path, 'end_early_epoch_'+str(epoch)+'.pth'))
        except Exception as E :
            print('Something error: \n', E)
            self.viz.append_text('Something error: ' + str(E) + '\n', win_name='error')
            traceback.print_exc()
        else:
            print('Train finished, best_train_loss:{:.4f} | best_val_mIoUs:{:.2f}'.format(\
                self.best_train_loss, self.best_val_mIoUs))
            self.viz.append_text(\
                'Train finished, best_train_loss:{:.4f} |   best_val_mIoUs:{:.2f}'.format(\
                self.best_train_loss, self.best_val_mIoUs),
                win_name='save_information')

    def evaluation(self, test_loader, loader_fn=None):
        # test_loader.dataset.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
		# test_loader.dataset.seg_label_to_classes = {} # {0:Airplane, 1:Airplane, ...49:Table}
        seg_classes = test_loader.dataset.seg_classes
        seg_label_to_classes = test_loader.dataset.seg_label_to_classes
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(len(seg_classes))]
        total_correct_class = [0 for _ in range(len(seg_classes))]
        shape_ious = {cat:[] for cat in seg_classes.keys()}
        with torch.no_grad():
            
            self.model.eval()

            for batch_data in test_loader:
                if loader_fn is not None:
                    data, target = loader_fn(batch_data)
                else:
                    data, target = batch_data[0], batch_data[1]
                if isinstance(data, dict):
                    for key in data.keys():
                        data[key] = data[key].to(self.device)
                else:
                    data = data.to(self.device)
                batch_size = target.size()[0]
                test_output = self.model(data) # B x 50 x npoints

                test_output = test_output.permute(0, 2, 1).contiguous()
                test_output = test_output.data.cpu().numpy()
                target = target.numpy()

                pred_val = np.zeros((batch_size, 2048)).astype(np.int32)
                for j in range(batch_size):
                    cat = seg_label_to_classes[target[j,0]]
                    logits = test_output[j,:,:]
                    pred_val[j,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
                correct = np.sum(pred_val == target)
                total_correct += correct
                total_seen += (batch_size*test_output.shape[1]) # batch_size*num_points

                for l in range(len(seg_classes)):
                    total_seen_class[l] += np.sum(target==l)
                    total_correct_class[l] += (np.sum((pred_val==l) & (target==l)))


                for k in range(batch_size):
                    segp = pred_val[k,:]
                    segl = target[k,:] 
                    cat = seg_label_to_classes[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                            part_ious[l-seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l-seg_classes[cat][0]] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
                    shape_ious[cat].append(np.mean(part_ious)) #每个shape_ious[cat]里记录了对应cat里面所有个体的ious
            all_shape_ious = []
            cat_ious = {}
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                cat_ious[cat] = np.mean(shape_ious[cat])

            mIoUs = np.mean(all_shape_ious)
            mpIoUs = np.mean(list(cat_ious.values()))
            oAcc = total_correct / float(total_seen)
            cAcc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))

        return mIoUs*100, mpIoUs*100, oAcc*100, cAcc*100, str(cat_ious) 
        #mIoUs : average IoU over all shapes
        #mpIoUs : average IoU over all category
        #oAcc : average accuracy over all points
        #cAcc : average accuracy over all parts

    def _checkpoint(self, epoch, loss, mIoUs, name='./checkpoints/best_model.pth',):
        torch.save(self.model.state_dict(), name)
        self.viz.append_text('save model to ' + name + \
                            '\n Epoch %d, loss:%.4f, mIoUs:%.2f \n' %(epoch, loss, mIoUs), win_name='save_information')




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

