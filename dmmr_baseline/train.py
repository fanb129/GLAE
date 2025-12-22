import os
from model import *
import numpy as np
from collections import defaultdict
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


def trainDMMR(data_loader_dict, optimizer_config, cuda, args, iteration, one_subject):
    # data of source subjects, which is used as the training set
    source_loader = data_loader_dict['source_loader']
    print("Number of source subjects: ", len(source_loader), "Number of category: ", args.cls_classes)
    # The pre-training phase
    preTrainModel = DMMRPreTrainingModel(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    print("===================== Pre-training Phase =========")
    for epoch in range(args.epoch_preTraining):
        preTrainModel.train()
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1 # for the gradient reverse layer (GRL)
            batch_dict = defaultdict(list) #Pre-fetch a batch of data for each subject in advance and store them in this dictionary.
            data_dict = defaultdict(list) #Store the data of each subject in the current batch
            label_dict = defaultdict(list) #Store the labels corresponding to the data of each subject in the current batch
            label_data_dict = defaultdict(set)
            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                # Assign a unique ID to each source subject
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()
                #the input of the model
                source_data, source_label = batch_dict[j]
                # Prepare corresponding new batch of each subject, the new batch has same label with current batch.
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                # Store the corresponding new batch of each subject, providing the supervision for different decoders
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data = corres_batch_data.cuda()
                data_set_all += len(source_label)
                optimizer_PreTraining.zero_grad()
                # Call the pretraining model
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                # The loss of the pre-training phase, beta is the balancing hyperparameter
                loss_pretrain = rec_loss + args.beta * sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all)+'\t' +
            'subject: '+str(one_subject+1)+'\t'+
            'epoch: '+str(epoch+1)+'\t'+
            'loss_pretrain:' + str(loss_pretrain.data)+'\t'+
             'rec_loss:' + str(rec_loss.data)+'\t'+
             'sim_loss:' + str(sim_loss.data))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    #Load the ABP module, the encoder from pretrained model and build a new model for the fine-tuning phase
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    print("===================== fine-tuning Phase =========")
    for epoch in range(args.epoch_fineTuning):
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                # Call the fine-tuning model
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        acc = float(count) / data_set_all
        # test the fine-tuned model with the data of unseen target subject
        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
        print("data set amount: "+str(data_set_all)+'\t' +
            'subject: '+str(one_subject+1)+'\t'+
            'epoch: '+str(epoch+1)+'\t'+
            'acc:' + str(acc)+'\t'+
             'test_acc:' + str(acc_DMMR)+'\t'+
             'best_acc:' + str(acc_final))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modelDir = os.path.join(current_dir, "model/" + args.way + "/" + args.index + "/")
    try:
        os.makedirs(modelDir)
    except:
        pass
    # save models
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final


def testDMMR(dataLoader, DMMRTestModel, cuda, batch_size):
    index = 0
    count = 0
    data_set_all = 0
    if cuda:
        DMMRTestModel = DMMRTestModel.cuda()
    DMMRTestModel.eval()
    with torch.no_grad():
        for _, (test_input, label) in enumerate(dataLoader):
            if cuda:
                test_input, label = test_input.cuda(), label.cuda()
            test_input, label = Variable(test_input), Variable(label)
            data_set_all += len(label)
            x_shared_pred = DMMRTestModel(test_input)
            _, pred = torch.max(x_shared_pred, dim=1)
            count += pred.eq(label.squeeze().data.view_as(pred)).sum()
            index += batch_size
    acc = float(count) / data_set_all
    return acc