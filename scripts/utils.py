import torch
from torch.nn import functional as F


def Score(SC, th, rate):
    score_count = 0.0
    for sc in SC:
        if rate=='FAR':
            if float(sc)>=float(th):
                score_count+=1
        elif rate=='FRR':
            if float(sc)<float(th):
                score_count+=1

    return round(score_count*100/float(len(SC)),4)


def scoreCosineDistance(emb1, emb2):

    dist = F.cosine_similarity(emb1,emb2, dim=-1, eps=1e-08)
    return dist

def chkptsave(opt,model,optimizer,epoch,step):
    ''' function to save the model and optimizer parameters '''
    if torch.cuda.device_count() > 1:
        checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}
    else:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}

    torch.save(checkpoint,'{}/{}_{}.chkpt'.format(opt.out_dir, opt.model_name,step))

def Accuracy(pred, labels):

    acc = 0.0
    num_pred = pred.size()[0]
    pred = torch.max(pred, 1)[1]
    for idx in range(num_pred):
        if pred[idx].item() == labels[idx].item():
            acc += 1

    return acc/num_pred

def getNumberOfSpeakers(labelsFilePath):

    speakersDict = dict()
    with open(labelsFilePath,'r') as labelsFile:
        for line in labelsFile.readlines():
            speakersDict[line.split()[1]] = 0
    return len(speakersDict)

def getModelName(params):

    model_name = params.model_name

    model_name = model_name + '_{}'.format(params.front_end) + '_{}'.format(params.window_size) + '_{}batchSize'.format(params.batch_size*params.gradientAccumulation) + '_{}lr'.format(params.learning_rate) + '_{}weightDecay'.format(params.weight_decay) + '_{}kernel'.format(params.kernel_size) +'_{}embSize'.format(params.embedding_size) + '_{}s'.format(params.scalingFactor) + '_{}m'.format(params.marginFactor)

    model_name += '_{}'.format(params.pooling_method) + '_{}'.format(params.heads_number)

    return model_name
