import os


def write_file(filename, epoch, train_result, valid_result):
    with open(filename, 'a') as file:
        file.write(str(epoch) + '\n')
        # train_loss, train_auc, train_ap, valid_loss, valid_auc, valid_ap
        train_loss, train_auc, train_ap = train_result
        valid_loss, valid_auc, valid_ap = valid_result
        file.write('{} {} {} {} {} {}\n'.format(train_loss, train_auc, train_ap, valid_loss, valid_auc, valid_ap))