from utils99 import *
from models import *
import warnings
import gc
import argparse

def parse_args():
    # 获取参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2021, required=False)
    parser.add_argument("--fold", type=int, default=5, required=False)
    parser.add_argument("--log_dir", type=str, default='LOG/', required=False)
    parser.add_argument("--epoch", type=int, default=50, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--lr", type=float, default=1e-3, required=False)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', required=False)
    parser.add_argument("--optimizer", type=str, default='AdamW', required=False)
    parser.add_argument("--metric", type=str, default='', required=False)

    parser.add_argument("--MIX_UP", type=int, default=0, required=False)
    parser.add_argument("--sample", type=int, default=0, required=False)
    parser.add_argument("--norm", type=int, default=0, required=False)
    parser.add_argument("--transform", type=int, default=0, required=False)

    parser.add_argument("--weight", type=str, default='norm_1_log', required=False)
    parser.add_argument("--loss", type=str, default='WeightedMultilabel', required=False)

    parser.add_argument("--train", type=int, default=1, required=False)
    parser.add_argument("--valid", type=int, default=1, required=False)
    parser.add_argument("--test", type=int, default=1, required=False)
    parser.add_argument("--probs", type=int, default=1, required=False)

    parser.add_argument("--model", type=str, default='se_resnet34_plus', required=False)
    parser.add_argument("--pseudo", type=int, default=0, required=False)
    parser.add_argument("--one", type=int, default=0, required=False)

    return parser.parse_args()

warnings.filterwarnings('ignore')
args = parse_args()

class CONFIG:
    # 参数配置
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = args.seed
    fold = args.fold
    log_dir = args.log_dir
    model_dir = f'model/v{args.version}/'
    num_epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    version = args.version
    scheduler = args.scheduler
    optimizer = args.optimizer

    metric = args.metric

    MIX_UP = args.MIX_UP
    sample = args.sample
    norm = args.norm
    transform = args.transform
    weight = args.weight
    loss = args.loss
    train = args.train
    valid = args.valid
    test = args.test
    probs = args.probs
    model = args.model
    pseudo = args.pseudo
    one = args.one


# 文件夹初始化
if not os.path.isdir(CONFIG.log_dir):
    os.makedirs(CONFIG.log_dir)
if not os.path.isdir(CONFIG.model_dir):
    os.makedirs(CONFIG.model_dir)
if not os.path.isdir('final/'):
    os.makedirs('final/')

# 日志初始化
LOGGER = init_logger(output_dir=CONFIG.log_dir, version=CONFIG.version)
fix_seed(CONFIG.seed)
LOGGER.info(f'seed is ok!')
LOGGER.info(f'version: {CONFIG.version} | seed: {CONFIG.seed} | fold: {CONFIG.fold} | device: {CONFIG.device}',)
LOGGER.info(f'num_epochs: {CONFIG.num_epochs} | batch_size: {CONFIG.batch_size} | lr: {CONFIG.lr}',)
LOGGER.info(f'scheduler: {CONFIG.scheduler} | optimizer: {CONFIG.optimizer} | metric: {CONFIG.metric}',)
LOGGER.info(f'sample: {CONFIG.sample} | norm: {CONFIG.norm} | transform: {CONFIG.transform} | MIX_UP: {CONFIG.MIX_UP}',)
LOGGER.info(f'weight: {CONFIG.weight} | loss: {CONFIG.loss} | model: {CONFIG.model} | pseudo: {CONFIG.pseudo}')

# 加载label数据
train_path = glob.glob('ecg_data/*.csv')
labels = pd.read_csv('label_and_example/train_label_1217.csv')
labels[[f'label_{i}' for i in range(18)]] = labels.label.str.split(',', expand=True)
lab_cols = [i for i in labels.columns if i not in ['id', 'label']]
labels[lab_cols] = labels[lab_cols].astype('int')

def get_model():
    # 获取模型
    if CONFIG.model == 'se_resnet34_plus':
        return se_resnet34_plus()
    elif CONFIG.model == 'se_resnet34':
        return se_resnet34()
    elif CONFIG.model == 'se_resnet34_plus2':
        return se_resnet34_plus2()
    elif CONFIG.model == 'se_resnet34_plus3':
        return se_resnet34_plus3()
    elif CONFIG.model == 'lstm':
        return res_lstm()
    else:
        raise ValueError('no this model', CONFIG.model)

if CONFIG.train:
    # 模型训练
    for fold in range(CONFIG.fold):
        net = get_model()
        trainer = Trainer(net, CONFIG, LOGGER, labels, fold=fold)
        trainer.run()

        del trainer
        del net
        torch.cuda.empty_cache()
        gc.collect()


if CONFIG.one:
    # 一次训练同时生成18个单类最优的模型
    for l in range(18):
        if CONFIG.valid:
            oof = np.zeros((len(labels), 18))
            KF = MultilabelStratifiedKFold(CONFIG.fold, random_state=CONFIG.seed, shuffle=True)

            for fold in range(CONFIG.fold):
                model = get_model().to(CONFIG.device)
                model.load_state_dict(
                    torch.load(f'{CONFIG.model_dir}best_se_model_score_{fold}_{l}.pth',
                               map_location=CONFIG.device)
                )
                trainer = Trainer(model, CONFIG, LOGGER, labels, fold=fold)
                for fold_, (trn_idx, val_idx) in enumerate(KF.split(labels['id'].values, labels[lab_cols].values)):
                    if fold == fold_:
                        pred_tmp = trainer._val_for_oof()
                        oof[val_idx, :] = pred_tmp

                del trainer
                del model
                torch.cuda.empty_cache()
                gc.collect()

            pred_cols = [f'pred_{i}' for i in range(18)]
            for col in pred_cols:
                labels[col] = 1

            labels[pred_cols] = oof
            labels[pred_cols + ['id']].to_csv(f'oof_v{CONFIG.version}_{l}.csv', index=False)

            predictions = np.where(oof > 0.5, 1, 0)
            labels[pred_cols] = predictions
            print(f"ori_score: {f1_score(labels[lab_cols], labels[pred_cols], average='macro')}")

            threshold_list = []
            score_list = []


            def post(true, pred):
                best_score = 0
                best_threshold = 0.5
                for i in range(100, 700):
                    threshold = i / 1000
                    pred_ = np.where(pred > threshold, 1, 0)
                    score = f1_score(true, pred_)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                return best_score, best_threshold


            for i in tqdm(range(18)):
                best_score, best_threshold = post(labels[f'label_{i}'], oof[:, i])
                threshold_list.append(best_threshold)
                score_list.append(best_score)

            print("post_score", np.mean(score_list))
            print(threshold_list)

        if CONFIG.test:
            test_path = [os.path.basename(i)[:-4] for i in train_path if os.path.basename(i)[:-4] not in labels.id.values]
            print(len(test_path))

            predictions = np.zeros((len(test_path), 18))

            for fold in range(CONFIG.fold):
                net = get_model().to(CONFIG.device)
                net.load_state_dict(
                    torch.load(f'{CONFIG.model_dir}best_se_model_score_{fold}_{l}.pth',
                               map_location=CONFIG.device)
                )
                net.eval()
                trainer = Trainer(net, CONFIG, LOGGER, labels, test_path, fold=fold)
                pred = trainer.make_test_stage()
                predictions += pred / CONFIG.fold

                del trainer
                del net
                torch.cuda.empty_cache()
                gc.collect()

            pred_cols = [f'pred_{i}' for i in range(18)]
            test = pd.DataFrame(predictions)
            test.columns = pred_cols
            test['id'] = test_path

            test[pred_cols + ['id']].to_csv(f'final/pred_v{CONFIG.version}_{l}.csv', index=False)

else:
    if CONFIG.valid:
        # 模型验证
        oof = np.zeros((len(labels), 18))
        KF = MultilabelStratifiedKFold(CONFIG.fold, random_state=CONFIG.seed, shuffle=True)

        for fold in range(CONFIG.fold):
            model = get_model().to(CONFIG.device)
            model.load_state_dict(
                torch.load(f'{CONFIG.model_dir}best_se_model_score_{fold}.pth',
                           map_location=CONFIG.device)
            )
            trainer = Trainer(model, CONFIG, LOGGER, labels, fold=fold)
            for fold_, (trn_idx, val_idx) in enumerate(KF.split(labels['id'].values, labels[lab_cols].values)):
                if fold == fold_:
                    pred_tmp = trainer._val_for_oof()
                    oof[val_idx, :] = pred_tmp

            del trainer
            del model
            torch.cuda.empty_cache()
            gc.collect()

        pred_cols = [f'pred_{i}' for i in range(18)]
        for col in pred_cols:
            labels[col] = 1

        labels[pred_cols] = oof
        labels[pred_cols + ['id']].to_csv(f'oof_v{CONFIG.version}.csv', index=False)

        predictions = np.where(oof > 0.5, 1, 0)
        labels[pred_cols] = predictions
        print(f"ori_score: {f1_score(labels[lab_cols], labels[pred_cols], average='macro')}")

        threshold_list = []
        score_list = []


        def post(true, pred):
            best_score = 0
            best_threshold = 0.5
            for i in range(100, 700):
                threshold = i / 1000
                pred_ = np.where(pred > threshold, 1, 0)
                score = f1_score(true, pred_)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            return best_score, best_threshold


        for i in tqdm(range(18)):
            best_score, best_threshold = post(labels[f'label_{i}'], oof[:, i])
            threshold_list.append(best_threshold)
            score_list.append(best_score)

        print("post_score", np.mean(score_list))
        print(threshold_list)

    if CONFIG.test:
        # 模型推理
        test_path = [os.path.basename(i)[:-4] for i in train_path if os.path.basename(i)[:-4] not in labels.id.values]
        print(len(test_path))

        predictions = np.zeros((len(test_path), 18))

        for fold in range(CONFIG.fold):
            net = get_model().to(CONFIG.device)
            net.load_state_dict(
                torch.load(f'{CONFIG.model_dir}best_se_model_score_{fold}.pth',
                           map_location=CONFIG.device)
            )
            net.eval()
            trainer = Trainer(net, CONFIG, LOGGER, labels, test_path, fold=fold)
            pred = trainer.make_test_stage()
            predictions += pred / CONFIG.fold

            del trainer
            del net
            torch.cuda.empty_cache()
            gc.collect()

        pred_cols = [f'pred_{i}' for i in range(18)]
        test = pd.DataFrame(predictions)
        test.columns = pred_cols
        test['id'] = test_path

        test[pred_cols + ['id']].to_csv(f'final/pred_v{CONFIG.version}.csv', index=False)
