python get_data_sd.py
python train99.py --version=8 --batch_size=128 --lr=1e-3 --model=se_resnet34 --sample=1 --weight=log_base --norm=1 --train=0 --val=0 
python train99.py --version=13 --batch_size=32 --lr=1e-4 --model=se_resnet34 --sample=0 --weight=log_base --norm=0 --train=0 --val=0
python train99.py --version=20 --batch_size=128 --lr=1e-3 --model=se_resnet34 --sample=1 --weight=norm_1_log --norm=1 --train=0 --val=0
python train99.py --version=21 --batch_size=32 --lr=1e-3 --model=se_resnet34 --sample=1 --weight=norm_1_log --norm=1 --train=0 --val=0
python train99.py --version=22 --batch_size=32 --lr=1e-3 --model=se_resnet34 --sample=0 --weight=norm_1_log --norm=0 --train=0 --val=0
python train99.py --version=23 --batch_size=32 --lr=1e-3 --model=se_resnet34 --sample=0 --weight=norm_1_log --norm=0 --MIX_UP=1 --train=0 --val=0
python train99.py --version=27 --batch_size=32 --lr=1e-3 --model=se_resnet34_plus --sample=0 --weight=norm_1_log --loss=MultiLabelCircleLoss --norm=0 --train=0 --val=0
python train99.py --version=28 --batch_size=32 --lr=1e-3 --model=se_resnet34_plus2 --sample=0 --weight=norm_1_log --norm=0 --train=0 --val=0
python train99.py --version=33 --batch_size=32 --lr=1e-3 --model=se_resnet34_plus --sample=0 --weight=norm_1_log --norm=0 --pseudo=1 --train=0 --val=0
python train99.py --version=44 --batch_size=32 --lr=1e-3 --model=se_resnet34_plus3 --sample=0 --weight=norm_1_log --norm=0 --pseudo=1 --train=0 --val=0
python train99.py --version=45 --epoch=25 --batch_size=32 --lr=1e-3 --model=se_resnet34_plus --sample=0 --weight=norm_1_log --norm=0 --pseudo=1 --one=1 --train=0 --val=0
python merge.py
