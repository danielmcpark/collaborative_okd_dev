# Mincheol Park Frameworks
# Main training shell scripts

# Datasets:   cifar10, cifar100, imagenet
# Benchmarks: vgg, resnet
# Epochs: 200
# optims: SGD
# Batch_size: 64 (cifar10), 128 (cifar100), 256 (imagenet)
# weight_decay: 1e-4(L2)

DISKPATH='/mnt/disk3/'
IMAGENET=${DISKPATH}'imagenet/'
CINIC10=${DISKPATH}'cinic10/'
TINY_IMAGENET=${DISKPATH}'tiny-imagenet-200/'
CIFAR10=${DISKPATH}'cifar10'
CIFAR100=${DISKPATH}'cifar100'
CUB200=${DISKPATH}'cub200'
CARS196=${DISKPATH}'cars196'
DOGS120=${DISKPATH}'dogs120'

LOGPATH=${DISKPATH}'logs/newkd/'
WRNPATH=${LOGPATH}'wideresnet/'
RESPATH=${LOGPATH}'resnet/'
VGGPATH=${LOGPATH}'vgg/'
SERESPATH=${LOGPATH}'seresnet/'
DENSEPATH=${LOGPATH}'densenet/'
MOBILEPATH1=${LOGPATH}'mobilenetv1/'
MOBILEPATH2=${LOGPATH}'mobilenetv2/'


PROGRAM_NAME=`/usr/bin/basename "$0"`
echo shell arg 0: $0
echo USING BASENAME: ${PROGRAM_NAME}
arg_data=default
arg_arch=default

function print_usage(){
/bin/cat << EOF
Usage:
    ${PROGRAM_NAME} [-d arg_data] [-a arg_arch]
Option:
    -d, dataset
    -a, model
EOF
}
if [ $# -eq 0 ];
then
    print_usage
    exit 1
fi

while getopts "d:a:h" opt
do
    case $opt in
        d) arg_data=$OPTARG; echo "ARG DATA: $arg_data";;
        a) arg_arch=$OPTARG; echo "ARG ARCH: $arg_arch";;
        h) print_usage;;
    esac
done

# CIFAR-100
if [ "$arg_data" = "cifar100" ]; then
    if [ "$arg_arch" = "NetBasedOurs" ]; then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
		--nb_arch='ResNet' \
                --depth=32 \
                --num_branches=8 \
                --epochs=300 \
		--milestones 150 225 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'nbrm32_201204_pcsc_p8_t1.5_g0.0_corr16.0_sig_gs0_stifleCE2.0_cifar100' \
                --tracefile_pearson='nbrm32_201204_pcsc_p8_t1.5_g0.0_corr16.0_sig_gs0_stifleCE2.0_pearson_cifar100.csv' \
                --tracefile_spearman='nbrm32_201204_pcsc_p8_t1.5_g0.0_corr16.0_sig_gs0_stifleCE2.0_spearman_cifar100.csv' \
                --gate_coeff='nbrm32_201204_pcsc_p8_t1.5_g0.0_corr16.0_sig_gs0_stifleCE2.0_gateweight_cifar100.pth.tar' \
                --coo_type='JSD' \
                --slope=16.0 \
                --JSD_temp=1.5 \
                --FD_temp=1.5 \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:3'
    elif [ "$arg_arch" = "My_ResNetV2" ]; then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --depth=32 \
                --num_branches=8 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --milestones 150 225 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'rm32_201208_pcsc_p8_t1.5_g0.0_corr2.0_sig_gs0_stifleCE2.0_cifar100' \
                --tracefile_pearson='rm32_201208_pcsc_p8_t1.5_g0.0_corr2.0_sig_gs0_stifleCE2.0_pearson_cifar100.csv' \
                --tracefile_spearman='rm32_201208_pcsc_p8_t1.5_g0.0_corr2.0_sig_gs0_stifleCE2.0_spearman_cifar100.csv' \
                --gate_coeff='rm32_201208_pcsc_p8_t1.5_g0.0_corr2.0_sig_gs0_stifleCE2.0_gateweight_cifar100.pth.tar' \
                --coo_type='JSD' \
                --slope=2.0 \
                --stifle_ce=2.0 \
                --JSD_temp=1.5 \
                --FD_temp=1.5 \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:3'
    elif [ "$arg_arch" = "My_ResNetO" ]; then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --depth=32 \
                --num_branches=8 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'ro32_201124_pcsc_p8_t1.5_g0.0_corr0.0_sig_gs0_cifar100' \
                --tracefile_pearson='ro32_201124_pcsc_p8_t1.5_g0.0_corr0.0_sig_gs0_pearson_cifar100.csv' \
                --tracefile_spearman='ro32_201124_pcsc_p8_t1.5_g0.0_cor0.0_sig_gs0_spearman_cifar100.csv' \
                --gate_coeff='ro32_201124_pcsc_p8_t1.5_g0.0_corr0.0_sig_gs0_gateweight_cifar100.pth.tar' \
		--coo_type='JSD' \
                --slope=0.0 \
		--JSD_temp=1.5 \
		--FD_temp=1.5 \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:2'
    elif [ "$arg_arch" = "ResNet" ]; then
        python3 pcsc.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=32 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=10 \
                --data=${CIFAR100} \
                --save=${RESPATH}'resnet32_cifar100_speedtest' \
                --tracefile_train_1='resnet32_train_cifar100.csv' \
                --tracefile_test_1='resnet32_test_cifar100.csv' \
                --tracefile_tr_loss='resnet32_train_loss.csv' \
                --baseline \
                --ngpu='cuda:0'
    elif [ "$arg_arch" = "DenseNet" ]; then
        python3 pcsc.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=32 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --data=${CIFAR100} \
                --save=${DENSEPATH}'densenet40k12_cifar100_200902' \
                --tracefile_train_1='densenet40k12_train_200902_cifar100.csv' \
                --tracefile_test_1='densenet40k12_test_200902_cifar100.csv' \
                --tracefile_tr_loss='densenet40k12_train_loss.csv' \
                --baseline \
                --ngpu='cuda:2'
    elif [ "$arg_arch" = "My_DenseNet" ]; then
        python3 pcsc.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${DENSEPATH}'my_densenet40k12_cifar100_200910_GAP_aJS_FD_BS_gamma1.0' \
                --tracefile_train_1='my_densenet40k12_train_200910_GAP_aJS_FD_BS_gamma1.0_stu_cifar100.csv' \
                --tracefile_train_2='my_densenet40k12_train_200910_GAP_aJS_FD_BS_gamma1.0_en_cifar100.csv' \
                --tracefile_test_1='my_densenet40k12_test_200910_GAP_aJS_FD_BS_gamma1.0_stu_cifar100.csv' \
                --tracefile_test_2='my_densenet40k12_test_200910_GAP_aJS_FD_BS_gamma1.0_en_cifar100.csv' \
                --tracefile_diversity='my_densenet40k12_200910_GAP_aJS_FD_BS_gamma1.0_diversity_cifar100.csv' \
                --trace \
                --mine \
                --ajs \
                --fd \
		--sim \
                --gamma=1.0 \
                --ngpu='cuda:2'
    elif [ "$arg_arch" = "WideResNet" ]; then
        python3 pcsc.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
		--wf=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=10 \
                --data=${CIFAR100} \
                --save=${WRNPATH}'wrn16_4_cifar100' \
                --tracefile_train_1='wrn16_4_train_cifar100.csv' \
                --tracefile_test_1='wrn16_4_test_cifar100.csv' \
                --tracefile_tr_loss='wrn16_4_train_loss.csv' \
                --baseline \
                --ngpu='cuda:2'
    elif [ "$arg_arch" = "My_WideResNet" ]; then
        python3 pcsc.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
		--wf=4 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${WRNPATH}'my_wrn16_4_cifar100_200922_ENL_aJS_FD_BS_gamma1.0' \
                --tracefile_train_1='my_wrn16_4_train_200922_ENL_aJS_FD_BS_gamma1.0_stu_cifar100.csv' \
                --tracefile_train_2='my_wrn16_4_train_200922_ENL_aJS_FD_BS_gamma1.0_en_cifar100.csv' \
                --tracefile_test_1='my_wrn16_4_test_200922_ENL_aJS_FD_BS_gamma1.0_stu_cifar100.csv' \
                --tracefile_test_2='my_wrn16_4_test_200922_ENL_aJS_FD_BS_gamma1.0_en_cifar100.csv' \
                --tracefile_diversity='my_wrn16_4_200922_ENL_aJS_FD_BS_gamma1.0_diversity_cifar100.csv' \
                --trace \
                --mine \
		--nl \
		--embedding \
                --ajs \
                --fd \
		--sim \
                --gamma=1.0 \
                --ngpu='cuda:0'
    elif [ "$arg_arch" = "My_VGG" ]; then
        python3 pcsc.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${VGGPATH}'my_vgg16_cifar100_200911_GAP_aJS_FD_gamma0.8' \
                --tracefile_train_1='my_vgg16_train_200911_GAP_aJS_FD_gamma0.8_stu_cifar100.csv' \
                --tracefile_train_2='my_vgg16_train_200911_GAP_aJS_FD_gamma0.8_en_cifar100.csv' \
                --tracefile_test_1='my_vgg16_test_200911_GAP_aJS_FD_gamma0.8_stu_cifar100.csv' \
                --tracefile_test_2='my_vgg16_test_200911_GAP_aJS_FD_gamma0.8_en_cifar100.csv' \
                --tracefile_diversity='my_vgg16_200911_GAP_aJS_FD_gamma0.8_diversity_cifar100.csv' \
                --trace \
                --mine \
                --ajs \
                --fd \
                --gamma=0.8 \
                --ngpu='cuda:0'
    elif [ "$arg_arch" = "VGG" ]; then
	python3 pcsc.py \
		--dataset=cifar100 \
		--arch=$arg_arch \
		--depth=16 \
		--epochs=300 \
		--batch-size=128 \
		--test-batch-size=256 \
		--data=${CIFAR100} \
		--save=${VGGPATH}'vgg16_cifar100' \
		--tracefile_train_1='vgg16_train_cifar100.csv' \
		--tracefile_test_1='vgg16_test_cifar100.csv' \
		--tracefile_tr_loss='vgg16_train_loss.csv' \
		--baseline \
		--ngpu='cuda:0'
    else
        echo "0"
    fi
fi

# Cub200
if [ "$arg_data" = "cub200" ];
then
    if [ "$arg_arch" = "resnet50_img" ]
    then
	python3 pcsc.py \
		--dataset=$arg_data \
		--arch=$arg_arch \
		--lr=0.001 \
		--epochs=150 \
		--wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${CUB200} \
		--save=${RESPATH}'ResNet50_pre_baseline_cub200' \
		--baseline \
		--pretrained \
		--ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "My_ResNet50" ]
    then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=300 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
		--num_branches=4 \
		--milestones 150 225 \
		--consistency_rampup=80 \
                --log-interval=$[5994/16+1] \
                --data=${CUB200} \
		--save=${RESPATH}'MyR50_201022_pre_p4_t2.0_g0.01_e300_b16_cub200' \
		--tracefile_train_1='MyR50_201022_pre_train_stu_p4_t2.0_g0.01_e300_b16_cub200.csv' \
		--tracefile_train_2='MyR50_201022_pre_train_en_p4_t2.0_g0.01_e300_b16_cub200.csv' \
		--tracefile_test_1='MyR50_201022_pre_test_stu_p4_t2.0_g0.01_e300_b16_cub200.csv' \
		--tracefile_test_2='MyR50_201022_pre_test_en_p4_t2.0_g0.01_e300_b16_cub200.csv' \
		--tracefile_tr_loss='MyR50_201022_pre_train_loss_p4_t2.0_g0.01_e300_b16_cub200.csv' \
		--tracefile_diversity='MyR50_201022_pre_diversity_p4_t2.0_g0.01_e300_b16_cub200.csv' \
                --trace \
                --mine \
		--pretrained \
                --coo_type='JSD' \
		--JSD_temp=2.0 \
		--FD_temp=2.0 \
                --ajs \
                --fd \
                --gamma=0.01 \
                --ngpu='cuda:2'
    fi
    if [ "$arg_arch" = "NetBasedOurs" ]
    then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
		--nb_arch='resnet50_img' \
                --depth=50 \
                --num_branches=4 \
                --epochs=150 \
		--lr=0.001 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=64 \
		--consistency_rampup=40 \
                --log-interval=$[5994/16+1] \
                --data=${CUB200} \
                --save=${RESPATH}'nbo_resnet50_cub200_201005_init_GAP_aJS_FD_gamma0.0' \
                --tracefile_train_1='nbo_resnet50_train_201005_init_GAP_aJS_FD_gamma0.0_stu_cub200.csv' \
                --tracefile_train_2='nbo_resnet50_train_201005_init_GAP_aJS_FD_gamma0.0_en_cub200.csv' \
                --tracefile_test_1='nbo_resnet50_test_201005_init_GAP_aJS_FD_gamma0.0_stu_cub200.csv' \
                --tracefile_test_2='nbo_resnet50_test_201005_init_GAP_aJS_FD_gamma0.0_en_cub200.csv' \
                --tracefile_diversity='nbo_resnet50_201005_init_GAP_aJS_FD_gamma0.0_diversity_cub200.csv' \
                --trace \
                --mine \
		--coo_type='JSD' \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:1'
    fi
fi

# Cub200
if [ "$arg_data" = "cars196" ];
then
    if [ "$arg_arch" = "resnet50_img" ]
    then
	python3 pcsc.py \
		--dataset=$arg_data \
		--arch=$arg_arch \
		--lr=0.001 \
		--epochs=150 \
		--wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${CARS196} \
		--save=${RESPATH}'ResNet50_pre_baseline_cars196' \
		--baseline \
		--pretrained \
		--ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "My_ResNet50" ]
    then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
		--num_branches=4 \
		--consistency_rampup=40 \
                --log-interval=$[8144/16+1] \
                --data=${CARS196} \
                --save=${RESPATH}'MyR50_201008_init_peer4_gamma0.4_cars196' \
                --tracefile_train_1='MyR50_201008_init_train_stu_peer4_gamma0.4_cars196.csv' \
                --tracefile_train_2='MyR50_201008_init_train_en_peer4_gamma0.4_cars196.csv' \
                --tracefile_test_1='MyR50_201008_init_test_stu_peer4_gamma0.4_cars196.csv' \
                --tracefile_test_2='MyR50_201008_init_test_en_peer4_gamma0.4_cars196.csv' \
                --tracefile_tr_loss='MyR50_201008_init_train_loss_peer4_gamma0.4_cars196.csv' \
                --tracefile_diversity='MyR50_201008_init_diversity_peer4_gamma0.4_cars196.csv' \
                --trace \
                --mine \
		--pretrained \
                --coo_type='JSD' \
                --ajs \
                --fd \
                --gamma=0.4 \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "NetBasedOurs" ]
    then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
		--nb_arch='resnet50_img' \
                --depth=50 \
                --num_branches=4 \
                --epochs=200 \
		--lr=0.001 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
		--milestones 100 150 \
		--consistency_rampup=53 \
                --log-interval=$[8144/16+1] \
                --data=${CARS196} \
                --save=${RESPATH}'NBr50_201014_init_p4_t1.5_g1.0_e200_cars196' \
                --tracefile_train_1='NBr50_train_201014_init_p4_t1.5_g1.0_e200_stu_cars196.csv' \
                --tracefile_train_2='NBr50_train_201014_init_p4_t1.5_g1.0_e200_en_cars196.csv' \
                --tracefile_test_1='NBr50_test_201014_init_p4_t1.5_g1.0_e200_stu_cars196.csv' \
                --tracefile_test_2='NBr50_test_201014_init_p4_t1.5_g1.0_e200_en_cars196.csv' \
                --tracefile_diversity='NBr50_201014_init_p4_t1.5_g1.0_e200_diversity_cars196.csv' \
                --trace \
                --mine \
		--coo_type='JSD' \
		--JSD_temp=1.5 \
		--FD_temp=1.5 \
                --ajs \
                --fd \
                --gamma=1.0 \
                --ngpu='cuda:3'
    fi
fi

# Dogs120
if [ "$arg_data" = "dogs120" ];
then
    if [ "$arg_arch" = "resnet50_img" ]
    then
	python3 pcsc.py \
		--dataset=$arg_data \
		--arch=$arg_arch \
		--lr=0.001 \
		--epochs=150 \
		--wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${DOGS120} \
		--save=${RESPATH}'ResNet50_pre_baseline_dogs120' \
		--baseline \
		--pretrained \
		--ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "My_ResNet50" ]
    then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
		--num_branches=4 \
		--milestones 60 100 \
		--consistency_rampup=40 \
                --log-interval=$[8144/16+1] \
                --data=${DOGS120} \
                --save=${RESPATH}'MyR50_201008_init_peer4_gamma0.1_dogs120' \
                --tracefile_train_1='MyR50_201008_init_train_stu_peer4_gamma0.1_dogs120.csv' \
                --tracefile_train_2='MyR50_201008_init_train_en_peer4_gamma0.1_dogs120.csv' \
                --tracefile_test_1='MyR50_201008_init_test_stu_peer4_gamma0.1_dogs120.csv' \
                --tracefile_test_2='MyR50_201008_init_test_en_peer4_gamma0.1_dogs120.csv' \
                --tracefile_tr_loss='MyR50_201008_init_train_loss_peer4_gamma0.1_dogs120.csv' \
                --tracefile_diversity='MyR50_201008_init_diversity_peer4_gamma0.1_dogs120.csv' \
                --trace \
                --mine \
		--pretrained \
                --coo_type='JSD' \
                --ajs \
                --fd \
                --gamma=0.1 \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "NetBasedOurs" ]
    then
        python3 pcsc.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
		--nb_arch='resnet50_img' \
                --depth=50 \
                --num_branches=4 \
                --epochs=150 \
		--lr=0.001 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
		--consistency_rampup=40 \
		--milestones 60 100 \
                --log-interval=$[8144/16+1] \
                --data=${DOGS120} \
                --save=${RESPATH}'nbo_resnet50_dogs120_201014_pre_GAP_aJS_FD_gamma1.0' \
                --tracefile_train_1='nbo_resnet50_train_201014_pre_GAP_aJS_FD_gamma1.0_stu_dogs120.csv' \
                --tracefile_train_2='nbo_resnet50_train_201014_pre_GAP_aJS_FD_gamma1.0_en_dogs120.csv' \
                --tracefile_test_1='nbo_resnet50_test_201014_pre_GAP_aJS_FD_gamma1.0_stu_dogs120.csv' \
                --tracefile_test_2='nbo_resnet50_test_201014_pre_GAP_aJS_FD_gamma1.0_en_dogs120.csv' \
                --tracefile_diversity='nbo_resnet50_201014_pre_GAP_aJS_FD_gamma1.0_diversity_dogs120.csv' \
                --trace \
                --mine \
		--pretrained \
		--coo_type='JSD' \
		--JSD_temp=3.0 \
		--FD_temp=5.0 \
                --ajs \
                --fd \
                --gamma=1.0 \
                --ngpu='cuda:1'
    fi
fi
