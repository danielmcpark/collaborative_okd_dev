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
if [ "$arg_data" = "cifar100" ];
then
    if [ "$arg_arch" = "NetBasedOurs" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=110 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'nbo_resnet110_cifar100_201003_GAP_JSD_FD_gamma0.1' \
                --tracefile_train_1='nbo_resnet110_train_201003_GAP_JSD_FD_gamma0.1_stu_cifar100.csv' \
                --tracefile_train_2='nbo_resnet110_train_201003_GAP_JSD_FD_gamma0.1_en_cifar100.csv' \
                --tracefile_test_1='nbo_resnet110_test_201003_GAP_JSD_FD_gamma0.1_stu_cifar100.csv' \
                --tracefile_test_2='nbo_resnet110_test_201003_GAP_JSD_FD_gamma0.1_en_cifar100.csv' \
                --tracefile_diversity='nbo_resnet110_201003_GAP_JSD_FD_gamma0.1_diversity_cifar100.csv' \
                --trace \
                --mine \
                --bottleneck \
                --coo_type='JSD' \
                --ajs \
                --fd \
                --gamma=0.1 \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "My_ResNet" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --depth=110 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --milestones 150 225 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'r110_cifar100_201028_2nd_p4_t1.5_g0.4' \
                --tracefile_train_1='r110_train_201028_2nd_p4_t1.5_g0.4_stu_cifar100.csv' \
                --tracefile_train_2='r110_train_201028_2nd_p4_t1.5_g0.4_en_cifar100.csv' \
                --tracefile_test_1='r110_test_201028_2nd_p4_t1.5_g0.4_stu_cifar100.csv' \
                --tracefile_test_2='r110_test_201028_2nd_p4_t1.5_g0.4_en_cifar100.csv' \
                --tracefile_diversity='r110_201028_2nd_p4_t1.5_g0.4_diversity_cifar100.csv' \
                --mine \
                --bottleneck \
                --coo_type='JSD' \
                --JSD_temp=1.5 \
                --FD_temp=1.5 \
                --ajs \
                --fd \
                --gamma=0.4 \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "ONE_ResNet" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=32 \
                --num_branches=32 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'one_resnet32_cifar100_peer32_throughput_mem' \
                --tracefile_train_1='one_resnet32_train_stu_cifar100_peer32.csv' \
                --tracefile_train_2='one_resnet32_train_en_cifar100_peer32.csv' \
                --tracefile_test_1='one_resnet32_test_stu_cifar100_peer32.csv' \
                --tracefile_test_2='one_resnet32_test_en_cifar100_peer32.csv' \
                --tracefile_tr_loss='one_resnet32_train_loss_peer32.csv' \
                --tracefile_diversity='one_resnet32_diversity_peer32.csv' \
                --tracefile_thrp='one_resnet32_cifar100_throughput_peer32.csv' \
                --tracefile_mem='one_resnet32_cifar100_mem_usage_peer32.csv' \
                --trace \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "OKDDip_ResNet" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=32 \
                --num_branches=16 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'okddip_resnet32_cifar100_peer16_throughput_mem' \
                --tracefile_train_1='okddip_resnet32_train_stu_cifar100_peer16.csv' \
                --tracefile_train_2='okddip_resnet32_train_en_cifar100_peer16.csv' \
                --tracefile_test_1='okddip_resnet32_test_stu_cifar100_peer16.csv' \
                --tracefile_test_2='okddip_resnet32_test_en_cifar100_peer16.csv' \
                --tracefile_tr_loss='okddip_resnet32_train_loss_peer16.csv' \
                --tracefile_diversity='okddip_resnet32_diversity_peer16.csv' \
                --tracefile_thrp='okddip_resnet32_cifar100_throughput_peer16.csv' \
                --tracefile_mem='okddip_resnet32_cifar100_mem_usage_peer16.csv' \
                --trace \
                --okd \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "DML" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${MOBILEPATH1}'dml_mbnv1_cifar100_peer4_throughput_mem' \
                --tracefile_train_1='dml_mbnv1_train_stu_cifar100_peer4.csv' \
                --tracefile_train_2='dml_mbnv1_train_en_cifar100_peer4.csv' \
                --tracefile_test_1='dml_mbnv1_test_stu_cifar100_peer4.csv' \
                --tracefile_test_2='dml_mbnv1_test_en_cifar100_peer4.csv' \
                --tracefile_tr_loss='dml_mbnv1_train_loss_peer4.csv' \
                --tracefile_diversity='dml_mbnv1_diversity_peer4.csv' \
                --tracefile_thrp='dml_mbnv1_cifar100_throughput_peer4.csv' \
                --tracefile_mem='dml_mbnv1_cifar100_mem_usage_peer4.csv' \
                --trace \
                --dml \
                --ngpu='cuda:0'
    fi
    if [ "$arg_arch" = "CLILR_ResNet" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=32 \
                --num_branches=32 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${RESPATH}'clilr_resnet32_cifar100_peer32_throughput_mem' \
                --tracefile_train_1='clilr_resnet32_train_stu_cifar100_peer32.csv' \
                --tracefile_train_2='clilr_resnet32_train_en_cifar100_peer32.csv' \
                --tracefile_test_1='clilr_resnet32_test_stu_cifar100_peer32.csv' \
                --tracefile_test_2='clilr_resnet32_test_en_cifar100_peer32.csv' \
                --tracefile_tr_loss='clilr_resnet32_train_cifar100_loss_peer32.csv' \
                --tracefile_diversity='clilr_resnet32_cifar100_diversity_peer32.csv' \
                --tracefile_thrp='clilr_resnet32_cifar100_throughput_peer32.csv' \
                --tracefile_mem='clilr_resnet32_cifar100_mem_usage_peer32.csv' \
                --trace \
                --bpscale \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "ResNet" ]
    then
        python3 main.py \
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
    fi
    if [ "$arg_arch" = "DenseNet" ]
    then
        python3 main.py \
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
    fi
    if [ "$arg_arch" = "DenseNet_OKDDip" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${DENSEPATH}'okddip_densenet40k12_cifar100_peer4_throughput_mem' \
                --tracefile_train_1='okddip_densenet40k12_train_stu_cifar100_peer4.csv' \
                --tracefile_train_2='okddip_densenet40k12_train_en_cifar100_peer4.csv' \
                --tracefile_test_1='okddip_densenet40k12_test_stu_cifar100_peer4.csv' \
                --tracefile_test_2='okddip_densenet40k12_test_en_cifar100_peer4.csv' \
                --tracefile_tr_loss='okddip_densenet40k12_train_loss_peer4.csv' \
                --tracefile_diversity='okddip_densenet40k12_diversity_peer4.csv' \
                --tracefile_thrp='okddip_densenet40k12_cifar100_throughput_peer4.csv' \
                --tracefile_mem='okddip_densenet40k12_cifar100_mem_usage_peer4.csv' \
                --trace \
                --okd \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "DenseNet_CLILR" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch="DenseNet_ONEILR" \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${DENSEPATH}'clilr_densenet40k12_cifar100_peer4_throughput_mem' \
                --tracefile_train_1='clilr_densenet40k12_train_stu_cifar100_peer4.csv' \
                --tracefile_train_2='clilr_densenet40k12_train_en_cifar100_peer4.csv' \
                --tracefile_test_1='clilr_densenet40k12_test_stu_cifar100_peer4.csv' \
                --tracefile_test_2='clilr_densenet40k12_test_en_cifar100_peer4.csv' \
                --tracefile_tr_loss='clilr_densenet40k12_train_cifar100_loss_peer4.csv' \
                --tracefile_diversity='clilr_densenet40k12_cifar100_diversity_peer4.csv' \
                --tracefile_thrp='clilr_densenet40k12_cifar100_throughput_peer4.csv' \
                --tracefile_mem='clilr_densenet40k12_cifar100_mem_usage_peer4.csv' \
                --trace \
                --bpscale \
                --ngpu='cuda:2'
    fi
    if [ "$arg_arch" = "DenseNet_ONE" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch="DenseNet_ONEILR" \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${DENSEPATH}'one_densenet40k12_cifar100_peer4_throughput_mem' \
                --tracefile_train_1='one_densenet40k12_train_stu_cifar100_peer4.csv' \
                --tracefile_train_2='one_densenet40k12_train_en_cifar100_peer4.csv' \
                --tracefile_test_1='one_densenet40k12_test_stu_cifar100_peer4.csv' \
                --tracefile_test_2='one_densenet40k12_test_en_cifar100_peer4.csv' \
                --tracefile_tr_loss='one_densenet40k12_train_loss_peer4.csv' \
                --tracefile_diversity='one_densenet40k12_diversity_peer4.csv' \
                --tracefile_thrp='one_densenet40k12_cifar100_throughput_peer4.csv' \
                --tracefile_mem='one_densenet40k12_cifar100_mem_usage_peer4.csv' \
                --trace \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "My_DenseNet" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${DENSEPATH}'my_densenet40k12_cifar100_200904_GAP_aJS_FD_BS_gamma0.9' \
                --tracefile_train_1='my_densenet40k12_train_200904_GAP_aJS_FD_BS_gamma0.9_stu_cifar100.csv' \
                --tracefile_train_2='my_densenet40k12_train_200904_GAP_aJS_FD_BS_gamma0.9_en_cifar100.csv' \
                --tracefile_test_1='my_densenet40k12_test_200904_GAP_aJS_FD_BS_gamma0.9_stu_cifar100.csv' \
                --tracefile_test_2='my_densenet40k12_test_200904_GAP_aJS_FD_BS_gamma0.9_en_cifar100.csv' \
                --tracefile_diversity='my_densenet40k12_200904_GAP_aJS_FD_BS_gamma0.9_diversity_cifar100.csv' \
                --trace \
                --mine \
                --ajs \
                --fd \
                --sim \
                --gamma=0.9 \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "ONE_VGG" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${VGGPATH}'one_vgg16_cifar100_peer4_throughput_mem' \
                --tracefile_train_1='one_vgg16_train_stu_cifar100_peer4.csv' \
                --tracefile_train_2='one_vgg16_train_en_cifar100_peer4.csv' \
                --tracefile_test_1='one_vgg16_test_stu_cifar100_peer4.csv' \
                --tracefile_test_2='one_vgg16_test_en_cifar100_peer4.csv' \
                --tracefile_tr_loss='one_vgg16_train_loss_peer4.csv' \
                --tracefile_diversity='one_vgg16_diversity_peer4.csv' \
                --tracefile_thrp='one_vgg16_cifar100_throughput_peer4.csv' \
                --tracefile_mem='one_vgg16_cifar100_mem_usage_peer4.csv' \
                --trace \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "OKDDip_VGG" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${VGGPATH}'okddip_vgg16_cifar100_peer4_throughput_mem' \
                --tracefile_train_1='okddip_vgg16_train_stu_cifar100_peer4.csv' \
                --tracefile_train_2='okddip_vgg16_train_en_cifar100_peer4.csv' \
                --tracefile_test_1='okddip_vgg16_test_stu_cifar100_peer4.csv' \
                --tracefile_test_2='okddip_vgg16_test_en_cifar100_peer4.csv' \
                --tracefile_tr_loss='okddip_vgg16_train_loss_peer4.csv' \
                --tracefile_diversity='okddip_vgg16_diversity_peer4.csv' \
                --tracefile_thrp='okddip_vgg16_cifar100_throughput_peer4.csv' \
                --tracefile_mem='okddip_vgg16_cifar100_mem_usage_peer4.csv' \
                --trace \
                --okd \
                --ngpu='cuda:2'
    fi
    if [ "$arg_arch" = "CLILR_VGG" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${VGGPATH}'clilr_vgg16_cifar100_peer4_throughput_mem' \
                --tracefile_train_1='clilr_vgg16_train_stu_cifar100_peer4.csv' \
                --tracefile_train_2='clilr_vgg16_train_en_cifar100_peer4.csv' \
                --tracefile_test_1='clilr_vgg16_test_stu_cifar100_peer4.csv' \
                --tracefile_test_2='clilr_vgg16_test_en_cifar100_peer4.csv' \
                --tracefile_tr_loss='clilr_vgg16_train_loss_peer4.csv' \
                --tracefile_diversity='clilr_vgg16_diversity_peer4.csv' \
                --tracefile_thrp='clilr_vgg16_cifar100_throughput_peer4.csv' \
                --tracefile_mem='clilr_vgg16_cifar100_mem_usage_peer4.csv' \
                --trace \
                --bpscale \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "My_VGG" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --depth=16 \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${VGGPATH}'my_vgg16_cifar100_200928_GAP_aJS_FD_BS_gamma1.0' \
                --tracefile_train_1='my_vgg16_train_200928_GAP_aJS_FD_BS_gamma1.0_stu_cifar100.csv' \
                --tracefile_train_2='my_vgg16_train_200928_GAP_aJS_FD_BS_gamma1.0_en_cifar100.csv' \
                --tracefile_test_1='my_vgg16_test_200928_GAP_aJS_FD_BS_gamma1.0_stu_cifar100.csv' \
                --tracefile_test_2='my_vgg16_test_200928_GAP_aJS_FD_BS_gamma1.0_en_cifar100.csv' \
                --tracefile_diversity='my_vgg16_200928_GAP_aJS_FD_BS_gamma1.0_diversity_cifar100.csv' \
                --mine \
                --coo_type='JSD' \
                --ajs \
                --fd \
                --sim \
                --gamma=1.0 \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "My_MobileNetV1" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --num_branches=8 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --milestones 150 225 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${MOBILEPATH1}'mbnv1_201029_p8_t1.3_g0.0_cifar100' \
                --tracefile_train_1='mbnv1_201029_train_p8_t1.3_g0.0_stu_cifar100.csv' \
                --tracefile_train_2='mbnv1_201029_train_p8_t1.3_g0.0_en_cifar100.csv' \
                --tracefile_test_1='mbnv1_201029_test_p8_t1.3_g0.0_stu_cifar100.csv' \
                --tracefile_test_2='mbnv1_201029_test_p8_t1.3_g0.0_en_cifar100.csv' \
                --tracefile_diversity='mbnv1_201029_p8_t1.3_g0.0_diversity_cifar100.csv' \
                --you_want_to_save \
                --mine \
                --coo_type='JSD' \
                --JSD_temp=1.5 \
                --FD_temp=1.3 \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:3'
    fi
    if [ "$arg_arch" = "ONE_MobileNetV1" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${MOBILEPATH1}'one_mbnv1_cifar100_201006' \
                --tracefile_train_1='one_mbnv1_train_201006_stu_peer4_cifar100.csv' \
                --tracefile_train_2='one_mbnv1_train_201006_en_peer4_cifar100.csv' \
                --tracefile_test_1='one_mbnv1_test_201006_stu_peer4_cifar100.csv' \
                --tracefile_test_2='one_mbnv1_test_201006_en_peer4_cifar100.csv' \
                --tracefile_diversity='one_mbnv1_201006_diversity_peer4_cifar100.csv' \
                --trace \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "CLILR_MobileNetV1" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${MOBILEPATH1}'clilr_mbnv1_cifar100_201006' \
                --tracefile_train_1='clilr_mbnv1_train_201006_stu_peer4_cifar100.csv' \
                --tracefile_train_2='clilr_mbnv1_train_201006_en_peer4_cifar100.csv' \
                --tracefile_test_1='clilr_mbnv1_test_201006_stu_peer4_cifar100.csv' \
                --tracefile_test_2='clilr_mbnv1_test_201006_en_peer4_cifar100.csv' \
                --tracefile_diversity='clilr_mbnv1_201006_diversity_peer4_cifar100.csv' \
                --trace \
                --bpscale \
                --ngpu='cuda:2'
    fi
    if [ "$arg_arch" = "OKDDip_MobileNetV1" ]
    then
        python3 main.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --num_branches=4 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --log-interval=$[50000/128+1] \
                --data=${CIFAR100} \
                --save=${MOBILEPATH1}'okddip_mbnv1_cifar100_201006' \
                --tracefile_train_1='okddip_mbnv1_train_201006_stu_peer4_cifar100.csv' \
                --tracefile_train_2='okddip_mbnv1_train_201006_en_peer4_cifar100.csv' \
                --tracefile_test_1='okddip_mbnv1_test_201006_stu_peer4_cifar100.csv' \
                --tracefile_test_2='okddip_mbnv1_test_201006_en_peer4_cifar100.csv' \
                --tracefile_diversity='okddip_mbnv1_201006_diversity_peer4_cifar100.csv' \
                --trace \
                --okd \
                --ngpu='cuda:3'
    fi
fi

# Cub200
if [ "$arg_data" = "cub200" ];
then
   if [ "$arg_arch" = "resnet50_img" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${CUB200} \
                --save=${RESPATH}'ResNet50_baseline_vanilla_cub200' \
                --tracefile_train_1='resnet50_train_vanilla_cub200.csv' \
                --tracefile_test_1='resnet50_test_vanilla_cub200.csv' \
                --tracefile_tr_loss='resnet50_train_loss_vanilla_cub200.csv' \
                --trace \
                --baseline \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "My_ResNet50" ]
    then
        python3 main.py \
                --dataset=cub200 \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --wd=4e-5 \
                --batch-size=64 \
                --test-batch-size=256 \
                --log-interval=$[5994/64+1] \
                --data=${CUB200} \
                --save=${RESPATH}'MyR50_201007_peer4_gamma0.0_cub200' \
                --tracefile_train_1='MyR50_201007_train_stu_peer4_gamma0.0_cub200.csv' \
                --tracefile_train_2='MyR50_201007_train_en_peer4_gamma0.0_cub200.csv' \
                --tracefile_test_1='MyR50_201007_test_stu_peer4_gamma0.0_cub200.csv' \
                --tracefile_test_2='MyR50_201007_test_en_peer4_gamma0.0_cub200.csv' \
                --tracefile_tr_loss='MyR50_201007_train_loss_peer4_gamma0.0_cub200.csv' \
                --tracefile_diversity='MyR50_201007_diversity_peer4_gamma0.0_cub200.csv' \
                --trace \
                --mine \
                --coo_type='JSD' \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:0'
    fi
   if [ "$arg_arch" = "MobileNet_V2" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${CUB200} \
                --save=${MOBILEPATH2}'mbnv2_baseline_init_cub200' \
                --tracefile_train_1='mbnv2_train_init_cub200.csv' \
                --tracefile_test_1='mbnv2_test_init_cub200.csv' \
                --tracefile_tr_loss='mbnv2_train_loss_init_cub200.csv' \
                --trace \
                --baseline \
                --ngpu='cuda:3'
    fi
fi

# Cars196
if [ "$arg_data" = "cars196" ];
then
   if [ "$arg_arch" = "resnet50_img" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${CARS196} \
                --save=${RESPATH}'ResNet50_baseline_pretrained_cars196' \
                --baseline \
                --pretrained \
                --ngpu='cuda:1'
    fi
    if [ "$arg_arch" = "My_ResNet50" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --wd=4e-5 \
                --batch-size=64 \
                --test-batch-size=256 \
                --log-interval=$[5994/64+1] \
                --data=${CARS196} \
                --save=${RESPATH}'MyR50_201007_peer4_gamma0.0_cars196' \
                --tracefile_train_1='MyR50_201007_train_stu_peer4_gamma0.0_cars196.csv' \
                --tracefile_train_2='MyR50_201007_train_en_peer4_gamma0.0_cars196.csv' \
                --tracefile_test_1='MyR50_201007_test_stu_peer4_gamma0.0_cars196.csv' \
                --tracefile_test_2='MyR50_201007_test_en_peer4_gamma0.0_cars196.csv' \
                --tracefile_tr_loss='MyR50_201007_train_loss_peer4_gamma0.0_cars196.csv' \
                --tracefile_diversity='MyR50_201007_diversity_peer4_gamma0.0_cars196.csv' \
                --trace \
                --mine \
                --coo_type='JSD' \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:0'
    fi
    if [ "$arg_arch" = "MobileNet_V2" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${CARS196} \
                --save=${MOBILEPATH2}'mbnv2_baseline_pre_cars196' \
                --tracefile_train_1='mbnv2_train_pre_cars196.csv' \
                --tracefile_test_1='mbnv2_test_pre_cars196.csv' \
                --tracefile_tr_loss='mbnv2_train_loss_pre_cars196.csv' \
                --trace \
                --baseline \
                --pretrained \
                --ngpu='cuda:2'
    fi
fi

# Dogs120
if [ "$arg_data" = "dogs120" ];
then
   if [ "$arg_arch" = "resnet50_img" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${DOGS120} \
                --save=${RESPATH}'ResNet50_baseline_pre_dogs120' \
                --tracefile_train_1='resnet50_train_pre_dogs120.csv' \
                --tracefile_test_1='resnet50_test_pre_dogs120.csv' \
                --tracefile_tr_loss='resnet50_train_loss_pre_dogs120.csv' \
                --baseline \
                --pretrained \
                --ngpu='cuda:0'
    fi
    if [ "$arg_arch" = "My_ResNet50" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --wd=4e-5 \
                --batch-size=16 \
                --test-batch-size=256 \
                --log-interval=$[5994/64+1] \
                --data=${DOGS120} \
                --save=${RESPATH}'MyR50_201007_peer4_gamma0.0_dogs120' \
                --tracefile_train_1='MyR50_201007_train_stu_peer4_gamma0.0_dogs120.csv' \
                --tracefile_train_2='MyR50_201007_train_en_peer4_gamma0.0_dogs120.csv' \
                --tracefile_test_1='MyR50_201007_test_stu_peer4_gamma0.0_dogs120.csv' \
                --tracefile_test_2='MyR50_201007_test_en_peer4_gamma0.0_dogs120.csv' \
                --tracefile_tr_loss='MyR50_201007_train_loss_peer4_gamma0.0_dogs120.csv' \
                --tracefile_diversity='MyR50_201007_diversity_peer4_gamma0.0_dogs120.csv' \
                --trace \
                --mine \
                --coo_type='JSD' \
                --ajs \
                --fd \
                --gamma=0.0 \
                --ngpu='cuda:0'
    fi
    if [ "$arg_arch" = "MobileNet_V2" ]
    then
        python3 main.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.001 \
                --epochs=150 \
                --wd=1e-4 \
                --batch-size=16 \
                --test-batch-size=256 \
                --data=${DOGS120} \
                --save=${MOBILEPATH2}'mbnv2_baseline_pre_dogs120' \
                --tracefile_train_1='mbnv2_train_pre_dogs120.csv' \
                --tracefile_test_1='mbnv2_test_pre_dogs120.csv' \
                --tracefile_tr_loss='mbnv2_train_loss_pre_dogs120.csv' \
                --trace \
                --baseline \
                --pretrained
                --ngpu='cuda:0'
    fi
fi


# Tiny-ImageNet
if [ "$arg_data" = "tiny-imagenet" ];
then
    if [ "$arg_arch" = "resnet18" ]
    then
        python3 main.py \
                --dataset=tiny-imagenet \
                --arch=$arg_arch \
                --lr=0.1 \
                --epochs=300 \
                --batch-size=128 \
                --test-batch-size=256 \
                --data=${TINY_IMAGENET} \
                --save=${RESPATH}'resnet18_original_tinyimagenet' \
                --tracefile_train_1='resnet18_training_stu_tinyimagenet.csv' \
                --tracefile_train_2='resnet18_training_en_tinyimagenet.csv' \
                --tracefile_test_1='resnet18_test_stu_tinyimagenet.csv' \
                --tracefile_test_2='resnet18_test_en_tinyimagenet.csv' \
                --tracefile_tr_loss='resnet18_train_loss_tinyimagenet.csv' \
                --tracefile_diversity='resnet18_diversity_tinyimagenet.csv' \
                --baseline \
                --log-interval=100 \
                --ngpu='cuda:0'
    fi
fi


